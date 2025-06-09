import json
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numba as nb
import numpy as np
import polars as pl
from polars import type_aliases as tap


@dataclass
class RangeCompressedMask:
    w: int
    h: int

    encodings: np.ndarray
    "表达区间用，shape: (n, 4)，[(start_x, end_x, value, row_index), ...]"

    row_indexes: np.ndarray
    "快速从 y 查找编码行，shape: (n, 2)，[(start_y, encoding_count), ...]"

    def save(
        self,
        base_dir: Union[str, PathLike],
        compression: tap.ParquetCompression = "gzip",
    ):
        base_dir = Path(base_dir)

        base_dir.mkdir(exist_ok=True)
        dfe = pl.from_numpy(self.encodings, schema=["start", "end", "v", "row_index"])
        dfe.write_parquet(base_dir / f"encodings.parquet", compression=compression)

        dfer = pl.from_numpy(self.row_indexes, schema=["row_start_index", "row_count"])
        dfer.write_parquet(base_dir / f"row_indexes.parquet", compression=compression)

        (base_dir / "meta.json").write_text(
            json.dumps(
                {
                    "w": self.w,
                    "h": self.h,
                    "datetime": datetime.now(),
                },
                default=str,
            )
        )

    def three_columns_encodings(self, try_contiguous=True):
        res = self.encodings[:, :3]
        if try_contiguous:
            res = np.ascontiguousarray(res)
        return res

    @staticmethod
    def load(
        base_dir: Union[str, PathLike], chip: Optional[str] = None, no_row_index=True
    ):
        """从文件夹中导入

        base_dir: 文件夹路径
        chip: 如果有 chip，那么会在 base_dir/chip 中寻找文件
        no_row_index: 原始 encodings 是四列的，最后一列是 row_index，
                      如果不需要 row_index，可以设置为 True，可以把更多数据放到缓存里
        """
        base_dir = Path(base_dir)

        if chip is not None:
            base_dir = base_dir / chip

        dfe = pl.read_parquet(base_dir / f"encodings.parquet")
        encodings = dfe.to_numpy()[:, :3]
        if no_row_index:
            encodings = np.ascontiguousarray(encodings[:, :3])
        dfer = pl.read_parquet(base_dir / f"row_indexes.parquet")
        row_indexes = dfer.to_numpy()

        meta = json.loads((base_dir / "meta.json").read_text())
        return RangeCompressedMask(
            w=meta["w"],
            h=meta["h"],
            encodings=encodings,
            row_indexes=row_indexes,
        )

    @staticmethod
    def targets(
        base_dir: Union[str, PathLike, None] = None, chip: Optional[str] = None
    ):
        base = [
            "encodings.parquet",
            "row_indexes.parquet",
            "meta.json",
        ]

        if base_dir is not None:
            base_dir = Path(base_dir)
            if chip is not None:
                base_dir = base_dir / chip
            base = [base_dir / x for x in base]

        return base

    def find_index(self, X: np.ndarray, Y: np.ndarray, binary_search=False):
        if self.encodings.shape[1] == 4:
            import warnings

            warnings.warn(f"`row_indexes` has 4 columns.")
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)

        args = (self.row_indexes, self.encodings, X, Y)
        if binary_search:
            return _find_index_binary(*args)
        else:
            return _find_index(*args)

    def to_mask(self):
        return to_mask(self.three_columns_encodings(), self.row_indexes, self.w, self.h)

    def calc_area(self):
        return calc_area_from_encodings(self.encodings, self.row_indexes)


@nb.njit
def _mask_encode(mask):
    n_rows, n_cols = mask.shape

    encodings = []
    row_indexes = []

    for row_index in nb.prange(n_rows):
        row = mask[row_index, :]
        last_i = 0
        last_v = row[last_i]

        start_len = len(encodings)

        for i in range(1, n_cols):
            v = row[i]
            # print(f'[{row_index}] ({last_i}, {i}) {v=} {last_v=}')
            if v != last_v:
                if last_v != 0:
                    encodings.append((last_i, i - 1, last_v, row_index))
                    # print('append')
                last_v = v
                last_i = i

        encodings.append((last_i, len(row) - 1, last_v, row_index))
        # print('append')
        row_indexes.append((start_len, len(encodings) - start_len))

    encodings_np = np.array(encodings, dtype="int32")
    row_indexes_np = np.array(row_indexes, dtype="int32")
    # 直接在返回中转换为np.array，numba无法编译

    return encodings_np, row_indexes_np


def mask_encode(mask: np.ndarray):
    """把 mask 编码为区间压缩。"""

    h, w = mask.shape
    encodings_np, row_indexes_np = _mask_encode(mask)
    return RangeCompressedMask(
        w=w, h=h, encodings=encodings_np, row_indexes=row_indexes_np
    )


@nb.njit
def find_encoding_in_row(row_encodings, col):
    """如果要直接调用此函数，应当手动保证 col 不要在非法值域"""
    if len(row_encodings) == 1 and row_encodings[0, 0] == 0:
        return 0
    if row_encodings[0][0] > col:
        # 由于总是从小到大匹配的，先试试第一个，之后就不用在循环里判断是否 start_index > col 了
        return 0
    # if row_encodings[-1][1] < col:
    #     # 但是不用判断最后一个，这种情况很少，也许会影响流水线
    #     return 0

    for start_index, stop_index, value in row_encodings:
        if start_index <= col <= stop_index:
            return value

    # 这个 return 0 用于处理 col > row_encodings[-1][1] 的情况
    return 0


@nb.njit
def find_encoding_in_row_binary(row_encodings, col):
    # 在细胞中，二分搜索不影响速度，在分区中，二分搜索拖慢 50%

    if len(row_encodings) == 1 and row_encodings[0, 0] == 0:
        return 0

    encoding_index = len(row_encodings) // 2
    upper_index = len(row_encodings)
    lower_index = 0

    while True:
        start_index, stop_index, value = row_encodings[encoding_index]

        if start_index <= col <= stop_index:
            return value

        if upper_index - lower_index < 1:
            return 0

        if start_index > col:
            upper_index = encoding_index
            encoding_index = (encoding_index + lower_index) // 2
        else:
            lower_index = encoding_index
            encoding_index = (encoding_index + upper_index) // 2


# TODO: `First-class function type feature is experimental` 不再警告时合并两函数
@nb.njit(parallel=True)
def _find_index(row_indexes, encodings_np, X, Y):
    out = np.zeros(len(X), dtype="int32")
    for i in nb.prange(len(X)):
        x, y = X[i], Y[i]
        row, col = y, x

        res = 0
        if x < 0 or y < 0:
            pass
        elif row >= len(row_indexes) or col >= encodings_np.shape[0]:
            pass
        else:
            start_index, length = row_indexes[row]
            row_encoding = encodings_np[start_index : start_index + length, :3]
            res = find_encoding_in_row(row_encoding, col)
        out[i] = res
    return out


@nb.njit(parallel=True)
def _find_index_binary(row_indexes, encodings_np, X, Y):
    out = np.zeros(len(X), dtype="int32")
    for i in nb.prange(len(X)):
        x, y = X[i], Y[i]
        row, col = y, x

        res = 0
        if x < 0 or y < 0:
            pass
        elif row >= len(row_indexes) or col >= encodings_np.shape[0]:
            pass
        else:
            start_index, length = row_indexes[row]
            row_encoding = encodings_np[start_index : start_index + length, :3]
            res = find_encoding_in_row_binary(row_encoding, col)
        out[i] = res
    return out


@nb.njit(parallel=True)
def to_mask(encodings: np.ndarray, row_indexes: np.ndarray, w: int, h: int):
    assert encodings.shape[1] == 3, "encodings should have 3 columns"

    out = np.zeros((h, w), dtype="int32")
    for row in nb.prange(h):
        start_index, length = row_indexes[row]
        row_encoding = encodings[start_index : start_index + length, :]
        for start, stop, value in row_encoding:
            out[row, start : stop + 1] = value
    return out


@nb.njit
def calc_area_from_encodings(encodings: np.ndarray, row_indexes: np.ndarray):
    """计算每一个分块儿的面积"""
    areas = {0: 0}

    for i in range(len(row_indexes)):
        start_index, length = row_indexes[i]
        row_encoding = encodings[start_index : start_index + length, :3]
        for start, stop, value in row_encoding:
            if value == 0:
                continue
            areas[value] = areas.get(value, 0) + (stop - start + 1)

    return areas


@nb.njit
def calc_area_from_mask(mask: np.ndarray):
    """计算每一个分块儿的面积"""
    areas = {0: {0: 0}}

    for i in range(nb.get_num_threads()):
        _areas = {}
        _areas[np.int64(0)] = np.int64(0)
        areas[i] = _areas

    for row in nb.prange(len(mask)):
        areas_ = calc_area_from_mask_row(mask[row, :])
        for col_v, cnt in areas_.items():
            # print(type(col_v))
            areas[nb.get_thread_id()][col_v] = (
                areas[nb.get_thread_id()].get(col_v, 0) + cnt
            )
    areas_res = {0: 0}
    for d in areas.values():
        for col_v, cnt in d.items():
            areas_res[col_v] = areas_res.get(col_v, 0) + cnt
    return areas_res


@nb.njit
def calc_area_from_mask_row(row: np.ndarray):
    areas = {}
    areas[np.int64(0)] = np.int64(0)

    for col_v in row:
        if col_v == 0:
            continue
        areas[col_v] = areas.get(col_v, 0) + 1
    return areas


@nb.njit
def _collect_remove_ids(enc_a, rows_a, enc_b, rows_b, h):
    remove_dict = nb.typed.Dict.empty(nb.int32, nb.int8)

    for row in range(h):
        sa, la = rows_a[row]
        sb, lb = rows_b[row]

        ia = 0
        ib = 0
        while ia < la and ib < lb:
            a_start, a_end, a_val, _ = enc_a[sa + ia]
            b_start, b_end, b_val, _ = enc_b[sb + ib]

            if a_val == 0:
                ia += 1
                continue
            if b_val == 0:
                ib += 1
                continue

            if a_end < b_start:
                ia += 1
                continue
            if b_end < a_start:
                ib += 1
                continue

            remove_dict[np.int32(a_val)] = 1

            if a_end <= b_end:
                ia += 1
            else:
                ib += 1

    remove_ids_list = nb.typed.List.empty_list(nb.int32)
    for k in remove_dict.keys():
        remove_ids_list.append(k)
    return remove_ids_list


def _merge_masks(enc_a, rows_a, enc_b, rows_b, remove_ids, w, h):
    new_encodings = []
    new_row_indexes = []

    for row in range(h):
        sa, la = rows_a[row]
        sb, lb = rows_b[row]

        row_segs = []

        for i in range(la):
            s, e, v, _ = enc_a[sa + i]
            if v == 0:
                continue
            skip = False
            for rid in remove_ids:
                if v == rid:
                    skip = True
                    break
            if skip:
                continue
            row_segs.append((int(s), int(e), int(v)))

        for i in range(lb):
            s, e, v, _ = enc_b[sb + i]
            if v == 0:
                continue
            row_segs.append((int(s), int(e), int(v)))

        row_segs.sort(key=lambda x: x[0])

        if not row_segs:
            row_segs = [(0, w - 1, 0)]

        start_index = len(new_encodings)
        for seg in row_segs:
            new_encodings.append(seg)
        new_row_indexes.append((start_index, len(row_segs)))

    enc_np = np.array(new_encodings, dtype=np.int32)
    idx_np = np.array(new_row_indexes, dtype=np.int32)
    return enc_np, idx_np


def mask_overlay(
    rcm_a: RangeCompressedMask, rcm_b: RangeCompressedMask
) -> RangeCompressedMask:
    """把 ``rcm_b`` 覆盖到 ``rcm_a`` 上, 直接在区间压缩数据上完成。

    ``rcm_a`` 与 ``rcm_b`` 的尺寸必须一致。若 ``rcm_b`` 与 ``rcm_a`` 的任
    意细胞有交集, ``rcm_a`` 中这些细胞会被完全删除, 最终返回 ``rcm_a``
    剩余细胞与 ``rcm_b`` 的并集。
    """

    if rcm_a.w != rcm_b.w or rcm_a.h != rcm_b.h:
        raise ValueError("rcm_a and rcm_b must have the same shape")

    w, h = rcm_a.w, rcm_a.h

    remove_ids_list = _collect_remove_ids(
        rcm_a.encodings, rcm_a.row_indexes, rcm_b.encodings, rcm_b.row_indexes, h
    )
    remove_ids = np.array(list(remove_ids_list), dtype=np.int32)
    new_enc, new_rows = _merge_masks(
        rcm_a.encodings,
        rcm_a.row_indexes,
        rcm_b.encodings,
        rcm_b.row_indexes,
        remove_ids,
        w,
        h,
    )

    return RangeCompressedMask(
        w=w,
        h=h,
        encodings=new_enc,
        row_indexes=new_rows,
    )


rcm_load = RangeCompressedMask.load
rcm_find_index = RangeCompressedMask.find_index
