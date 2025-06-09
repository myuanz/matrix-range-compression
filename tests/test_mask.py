import random
import os

import cv2
import numpy as np

from range_compression import (
    RangeCompressedMask,
    mask_encode,
    calc_area_from_encodings,
    mask_overlay,
)
from range_compression.range_compression import calc_area_from_mask

RND_IMG_W = 4096 + 1
RND_IMG_H = 4096 - 1

def get_rand_image():
    width, height = RND_IMG_W, RND_IMG_H
    image = np.zeros((height, width), dtype=np.uint8)

    num_polygons = 50

    for _ in range(num_polygons):
        num_edges = random.randint(3, 8)
        color = random.randint(0, 255)
        vertices = []

        for _ in range(num_edges):
            x = random.randint(0, width)
            y = random.randint(0, height)
            vertices.append((x, y))

        pts = np.array(vertices, np.int32)

        cv2.fillPoly(image, [pts], (color, ))
    return image

def _test_encode(image):
    height, width = image.shape

    rcm = mask_encode(image)
    assert isinstance(rcm, RangeCompressedMask)
    assert rcm.w == width and rcm.h == height
    assert len(rcm.row_indexes) == height
    return rcm

def get_randxy(width, height):
    random_n = 10000
    randx, randy = np.random.randint(0, width, random_n), np.random.randint(0, height, random_n)
    return randx, randy

def _test_find_mask(rcm, image, randx, randy):
    res = rcm.find_index(randx, randy)
    actual_res = image[randy, randx]
    assert np.array_equal(res, actual_res)
    return actual_res


def test_mask(benchmark):
    image = get_rand_image()
    height, width = image.shape

    rcm = _test_encode(image)
    randx, randy = get_randxy(width, height)
    actual_res = _test_find_mask(rcm, image, randx, randy)

    # test invalid rxry
    for rx in (-2**32, -2**31, -RND_IMG_W-1, -RND_IMG_W, -1, RND_IMG_W, RND_IMG_W+1, RND_IMG_W*1000, 2**31, 2**32):
        for ry in (-2**32, -2**31, -RND_IMG_H-1, -RND_IMG_H, -1, RND_IMG_H, RND_IMG_H+1, RND_IMG_H*1000, 2**31, 2**32):
            print(rx, ry)
            assert rcm.find_index(np.array([rx]), np.array([ry])) == 0

    benchmark.pedantic(_test_find_mask, args=(rcm, image, randx, randy), iterations=10, rounds=50)
    
    rcm.save('tests/output')
    rcm2 = RangeCompressedMask.load('tests/output')
    assert np.array_equal(rcm.encodings[:, :3], rcm2.encodings)
    
    actual_res2 = rcm2.find_index(randx, randy)
    assert np.array_equal(actual_res, actual_res2)

    os.remove('tests/output/encodings.parquet')
    os.remove('tests/output/row_indexes.parquet')
    os.remove('tests/output/meta.json')
    os.rmdir('tests/output')

def test_area():
    image = get_rand_image()
    rcm = _test_encode(image)
    res1 = calc_area_from_encodings(rcm.encodings, rcm.row_indexes)
    res2 = calc_area_from_mask(image)
    res3 = rcm.calc_area()
    del res1[0]; del res2[0]; del res3[0]
    print(res1)
    assert res1 == res2 == res3
    mask = rcm.to_mask()
    assert np.all(image == mask)


def test_mask_overlay():
    img_a = np.zeros((10, 10), dtype=np.int32)
    img_b = np.zeros((10, 10), dtype=np.int32)

    img_a[1:4, 1:4] = 1
    img_a[6:9, 6:9] = 2

    img_b[2:5, 2:5] = 5

    expected = img_a.copy()
    expected[expected == 1] = 0
    expected[img_b != 0] = img_b[img_b != 0]

    rcm_a = mask_encode(img_a)
    rcm_b = mask_encode(img_b)
    rcm_res = mask_overlay(rcm_a, rcm_b)

    mask_res = rcm_res.to_mask()

    assert np.all(mask_res == expected)

