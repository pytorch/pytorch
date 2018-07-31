from __future__ import print_function, division

import itertools
import math
import random
import unittest

import torch
from common import run_tests, TestCase, TEST_NUMPY, TEST_SCIPY

if TEST_SCIPY:
    import scipy.ndimage
    # print("scipy version: ", scipy.version.version)

if TEST_NUMPY:
    import numpy as np


import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# from .affine import affine_grid_generator


# flake8: noqa: E241

if torch.cuda.is_available():
    def device_():
        return ['cpu', 'cuda']
else:
    def device_():
        return ['cpu']


def angle_rad_():
    return [r * math.pi * 2 for r in [0.0, 0.5, 0.25, 0.125, random.random()]]


def axis_vector_():
    t = (random.random(), random.random(), random.random())
    l = sum(x ** 2 for x in t) ** 0.5

    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(x/l for x in t)]


def input_size2d_():
    return [[1, 1, 3, 5], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 3, 4]]


def output_size2d_():
    return [[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 4, 3], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 6, 6]]


def output_size2dsq_():
    return [[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 5], [1, 1, 6, 6]]


def input_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 2, 3, 4], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 3, 4, 5]]


def input_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 6, 6, 6]]


def output_size3dsq_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def output_size3d_():
    return [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5], [1, 1, 4, 3, 2], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]]


def _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad):
    log.debug(["_buildEquivalentTransforms2d", device, input_size, output_size, angle_rad * 180 / math.pi])
    input_center = [(x - 1) / 2 for x in input_size]
    output_center = [(x - 1) / 2 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    intrans_ary = np.array([
        [1, 0, input_center[2]],
        [0, 1, input_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0],
        [0, input_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    rotation_ary = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1 / output_center[2], 0, 0],
        [0, 1 / output_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, -output_center[2]],
        [0, 1, -output_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    
    transform_ary = np.dot(np.dot(np.dot(np.dot(intrans_ary, inscale_ary), rotation_ary.T), outscale_ary), outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, rotation_ary.T), outscale_ary), outtrans_ary)

    # transform_ary = intrans_ary @ inscale_ary @ rotation_ary.T @ outscale_ary @ outtrans_ary
    # grid_ary = reorder_ary @ rotation_ary.T @ outscale_ary @ outtrans_ary
    transform_tensor = torch.from_numpy((rotation_ary)).to(device, torch.float32)

    transform_tensor = transform_tensor[:2].unsqueeze(0)

    log.debug(['transform_tensor', transform_tensor.size(), transform_tensor.dtype, transform_tensor.device])
    log.debug([transform_tensor])
    log.debug(['outtrans_ary', outtrans_ary.shape, outtrans_ary.dtype])
    log.debug([outtrans_ary.round(3)])
    log.debug(['outscale_ary', outscale_ary.shape, outscale_ary.dtype])
    log.debug([outscale_ary.round(3)])
    log.debug(['rotation_ary', rotation_ary.shape, rotation_ary.dtype])
    log.debug([rotation_ary.round(3)])
    log.debug(['inscale_ary', inscale_ary.shape, inscale_ary.dtype])
    log.debug([inscale_ary.round(3)])
    log.debug(['intrans_ary', intrans_ary.shape, intrans_ary.dtype])
    log.debug([intrans_ary.round(3)])
    log.debug(['transform_ary', transform_ary.shape, transform_ary.dtype])
    log.debug([transform_ary.round(3)])
    log.debug(['grid_ary', grid_ary.shape, grid_ary.dtype])
    log.debug([grid_ary.round(3)])

    # def prtf(pt):
    #     log.debug([pt, 'transformed', (transform_ary @ (pt + [1]))[:2].round(3)])
    #
    # prtf([0, 0])
    # prtf([1, 0])
    # prtf([2, 0])
    #
    # log.debug([''])
    #
    # prtf([0, 0])
    # prtf([0, 1])
    # prtf([0, 2])
    # prtf(output_center[2:])

    return transform_tensor, transform_ary, grid_ary


def _buildEquivalentTransforms3d(device, input_size, output_size, angle_rad, axis_vector):
    log.debug(["_buildEquivalentTransforms2d", device, input_size, output_size, angle_rad * 180 / math.pi, axis_vector])
    input_center = [(x - 1) / 2 for x in input_size]
    output_center = [(x - 1) / 2 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    c1 = 1 - c

    intrans_ary = np.array([
        [1, 0, 0, input_center[2]],
        [0, 1, 0, input_center[3]],
        [0, 0, 1, input_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0, 0],
        [0, input_center[3], 0, 0],
        [0, 0, input_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    l, m, n = axis_vector
    scipyRotation_ary = np.array([
        [l * l * c1 + c,     m * l * c1 - n * s, n * l * c1 + m * s, 0],
        [l * m * c1 + n * s, m * m * c1 + c,     n * m * c1 - l * s, 0],
        [l * n * c1 - m * s, m * n * c1 + l * s, n * n * c1 + c,     0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    z, y, x = axis_vector
    torchRotation_ary = np.array([
        [x * x * c1 + c,     y * x * c1 - z * s, z * x * c1 + y * s, 0],
        [x * y * c1 + z * s, y * y * c1 + c,     z * y * c1 - x * s, 0],
        [x * z * c1 - y * s, y * z * c1 + x * s, z * z * c1 + c,     0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1 / output_center[2], 0, 0, 0],
        [0, 1 / output_center[3], 0, 0],
        [0, 0, 1 / output_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, 0, -output_center[2]],
        [0, 1, 0, -output_center[3]],
        [0, 0, 1, -output_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    transform_ary = np.dot(np.dot(np.dot(np.dot(intrans_ary, inscale_ary), np.linalg.inv(scipyRotation_ary)), outscale_ary), outtrans_ary)
    grid_ary = np.dot(np.dot(np.dot(reorder_ary, np.linalg.inv(scipyRotation_ary)), outscale_ary), outtrans_ary)

    # transform_ary = intrans_ary @ inscale_ary @ np.linalg.inv(scipyRotation_ary) @ outscale_ary @ outtrans_ary
    # grid_ary = reorder_ary @ np.linalg.inv(scipyRotation_ary) @ outscale_ary @ outtrans_ary
    transform_tensor = torch.from_numpy((torchRotation_ary)).to(device, torch.float32)
    transform_tensor = transform_tensor[:3].unsqueeze(0)

    log.debug(['transform_tensor', transform_tensor.size(), transform_tensor.dtype, transform_tensor.device])
    log.debug([transform_tensor])
    log.debug(['outtrans_ary', outtrans_ary.shape, outtrans_ary.dtype])
    log.debug([outtrans_ary.round(3)])
    log.debug(['outscale_ary', outscale_ary.shape, outscale_ary.dtype])
    log.debug([outscale_ary.round(3)])
    log.debug(['rotation_ary', scipyRotation_ary.shape, scipyRotation_ary.dtype, axis_vector, angle_rad])
    log.debug([scipyRotation_ary.round(3)])
    log.debug(['inscale_ary', inscale_ary.shape, inscale_ary.dtype])
    log.debug([inscale_ary.round(3)])
    log.debug(['intrans_ary', intrans_ary.shape, intrans_ary.dtype])
    log.debug([intrans_ary.round(3)])
    log.debug(['transform_ary', transform_ary.shape, transform_ary.dtype])
    log.debug([transform_ary.round(3)])
    log.debug(['grid_ary', grid_ary.shape, grid_ary.dtype])
    log.debug([grid_ary.round(3)])

    # def prtf(pt):
    #     log.debug([pt, 'transformed', (transform_ary @ (pt + [1]))[:3].round(3)])
    #
    # prtf([0, 0, 0])
    # prtf([1, 0, 0])
    # prtf([2, 0, 0])
    #
    # log.debug([''])
    #
    # prtf([0, 0, 0])
    # prtf([0, 1, 0])
    # prtf([0, 2, 0])
    #
    # log.debug([''])
    #
    # prtf([0, 0, 0])
    # prtf([0, 0, 1])
    # prtf([0, 0, 2])
    #
    # prtf(output_center[2:])

    return transform_tensor, transform_ary, grid_ary


class TestAffine(TestCase):

    @unittest.skipIf(not (TEST_SCIPY and TEST_NUMPY), "Scipy and/or numpy not found")
    def test_affine_2d_rotate0(self):
        for device in device_():
            try:
                input_size = [1, 1, 3, 3]
                input_ary = np.array(np.random.random(input_size), dtype=np.float32)
                output_size = [1, 1, 5, 5]
                angle_rad = 0.

                transform_tensor, transform_ary, offset = \
                    _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

                # reference
                # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
                scipy_ary = scipy.ndimage.affine_transform(
                    input_ary[0, 0],
                    transform_ary,
                    offset=offset,
                    output_shape=output_size[2:],
                    # output=None,
                    order=1,
                    mode='nearest',
                    # cval=0.0,
                    prefilter=False)

                log.debug(['input_ary', input_ary.shape, input_ary.dtype])
                log.debug([input_ary])
                log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
                log.debug([scipy_ary])

                affine_tensor = torch.nn.functional.affine_grid(
                    transform_tensor,
                    torch.Size(output_size)
                )

                log.debug(['affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device])
                log.debug([affine_tensor])

                gridsample_ary = torch.nn.functional.grid_sample(
                    torch.tensor(input_ary, device=device).to(device),
                    affine_tensor,
                    padding_mode='border'
                ).to('cpu').numpy()

                log.debug(['input_ary', input_ary.shape, input_ary.dtype])
                log.debug([input_ary])
                log.debug(['gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype])
                log.debug([gridsample_ary])
                log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
                log.debug([scipy_ary])

                assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
                assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
                # assert False
            except Exception as e:
                log.error([device])
                raise

    @unittest.skipIf(not (TEST_SCIPY and TEST_NUMPY), "Scipy and/or numpy not found")
    def test_affine_2d_rotate90(self):
        for device, input_size2dsq, output_size2dsq in \
                itertools.product(device_(), input_size2dsq_(), output_size2dsq_()):
            try:
                input_size = input_size2dsq
                input_ary = np.array(np.random.random(input_size), dtype=np.float32)
                output_size = output_size2dsq
                angle_rad = 0.25 * math.pi * 2

                transform_tensor, transform_ary, offset = \
                    _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

                # reference
                # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
                scipy_ary = scipy.ndimage.affine_transform(
                    input_ary[0, 0],
                    transform_ary,
                    offset=offset,
                    output_shape=output_size[2:],
                    # output=None,
                    order=1,
                    mode='nearest',
                    # cval=0.0,
                    prefilter=True)

                log.debug(['input_ary', input_ary.shape, input_ary.dtype, input_ary.mean()])
                log.debug([input_ary])
                log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype, scipy_ary.mean()])
                log.debug([scipy_ary])

                if input_size2dsq == output_size2dsq:
                    assert np.abs(scipy_ary.mean() - input_ary.mean()) < 1e-6
                assert np.abs(scipy_ary[0, 0] - input_ary[0, 0, 0, -1]).max() < 1e-6
                assert np.abs(scipy_ary[0, -1] - input_ary[0, 0, -1, -1]).max() < 1e-6
                assert np.abs(scipy_ary[-1, -1] - input_ary[0, 0, -1, 0]).max() < 1e-6
                assert np.abs(scipy_ary[-1, 0] - input_ary[0, 0, 0, 0]).max() < 1e-6

                affine_tensor = torch.nn.functional.affine_grid(
                    transform_tensor,
                    torch.Size(output_size)
                )

                log.debug(['affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device])
                log.debug([affine_tensor])

                gridsample_ary = torch.nn.functional.grid_sample(
                    torch.tensor(input_ary, device=device).to(device),
                    affine_tensor,
                    padding_mode='border'
                ).to('cpu').numpy()

                log.debug(['input_ary', input_ary.shape, input_ary.dtype])
                log.debug([input_ary])
                log.debug(['gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype])
                log.debug([gridsample_ary])
                log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
                log.debug([scipy_ary])

                assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
                assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
                # assert False
            except Exception as e:
                log.error([device, input_size2dsq, output_size2dsq])
                raise

    @unittest.skipIf(not (TEST_SCIPY and TEST_NUMPY), "Scipy and/or numpy not found")
    def test_affine_2d_rotate45(self):
        for device in device_():
            input_size = [1, 1, 3, 3]
            input_ary = np.array(np.zeros(input_size), dtype=np.float32)
            input_ary[0, 0, 0, :] = 0.5
            input_ary[0, 0, 2, 2] = 1.0
            output_size = [1, 1, 3, 3]
            angle_rad = 0.125 * math.pi * 2

            transform_tensor, transform_ary, offset = \
                _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

            # reference
            # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                offset=offset,
                output_shape=output_size[2:],
                # output=None,
                order=1,
                mode='nearest',
                # cval=0.0,
                prefilter=False)

            log.debug(['input_ary', input_ary.shape, input_ary.dtype])
            log.debug([input_ary])
            log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
            log.debug([scipy_ary])

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size)
            )

            log.debug(['affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device])
            log.debug([affine_tensor])

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border'
            ).to('cpu').numpy()

            log.debug(['input_ary', input_ary.shape, input_ary.dtype])
            log.debug([input_ary])
            log.debug(['gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype])
            log.debug([gridsample_ary])
            log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
            log.debug([scipy_ary])

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
            # assert False

    @unittest.skipIf(not (TEST_SCIPY and TEST_NUMPY), "Scipy and/or numpy not found")
    def test_affine_2d_rotateRandom(self):
        for device, angle_rad, input_size2d, output_size2d in \
                itertools.product(device_(), angle_rad_(), input_size2d_(), output_size2d_()):

            input_size = input_size2d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
            output_size = output_size2d

            input_ary[0, 0, 0, 0] = 2
            input_ary[0, 0, 0, -1] = 4
            input_ary[0, 0, -1, 0] = 6
            input_ary[0, 0, -1, -1] = 8

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

            # reference
            # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                # offset=offset,
                output_shape=output_size[2:],
                # output=None,
                order=1,
                mode='nearest',
                # cval=0.0,
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size)
            )

            log.debug(['affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device])
            log.debug([affine_tensor])

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])
                    log.debug([r, c, 'affine:', affine_tensor[0, r, c], 'grid:', grid_out[:2]])

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border'
            ).to('cpu').numpy()

            log.debug(['input_ary', input_ary.shape, input_ary.dtype])
            log.debug([input_ary.round(3)])
            log.debug(['gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype])
            log.debug([gridsample_ary.round(3)])
            log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
            log.debug([scipy_ary.round(3)])

            for r in range(affine_tensor.size(1)):
                for c in range(affine_tensor.size(2)):
                    grid_out = np.dot(grid_ary, [r, c, 1])

                    try:
                        assert np.allclose(affine_tensor[0, r, c], grid_out[:2], atol=1e-5)
                    except Exception:
                        log.debug([r, c, 'affine:', affine_tensor[0, r, c], 'grid:', grid_out[:2]])
                        raise

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5
            # assert False

    @unittest.skipIf(not (TEST_SCIPY and TEST_NUMPY), "Scipy and/or numpy not found")
    def test_affine_3d_rotateRandom(self):
        for device, angle_rad, axis_vector, input_size3d, output_size3d in \
                itertools.product(device_(), angle_rad_(), axis_vector_(), input_size3d_(), output_size3d_()):
            input_size = input_size3d
            input_ary = np.array(np.random.random(input_size), dtype=np.float32)
            output_size = output_size3d

            input_ary[0, 0,  0,  0,  0] = 2
            input_ary[0, 0,  0,  0, -1] = 3
            input_ary[0, 0,  0, -1,  0] = 4
            input_ary[0, 0,  0, -1, -1] = 5
            input_ary[0, 0, -1,  0,  0] = 6
            input_ary[0, 0, -1,  0, -1] = 7
            input_ary[0, 0, -1, -1,  0] = 8
            input_ary[0, 0, -1, -1, -1] = 9

            transform_tensor, transform_ary, grid_ary = \
                _buildEquivalentTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

            # reference
            # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
            scipy_ary = scipy.ndimage.affine_transform(
                input_ary[0, 0],
                transform_ary,
                # offset=offset,
                output_shape=output_size[2:],
                # output=None,
                order=1,
                mode='nearest',
                # cval=0.0,
                prefilter=False)

            affine_tensor = torch.nn.functional.affine_grid(
                transform_tensor,
                torch.Size(output_size)
            )

            log.debug(['affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device])
            log.debug([affine_tensor])

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        log.debug([i, r, c, 'affine:', affine_tensor[0, i, r, c], 'grid:', grid_out[:3].round(3)])

            log.debug(['input_ary', input_ary.shape, input_ary.dtype])
            log.debug([input_ary.round(3)])

            gridsample_ary = torch.nn.functional.grid_sample(
                torch.tensor(input_ary, device=device).to(device),
                affine_tensor,
                padding_mode='border'
            ).to('cpu').numpy()

            log.debug(['gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype])
            log.debug([gridsample_ary.round(3)])
            log.debug(['scipy_ary', scipy_ary.shape, scipy_ary.dtype])
            log.debug([scipy_ary.round(3)])

            for i in range(affine_tensor.size(1)):
                for r in range(affine_tensor.size(2)):
                    for c in range(affine_tensor.size(3)):
                        grid_out = np.dot(grid_ary, [i, r, c, 1])
                        try:
                            assert np.allclose(affine_tensor[0, i, r, c], grid_out[:3], atol=1e-5)
                        except Exception:
                            log.debug([i, r, c, 'affine:', affine_tensor[0, i, r, c], 'grid:', grid_out[:3].round(3)])
                            raise

            assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5
            # assert False

if __name__ == '__main__':
    run_tests()
