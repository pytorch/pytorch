from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hypothesis import given
import hypothesis.strategies as st
import unittest

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestTile(hu.HypothesisTestCase):
    @given(M=st.integers(min_value=1, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           N=st.integers(min_value=1, max_value=10),
           tiles=st.integers(min_value=1, max_value=3),
           axis=st.integers(min_value=0, max_value=2),
           **hu.gcs)
    def test_tile(self, M, K, N, tiles, axis, gc, dc):
        X = np.random.rand(M, K, N).astype(np.float32)

        op = core.CreateOperator(
            'Tile', ['X'], 'out',
            tiles=tiles,
            axis=axis,
        )

        def tile_ref(X, tiles, axis):
            dims = np.asarray([1, 1, 1], dtype=np.int)
            dims[axis] = tiles
            tiled_data = np.tile(X, dims)
            return (tiled_data,)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, tiles, axis],
                                   tile_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(M=st.integers(min_value=1, max_value=200),
           N=st.integers(min_value=1, max_value=200),
           tiles=st.integers(min_value=50, max_value=100),
           **hu.gcs)
    def test_tile_grad(self, M, N, tiles, gc, dc):
        X = np.random.rand(M, N).astype(np.float32)
        axis = 1

        op = core.CreateOperator(
            'Tile', ['X'], 'out',
            tiles=tiles,
            axis=axis,
        )

        def tile_ref(X, tiles, axis):
            dims = np.asarray([1, 1], dtype=np.int)
            dims[axis] = tiles
            tiled_data = np.tile(X, dims)
            return (tiled_data,)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, tiles, axis],
                                   tile_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])

        # Gradient check wrt X
        grad_op = core.CreateOperator(
            'TileGradient', ['dOut'], 'dX',
            tiles=tiles,
            axis=axis,
        )
        dX = np.random.rand(M, N * tiles).astype(np.float32)
        self.assertDeviceChecks(dc, grad_op, [dX], [0])

    @given(M=st.integers(min_value=1, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           N=st.integers(min_value=1, max_value=10),
           tiles=st.integers(min_value=1, max_value=3),
           axis=st.integers(min_value=0, max_value=2),
           **hu.gcs)
    def test_tilewinput(self, M, K, N, tiles, axis, gc, dc):
        X = np.random.rand(M, K, N).astype(np.float32)

        tiles_arg = np.array([tiles], dtype=np.int32)
        axis_arg = np.array([axis], dtype=np.int32)

        op = core.CreateOperator(
            'Tile', ['X', 'tiles', 'axis'], 'out',
        )

        def tile_ref(X, tiles, axis):
            dims = np.asarray([1, 1, 1], dtype=np.int)
            dims[axis] = tiles
            tiled_data = np.tile(X, dims)
            return (tiled_data,)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, tiles_arg, axis_arg],
                                   tile_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, tiles_arg, axis_arg], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, tiles_arg, axis_arg], 0, [0])


if __name__ == "__main__":
    unittest.main()
