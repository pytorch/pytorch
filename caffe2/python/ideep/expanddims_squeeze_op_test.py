




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ExpandDimsSqueezeTest(hu.HypothesisTestCase):
    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs
        )
    def test_squeeze(self, squeeze_dims, inplace, gc, dc):
        shape = [
            1 if dim in squeeze_dims else np.random.randint(1, 5)
            for dim in range(4)
        ]
        X = np.random.rand(*shape).astype(np.float32)
        op = core.CreateOperator(
            "Squeeze", "X", "X" if inplace else "Y", dims=squeeze_dims
        )
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs_cpu_ideep
        )
    def test_squeeze_fallback(self, squeeze_dims, inplace, gc, dc):
        shape = [
            1 if dim in squeeze_dims else np.random.randint(1, 5)
            for dim in range(4)
        ]
        X = np.random.rand(*shape).astype(np.float32)
        op0 = core.CreateOperator(
            "Squeeze",
            "X0",
            "X0" if inplace else "Y0",
            dims=squeeze_dims,
            device_option=dc[0]
        )
        workspace.FeedBlob('X0', X, dc[0])
        workspace.RunOperatorOnce(op0)
        Y0 = workspace.FetchBlob("X0" if inplace else "Y0")

        op1 = core.CreateOperator(
            "Squeeze",
            "X1",
            "X1" if inplace else "Y1",
            dims=squeeze_dims,
            device_option=dc[1]
        )
        workspace.FeedBlob('X1', X, dc[0])
        workspace.RunOperatorOnce(op1)
        Y1 = workspace.FetchBlob("X1" if inplace else "Y1")

        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)


    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs
        )
    def test_expand_dims(self, squeeze_dims, inplace, gc, dc):
        oshape = [
            1 if dim in squeeze_dims else np.random.randint(2, 5)
            for dim in range(4)
        ]
        nshape = [s for s in oshape if s!=1]
        expand_dims = [i for i in range(len(oshape)) if oshape[i]==1]

        X = np.random.rand(*nshape).astype(np.float32)
        op = core.CreateOperator(
            "ExpandDims", "X", "X" if inplace else "Y", dims=expand_dims
        )
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs_cpu_ideep
        )
    def test_expand_dims_fallback(self, squeeze_dims, inplace, gc, dc):
        oshape = [
            1 if dim in squeeze_dims else np.random.randint(2, 5)
            for dim in range(4)
        ]
        nshape = [s for s in oshape if s!=1]
        expand_dims = [i for i in range(len(oshape)) if oshape[i]==1]

        X = np.random.rand(*nshape).astype(np.float32)
        op0 = core.CreateOperator(
            "ExpandDims",
            "X0",
            "X0" if inplace else "Y0",
            dims=expand_dims,
            device_option=dc[0]
        )
        workspace.FeedBlob('X0', X, dc[0])
        workspace.RunOperatorOnce(op0)
        Y0 = workspace.FetchBlob("X0" if inplace else "Y0")

        op1 = core.CreateOperator(
            "ExpandDims",
            "X1",
            "X1" if inplace else "Y1",
            dims=expand_dims,
            device_option=dc[1]
        )
        workspace.FeedBlob('X1', X, dc[0])
        workspace.RunOperatorOnce(op1)
        Y1 = workspace.FetchBlob("X1" if inplace else "Y1")

        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
