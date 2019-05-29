from __future__ import absolute_import, division, print_function, unicode_literals

import inspect

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


class TestParallelBatchMatMul(hu.HypothesisTestCase):
    @settings(max_examples=30)
    @given(
        C=st.integers(min_value=0, max_value=3),  # number of batch dims
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        **hu.gcs
    )
    def test_batch_matmul(self, C, M, K, N, trans_a, trans_b, engine, gc, dc):
        dtype = np.float32

        batch_dims = np.random.randint(low=1, high=3, size=C, dtype=np.int64).tolist()
        X = np.random.rand(*(batch_dims + [M, K])).astype(dtype) - 0.5
        if trans_a:
            X = X.swapaxes(-1, -2)
        Y = np.random.rand(*(batch_dims + [K, N])).astype(dtype) - 0.5
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)

        net = core.Net("test_net")
        net.Proto().type = (
            "async_scheduling" if engine == "INTRA_OP_PARALLEL" else "parallel"
        )
        net.Proto().num_workers = 7
        net.BatchMatMul(
            ["X", "Y"], "out", trans_a=trans_a, trans_b=trans_b, engine=engine
        )

        ref_net = core.Net("ref_test_net")
        ref_net.BatchMatMul(["X", "Y"], "out_ref", trans_a=trans_a, trans_b=trans_b)

        workspace.RunNetOnce(net)
        workspace.RunNetOnce(ref_net)

        output = workspace.FetchBlob("out")
        ref_output = workspace.FetchBlob("out_ref")
        np.testing.assert_allclose(
            output,
            ref_output,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Out is not matching the reference",
        )

    def _test_batch_matmul_with_broadcast_common(
        self, X, Y, engine, dtype, gc, dc, trans_a=None, trans_b=None
    ):
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)

        net = core.Net("test_net")
        net.Proto().type = (
            "async_scheduling" if engine == "INTRA_OP_PARALLEL" else "parallel"
        )
        net.Proto().num_workers = 7

        ref_net = core.Net("ref_test_net")

        if trans_a is not None and trans_b is not None:
            net.BatchMatMul(
                ["X", "Y"],
                "out",
                trans_a=trans_a,
                trans_b=trans_b,
                broadcast=1,
                engine=engine,
            )
            ref_net.BatchMatMul(
                ["X", "Y"], "out_ref", trans_a=trans_a, trans_b=trans_b, broadcast=1
            )
        else:
            net.BatchMatMul(["X", "Y"], "out", broadcast=1, engine=engine)
            ref_net.BatchMatMul(["X", "Y"], "out_ref", broadcast=1)

        workspace.RunNetOnce(net)
        workspace.RunNetOnce(ref_net)

        output = workspace.FetchBlob("out")
        ref_output = workspace.FetchBlob("out_ref")
        np.testing.assert_allclose(
            output,
            ref_output,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Out is not matching the reference",
        )

    @given(
        C_1=st.integers(min_value=0, max_value=3),  # number of batch dims
        C_2=st.integers(min_value=0, max_value=3),
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        **hu.gcs
    )
    def test_numpy_batch_matmul(
        self, C_1, C_2, M, K, N, trans_a, trans_b, engine, gc, dc
    ):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        batch_dims = np.random.randint(
            low=0, high=3, size=max(C_1, C_2), dtype=np.int64
        ).tolist()
        lbd = len(batch_dims)
        X = np.random.rand(*(batch_dims[lbd - C_1 :] + [M, K])).astype(dtype) - 0.5
        if trans_a:
            X = X.swapaxes(-1, -2)
        Y = np.random.rand(*(batch_dims[lbd - C_2 :] + [K, N])).astype(dtype) - 0.5
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        self._test_batch_matmul_with_broadcast_common(
            X, Y, engine, dtype, gc, dc, trans_a, trans_b
        )

    @settings(max_examples=30)
    @given(
        K=st.integers(min_value=1, max_value=10),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        **hu.gcs
    )
    def test_numpy_batch_matmul_1d(self, K, engine, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(K).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(K).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, engine, dtype, gc, dc)

    @settings(max_examples=30)
    @given(
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        **hu.gcs
    )
    def test_numpy_batch_matmul_1d_2d(self, K, N, engine, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(K).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(*[K, N]).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, engine, dtype, gc, dc)

    @settings(max_examples=30)
    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        **hu.gcs
    )
    def test_numpy_batch_matmul_2d_1d(self, M, K, engine, gc, dc):
        np.set_printoptions(threshold=np.nan)
        dtype = np.float32
        X = np.random.rand(*[M, K]).astype(dtype) - 0.5
        # TODO: test trans_a and trans_b
        Y = np.random.rand(K).astype(dtype) - 0.5

        self._test_batch_matmul_with_broadcast_common(X, Y, engine, dtype, gc, dc)


if __name__ == "__main__":
    import unittest

    unittest.main()
