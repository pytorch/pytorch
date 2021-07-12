

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPResizeNearestOpTest(hu.HypothesisTestCase):
    @given(
        N=st.integers(0, 3),
        H=st.integers(10, 300),
        W=st.integers(10, 300),
        C=st.integers(1, 32),
        scale_w=st.floats(0.25, 4.0) | st.just(2.0),
        scale_h=st.floats(0.25, 4.0) | st.just(2.0),
        **hu.gcs_cpu_only
    )
    @settings(deadline=None, max_examples=50)
    def test_resize_nearest(self, N, H, W, C, scale_w, scale_h, gc, dc):
        X = np.round(np.random.rand(N, H, W, C) * 255).astype(np.float32)

        quantize = core.CreateOperator("Quantize", ["X"], ["X_q"], engine="DNNLOWP")
        resize_nearest = core.CreateOperator(
            "Int8ResizeNearest",
            ["X_q"],
            ["Y_q"],
            width_scale=scale_w,
            height_scale=scale_h,
            engine="DNNLOWP",
        )

        net = core.Net("test_net")
        net.Proto().op.extend([quantize, resize_nearest])

        workspace.FeedBlob("X", X)
        workspace.RunNetOnce(net)
        X_q = workspace.FetchInt8Blob("X_q").data
        Y_q = workspace.FetchInt8Blob("Y_q").data

        def resize_nearest_ref(X):
            outH = np.int32(H * scale_h)
            outW = np.int32(W * scale_w)
            outH_idxs, outW_idxs = np.meshgrid(
                np.arange(outH), np.arange(outW), indexing="ij"
            )
            inH_idxs = np.minimum(outH_idxs / scale_h, H - 1).astype(np.int32)
            inW_idxs = np.minimum(outW_idxs / scale_w, W - 1).astype(np.int32)
            Y = X[:, inH_idxs, inW_idxs, :]
            return Y

        Y_q_ref = resize_nearest_ref(X_q)
        np.testing.assert_allclose(Y_q, Y_q_ref)
