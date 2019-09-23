from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


def mse(x, xh):
    d = (x - xh).reshape(-1)
    return np.sqrt(np.matmul(d, d.transpose())) / len(d)


class FullyConnectedFP16Test(hu.HypothesisTestCase):
    @given(
        input_channels=st.integers(128, 256),
        output_channels=st.integers(128, 256),
        batch_size=st.integers(0, 256),
        **hu.gcs_cpu_only
    )
    def test_fully_connected(self, input_channels, output_channels, batch_size, gc, dc):
        W = np.random.randn(output_channels, input_channels).astype(np.float32)
        X = np.random.randn(batch_size, input_channels).astype(np.float32)
        b = np.random.randn(output_channels).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "engine", "order"])

        order = "NHWC"
        net = core.Net("test_net")
        engine = "FAKE_FP16"

        fc = core.CreateOperator(
            "FC", ["X", "W", "b"], ["Y"], order=order, engine=engine, device_option=gc
        )
        net.Proto().op.extend([fc])

        self.ws.create_blob("X").feed(X, device_option=gc)
        self.ws.create_blob("W").feed(W, device_option=gc)
        self.ws.create_blob("b").feed(b, device_option=gc)
        self.ws.run(net)
        output = Output(Y=self.ws.blobs["Y"].fetch(), engine=engine, order=order)

        # Mimic the quantization in python
        Wh = W.astype(np.float16)
        Xh = X.astype(np.float16)
        bh = b.astype(np.float16)

        bbh = np.outer(np.ones(batch_size, dtype=np.float16), bh)
        assert bbh.dtype == np.float16
        Yrefh = np.matmul(Xh, Wh.transpose()) + bbh
        assert Yrefh.dtype == np.float16

        bb = np.outer(np.ones(batch_size, dtype=np.float32), b)
        Yref = np.matmul(X, W.transpose()) + bb
        assert Yref.dtype == np.float32

        # The error between plain->quantized, and plain->python_quantized
        # should be very close
        mse_c2 = mse(Yref, output.Y)
        mse_py = mse(Yref, Yrefh)
        print(np.abs(mse_c2 - mse_py))
        if batch_size != 0:
            assert np.isclose(mse_c2, mse_py, atol=1e-3), np.abs(mse_c2 - mse_py)
