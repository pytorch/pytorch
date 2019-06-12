from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep
from hypothesis import given
import hypothesis.strategies as st

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPQuantizeOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048),
           **hu.gcs_cpu_only)
    def test_dnnlowp_quantize(self, size, gc, dc):
        min_ = -10.
        max_ = 20.
        X = (np.random.rand(size) * (max_ - min_) + min_).astype(np.float32)

        op_type_list = ["Quantize", "Int8Quantize"]
        engine = "DNNLOWP"

        for op_type in op_type_list:
            net = core.Net("test_net")

            quantize = core.CreateOperator(
                op_type,
                ["X"],
                ["X_q"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([quantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            X_q = self.ws.blobs["X_q"].fetch()[0]

            # Dequantize results and measure quantization error against inputs
            X_scale = (max(np.max(X), 0) - min(np.min(X), 0)) / 255
            X_zero = np.round(-np.min(X) / X_scale)
            X_dq = X_scale * (X_q - X_zero)

            # should be divided by 2 in an exact math, but divide by 1.9 here
            # considering finite precision in floating-point numbers
            atol = X_scale / 1.9
            np.testing.assert_allclose(X_dq, X, atol=atol, rtol=0)
