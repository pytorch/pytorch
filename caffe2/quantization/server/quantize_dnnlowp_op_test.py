from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPQuantizeOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048), is_empty=st.booleans(), **hu.gcs_cpu_only)
    def test_dnnlowp_quantize(self, size, is_empty, gc, dc):
        if is_empty:
            size = 0
        min_ = -10.0
        max_ = 20.0
        X = (np.random.rand(size) * (max_ - min_) + min_).astype(np.float32)

        op_type_list = ["Quantize", "Int8Quantize"]
        engine = "DNNLOWP"

        for op_type in op_type_list:
            net = core.Net("test_net")

            quantize = core.CreateOperator(
                op_type, ["X"], ["X_q"], engine=engine, device_option=gc
            )
            net.Proto().op.extend([quantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            X_q = self.ws.blobs["X_q"].fetch()[0]

            # Dequantize results and measure quantization error against inputs
            X_min = 0 if X.size == 0 else X.min()
            X_max = 1 if X.size == 0 else X.max()
            X_scale = (max(X_max, 0) - min(X_min, 0)) / 255
            X_zero = np.round(-X_min / X_scale)
            X_dq = X_scale * (X_q - X_zero)

            # should be divided by 2 in an exact math, but divide by 1.9 here
            # considering finite precision in floating-point numbers
            atol = X_scale / 1.9
            np.testing.assert_allclose(X_dq, X, atol=atol, rtol=0)
