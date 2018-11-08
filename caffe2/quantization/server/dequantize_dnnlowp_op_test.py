from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep
from hypothesis import given
import hypothesis.strategies as st
import collections
from dnnlowp_test_utils import check_quantized_results_close

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPDequantizeOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048),
           **hu.gcs_cpu_only)
    def test_dnnlowp_dequantize(self, size, gc, dc):
        min_ = -10.
        max_ = 20.
        X = (np.random.rand(size) * (max_ - min_) + min_).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_type_list = ["Dequantize", "Int8Dequantize"]
        engine = "DNNLOWP"

        outputs.append(Output(X, op_type="", engine=""))

        for op_type in op_type_list:
            net = core.Net("test_net")

            quantize = core.CreateOperator(
                "Quantize",
                ["X"],
                ["X_q"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([quantize])

            dequantize = core.CreateOperator(
                op_type,
                ["X_q"],
                ["Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            outputs.append(Output(
                Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine))

        check_quantized_results_close(outputs)
