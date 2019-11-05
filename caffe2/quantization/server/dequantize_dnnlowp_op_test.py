from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from dnnlowp_test_utils import check_quantized_results_close
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPDequantizeOpTest(hu.HypothesisTestCase):
    @given(size=st.integers(1024, 2048), is_empty=st.booleans(), **hu.gcs_cpu_only)
    def test_dnnlowp_dequantize(self, size, is_empty, gc, dc):
        if is_empty:
            size = 0
        min_ = -10.0
        max_ = 20.0
        X = (np.random.rand(size) * (max_ - min_) + min_).astype(np.float32)

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_type_list = ["Dequantize", "Int8Dequantize"]
        engine = "DNNLOWP"

        outputs.append(Output(X, op_type="", engine=""))

        for op_type in op_type_list:
            net = core.Net("test_net")

            quantize = core.CreateOperator(
                "Quantize", ["X"], ["X_q"], engine=engine, device_option=gc
            )
            net.Proto().op.extend([quantize])

            dequantize = core.CreateOperator(
                op_type, ["X_q"], ["Y"], engine=engine, device_option=gc
            )
            net.Proto().op.extend([dequantize])

            self.ws.create_blob("X").feed(X, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)
