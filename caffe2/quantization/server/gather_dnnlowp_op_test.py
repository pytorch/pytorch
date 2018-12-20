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


class DNNLowPGatherOpTest(hu.HypothesisTestCase):
    @given(
        dim1=st.integers(256, 512),
        dim2=st.integers(32, 256),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_gather(self, dim1, dim2, in_quantized, out_quantized, gc, dc):
        # FIXME : DNNLOWP Gather doesn't support quantized input and
        # dequantized output
        if in_quantized:
            out_quantized = True

        data = (np.random.rand(dim1) * 2 - 1).astype(np.float32)
        index = np.floor(np.random.rand(dim2) * dim1).astype(np.int32)

        Output = collections.namedtuple("Output", ["out", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("Gather", ""),
            ("Gather", "DNNLOWP"),
            ("Int8Gather", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

            if do_quantize:
                quantize_data = core.CreateOperator(
                    "Quantize", ["data"], ["data_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize_data])

            gather = core.CreateOperator(
                op_type,
                ["data_q" if do_quantize else "data", "index"],
                ["out_q" if do_dequantize else "out"],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([gather])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", ["out_q"], ["out"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("data").feed(data, device_option=gc)
            self.ws.create_blob("index").feed(index, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(out=self.ws.blobs["out"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs, ref=data)
