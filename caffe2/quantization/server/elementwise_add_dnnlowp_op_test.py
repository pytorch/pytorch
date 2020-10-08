

import collections

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from caffe2.quantization.server.dnnlowp_test_utils import check_quantized_results_close
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPAddOpTest(hu.HypothesisTestCase):
    @given(
        N=st.integers(32, 256),
        is_empty=st.booleans(),
        in_quantized=st.booleans(),
        out_quantized=st.booleans(),
        in_place=st.sampled_from([(False, False), (True, False), (False, True)]),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_elementwise_add_int(
        self, N, is_empty, in_quantized, out_quantized, in_place, gc, dc
    ):
        if is_empty:
            N = 0
        # FIXME: DNNLOWP Add doesn't support inplace operation and
        # dequantize_output=1 at the same time
        if in_place[0] or in_place[1]:
            in_quantized = True
            out_quantized = True

        # A has scale 1, so exactly represented after quantization
        min_ = -100
        max_ = min_ + 255
        A = np.round(np.random.rand(N) * (max_ - min_) + min_)
        A = A.astype(np.float32)
        if N != 0:
            A[0] = min_
            A[1] = max_

        # B has scale 1/2, so exactly represented after quantization
        B = np.round(np.random.rand(N) * 255 / 2 - 64).astype(np.float32)
        if N != 0:
            B[0] = -64
            B[1] = 127.0 / 2

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [("Add", ""), ("Add", "DNNLOWP"), ("Int8Add", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            do_quantize = "DNNLOWP" in engine and in_quantized
            do_dequantize = "DNNLOWP" in engine and out_quantized

            if do_quantize:
                quantize_A = core.CreateOperator(
                    "Quantize", ["A"], ["A_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize_A])

                quantize_B = core.CreateOperator(
                    "Quantize", ["B"], ["B_q"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([quantize_B])

            out = "Y"
            if in_place[0]:
                out = "A"
            elif in_place[1]:
                out = "B"

            add = core.CreateOperator(
                op_type,
                ["A_q", "B_q"] if do_quantize else ["A", "B"],
                [(out + "_q") if do_dequantize else out],
                dequantize_output=not do_dequantize,
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([add])

            if do_dequantize:
                dequantize = core.CreateOperator(
                    "Dequantize", [out + "_q"], [out], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            self.ws.create_blob("A").feed(A, device_option=gc)
            self.ws.create_blob("B").feed(B, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs[out].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)

    @given(**hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_add_broadcast(self, gc, dc):
        # Set broadcast and no axis, i.e. broadcasting last dimensions.
        min_ = -100
        max_ = min_ + 255
        A = np.round(np.random.rand(2, 3, 4, 5) * (max_ - min_) + min_)
        A = A.astype(np.float32)
        A[0, 0, 0, 0] = min_
        A[0, 0, 0, 1] = max_

        B = np.round(np.random.rand(4, 5) * 255 / 2 - 64).astype(np.float32)
        B[0, 0] = -64
        B[0, 1] = 127.0 / 2

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [("Add", ""), ("Add", "DNNLOWP"), ("Int8Add", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            add = core.CreateOperator(
                op_type,
                ["A", "B"],
                ["Y"],
                engine=engine,
                device_option=gc,
                broadcast=1,
                dequantize_output=1,
            )
            net.Proto().op.extend([add])

            self.ws.create_blob("A").feed(A, device_option=gc)
            self.ws.create_blob("B").feed(B, device_option=gc)
            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)

    @given(**hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_add_broadcast_axis(self, gc, dc):
        for bdim, axis in [
            ((3, 4), 1),  # broadcasting intermediate dimensions
            ((2,), 0),  # broadcasting the first dimension
            ((1, 4, 1), 1),
        ]:
            # broadcasting with single elem dimensions at both ends

            min_ = -100
            max_ = min_ + 255
            A = np.round(np.random.rand(2, 3, 4, 5) * (max_ - min_) + min_)
            A = A.astype(np.float32)
            B = np.round(np.random.rand(*bdim) * 255 / 2 - 64).astype(np.float32)

            A.flat[0] = min_
            A.flat[1] = max_
            B.flat[0] = -64
            B.flat[1] = 127.0 / 2

            Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
            outputs = []

            op_engine_list = [("Add", ""), ("Add", "DNNLOWP"), ("Int8Add", "DNNLOWP")]

            for op_type, engine in op_engine_list:
                net = core.Net("test_net")

                add = core.CreateOperator(
                    op_type,
                    ["A", "B"],
                    ["Y"],
                    engine=engine,
                    device_option=gc,
                    broadcast=1,
                    axis=axis,
                    dequantize_output=1,
                )
                net.Proto().op.extend([add])

                self.ws.create_blob("A").feed(A, device_option=gc)
                self.ws.create_blob("B").feed(B, device_option=gc)
                self.ws.run(net)
                outputs.append(
                    Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
                )

            check_quantized_results_close(outputs)
