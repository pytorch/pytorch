from __future__ import absolute_import, division, print_function, unicode_literals

import collections
from itertools import product

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from caffe2.quantization.server import utils as dnnlowp_utils
from dnnlowp_test_utils import (
    avoid_vpmaddubsw_overflow_fc,
    check_quantized_results_close,
)
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class DNNLowPBatchMatMulOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(
        m=st.integers(0, 32),
        n=st.integers(4, 32),
        k=st.integers(4, 32),
        batch_size=st.integers(0, 4),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_batch_matmul_int(self, m, n, k, batch_size, gc, dc):
        # A and B have scale 1, so exactly represented after quantization
        A_min = -77
        A_max = A_min + 255
        A = np.round(np.random.rand(batch_size, m, k) * 255 + A_min)
        A = A.astype(np.float32)
        # input channels 0 and 1 are all A_min to avoid overflow from vpmaddubsw
        # when multiplied with B_min and B_max
        if batch_size > 0 and m > 0:
            A[0, :, 0] = A_min
            A[0, 0, 1] = A_max

        B_min = -100
        B_max = B_min + 255
        B = np.round(np.random.rand(batch_size, n, k) * 255 + B_min)
        B = B.astype(np.float32)
        if batch_size > 0:
            B[0, 0, 0] = B_min
            B[0, 1, 0] = B_max

        for i in range(batch_size):
            avoid_vpmaddubsw_overflow_fc(
                m, k, n, A[i,], A_min, A_max, B[i,], B_min, B_max
            )

        for trans_a, trans_b in product([0, 1], [0, 1]):
            Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
            outputs = []

            op_engine_list = [
                ("BatchMatMul", ""),
                ("BatchMatMul", "DNNLOWP"),
                ("BatchMatMul", "DNNLOWP_16"),
                ("Int8BatchMatMul", "DNNLOWP"),
            ]

            for op_type, engine in op_engine_list:
                net = core.Net("test_net")

                if "DNNLOWP" in engine:
                    quantize_A = core.CreateOperator(
                        "Quantize", ["A"], ["A_q"], engine=engine, device_option=gc
                    )
                    net.Proto().op.extend([quantize_A])

                    quantize_B = core.CreateOperator(
                        "Quantize", ["B"], ["B_q"], engine=engine, device_option=gc
                    )
                    net.Proto().op.extend([quantize_B])

                batch_matmul = core.CreateOperator(
                    op_type,
                    [
                        "A_q" if "DNNLOWP" in engine else "A",
                        "B_q" if "DNNLOWP" in engine else "B",
                    ],
                    ["Y_q" if "DNNLOWP" in engine else "Y"],
                    trans_a=trans_a,
                    trans_b=trans_b,
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([batch_matmul])

                if "DNNLOWP" in engine:
                    dequantize = core.CreateOperator(
                        "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                    )
                    net.Proto().op.extend([dequantize])

                self.ws.create_blob("A").feed(
                    np.transpose(A, (0, 2, 1)) if trans_a else A, device_option=gc
                )
                self.ws.create_blob("B").feed(
                    B if trans_b else np.transpose(B, (0, 2, 1)), device_option=gc
                )
                self.ws.run(net)
                outputs.append(
                    Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
                )

            check_quantized_results_close(outputs)

    # correctness test with no quantization error in inputs
    @given(
        m=st.integers(0, 32),
        n=st.integers(4, 32),
        k=st.integers(4, 32),
        C_1=st.integers(0, 3),  # number of batch dims
        C_2=st.integers(0, 3),
        A_quantized=st.booleans(),
        B_quantized=st.booleans(),
        out_quantized=st.booleans(),
        **hu.gcs_cpu_only
    )
    def test_dnnlowp_batch_matmul_int_constant_B(
        self, m, n, k, C_1, C_2, A_quantized, B_quantized, out_quantized, gc, dc
    ):
        batch_dims = tuple(np.random.randint(3, size=max(C_1, C_2)))
        batch_dims_A = batch_dims[-C_1:]
        batch_dims_B = batch_dims[-C_2:]
        A = np.zeros(batch_dims_A + (m, k)).astype(np.float32)
        B = np.zeros(batch_dims_B + (n, k)).astype(np.float32)

        if np.prod(batch_dims) > 0:
            for index in np.ndindex(batch_dims_A):
                # When both input and output are float, each input of the batch has
                # scale 1 but with different offset, so input-wise quantization
                # shouldn't have any input quantization error
                # A_min = -77 if (A_quantized or out_quantized) else -77 + i
                A_min = -77
                A_max = A_min + 255
                A[index] = np.round(np.random.rand(m, k) * 255 + A_min)
                # input channels 0 and 1 are all A_min to avoid overflow from vpmaddubsw
                # when multiplied with B_min and B_max
                A[index][:, 0] = A_min
                if m != 0:
                    A[index][0, 1] = A_max

            i = 0
            for index in np.ndindex(batch_dims_B):
                # When weight is quantized in a lazy manner, each input of the batch has
                # scale 1 but with different offset, so input-wise quantization
                # shouldn't have any input quantization error when weight is quantized
                # in a lazy manner.
                B_min = -100 if B_quantized else -100 + i
                # B_min = -100
                B_max = B_min + 255
                B[index] = np.round(np.random.rand(n, k) * 255 + B_min)
                B[index][0, 0] = B_min
                B[index][1, 0] = B_max

                if C_1 > C_2:
                    # A has more dims
                    for outer_index in np.ndindex(batch_dims_A[: C_1 - C_2]):
                        avoid_vpmaddubsw_overflow_fc(
                            m,
                            k,
                            n,
                            A[outer_index] if C_2 == 0 else A[outer_index + index],
                            A_min,
                            A_max,
                            B[index],
                            B_min,
                            B_max,
                        )
                else:
                    avoid_vpmaddubsw_overflow_fc(
                        m, k, n, A[index[-C_1:]], A_min, A_max, B[index], B_min, B_max
                    )
                i += 1

        for trans_a, trans_b in product([0, 1], [0, 1]):
            Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
            outputs = []

            op_engine_list = [
                ("BatchMatMul", ""),
                ("BatchMatMul", "DNNLOWP"),
                ("Int8BatchMatMul", "DNNLOWP"),
            ]

            for op_type, engine in op_engine_list:
                net = core.Net("test_net")

                do_quantize_A = "DNNLOWP" in engine and A_quantized
                do_quantize_B = "DNNLOWP" in engine and B_quantized
                do_dequantize = "DNNLOWP" in engine and out_quantized

                if do_quantize_A:
                    quantize_A = core.CreateOperator(
                        "Quantize", ["A"], ["A_q"], engine=engine, device_option=gc
                    )
                    net.Proto().op.extend([quantize_A])

                if do_quantize_B:
                    int8_given_tensor_fill, B_q_param = dnnlowp_utils.create_int8_given_tensor_fill(
                        B if trans_b else B.swapaxes(-1, -2), "B_q"
                    )
                    net.Proto().op.extend([int8_given_tensor_fill])

                batch_matmul = core.CreateOperator(
                    op_type,
                    ["A_q" if do_quantize_A else "A", "B_q" if do_quantize_B else "B"],
                    ["Y_q" if do_dequantize else "Y"],
                    trans_a=trans_a,
                    trans_b=trans_b,
                    broadcast=True,
                    constant_B=True,
                    dequantize_output=not do_dequantize,
                    engine=engine,
                    device_option=gc,
                )
                if do_quantize_B:
                    # When quantized weight is provided, we can't rescale the
                    # output dynamically by looking at the range of output of each
                    # batch, so here we provide the range of output observed from
                    # fp32 reference implementation
                    dnnlowp_utils.add_quantization_param_args(
                        batch_matmul, outputs[0][0]
                    )
                net.Proto().op.extend([batch_matmul])

                if do_dequantize:
                    dequantize = core.CreateOperator(
                        "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                    )
                    net.Proto().op.extend([dequantize])

                self.ws.create_blob("A").feed(
                    A.swapaxes(-1, -2) if trans_a else A, device_option=gc
                )
                self.ws.create_blob("B").feed(
                    B if trans_b else B.swapaxes(-1, -2), device_option=gc
                )
                self.ws.run(net)
                outputs.append(
                    Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
                )

            if np.prod(batch_dims) > 0:
                check_quantized_results_close(outputs)
