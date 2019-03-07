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


class DNNLowPOpSumOpTest(hu.HypothesisTestCase):
    # correctness test with no quantization error in inputs
    @given(N=st.integers(32, 256), M=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_sum_int(self, N, M, gc, dc):
        # All inputs have scale 1, so exactly represented after quantization
        inputs = M * [None]
        X_names = M * [None]
        X_q_names = M * [None]

        for i in range(M):
            X = np.random.randint(-128, 127, N, np.int8).astype(np.float32)
            X[0] = -128
            X[-1] = 127
            inputs[i] = X
            X_names[i] = chr(ord("A") + i)
            X_q_names[i] = X_names[i] + "_q"

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [("Sum", ""), ("Sum", "DNNLOWP"), ("Int8Sum", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                for i in range(M):
                    quantize = core.CreateOperator(
                        "Quantize",
                        X_names[i],
                        X_q_names[i],
                        engine=engine,
                        device_option=gc,
                    )
                    net.Proto().op.extend([quantize])

            sum_ = core.CreateOperator(
                op_type,
                X_q_names if engine == "DNNLOWP" else X_names,
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([sum_])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            for i in range(M):
                self.ws.create_blob(X_names[i]).feed(X, device_option=gc)

            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)

    # correctness test with no quantization error in inputs
    @given(N=st.integers(32, 256), M=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_sum_int_inplace(self, N, M, gc, dc):
        # All inputs have scale 1, so exactly represented after quantization
        inputs = M * [None]
        X_names = M * [None]
        X_q_names = M * [None]

        for i in range(M):
            X = np.random.randint(-128, 127, N, np.int8).astype(np.float32)
            X[0] = -128
            X[-1] = 127
            inputs[i] = X
            X_names[i] = chr(ord("A") + i)
            X_q_names[i] = X_names[i] + "_q"

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [("Sum", ""), ("Sum", "DNNLOWP"), ("Int8Sum", "DNNLOWP")]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                for i in range(M):
                    quantize = core.CreateOperator(
                        "Quantize",
                        X_names[i],
                        X_q_names[i],
                        engine=engine,
                        device_option=gc,
                    )
                    net.Proto().op.extend([quantize])

            sum_ = core.CreateOperator(
                op_type,
                X_q_names if engine == "DNNLOWP" else X_names,
                [X_q_names[0] if engine == "DNNLOWP" else X_names[0]],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([sum_])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize",
                    [X_q_names[0]],
                    [X_names[0]],
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize])

            for i in range(M):
                self.ws.create_blob(X_names[i]).feed(X, device_option=gc)

            self.ws.run(net)
            outputs.append(
                Output(
                    Y=self.ws.blobs[X_names[0]].fetch(), op_type=op_type, engine=engine
                )
            )

        check_quantized_results_close(outputs)

    # correctness test with no quantization error in inputs
    @given(N=st.integers(32, 256), M=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_sum_relu_int(self, N, M, gc, dc):
        # All inputs have scale 1, so exactly represented after quantization
        inputs = M * [None]
        X_names = M * [None]
        X_q_names = M * [None]

        for i in range(M):
            X = np.random.randint(-128, 127, N, np.int8).astype(np.float32)
            X[0] = -128
            X[-1] = 127
            inputs[i] = X
            X_names[i] = chr(ord("A") + i)
            X_q_names[i] = X_names[i] + "_q"

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("SumRelu", ""),
            ("SumRelu", "DNNLOWP"),
            ("Int8SumRelu", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                for i in range(M):
                    quantize = core.CreateOperator(
                        "Quantize",
                        X_names[i],
                        X_q_names[i],
                        engine=engine,
                        device_option=gc,
                    )
                    net.Proto().op.extend([quantize])

            sum_relu = core.CreateOperator(
                op_type,
                X_q_names if engine == "DNNLOWP" else X_names,
                ["Y_q" if engine == "DNNLOWP" else "Y"],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([sum_relu])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize", ["Y_q"], ["Y"], engine=engine, device_option=gc
                )
                net.Proto().op.extend([dequantize])

            for i in range(M):
                self.ws.create_blob(X_names[i]).feed(X, device_option=gc)

            self.ws.run(net)
            outputs.append(
                Output(Y=self.ws.blobs["Y"].fetch(), op_type=op_type, engine=engine)
            )

        check_quantized_results_close(outputs)

    # correctness test with no quantization error in inputs
    @given(N=st.integers(32, 256), M=st.integers(1, 3), **hu.gcs_cpu_only)
    def test_dnnlowp_elementwise_sum_relu_int_inplace(self, N, M, gc, dc):
        # All inputs have scale 1, so exactly represented after quantization
        inputs = M * [None]
        X_names = M * [None]
        X_q_names = M * [None]

        for i in range(M):
            X = np.random.randint(-128, 127, N, np.int8).astype(np.float32)
            X[0] = -128
            X[-1] = 127
            inputs[i] = X
            X_names[i] = chr(ord("A") + i)
            X_q_names[i] = X_names[i] + "_q"

        Output = collections.namedtuple("Output", ["Y", "op_type", "engine"])
        outputs = []

        op_engine_list = [
            ("SumRelu", ""),
            ("SumRelu", "DNNLOWP"),
            ("Int8SumRelu", "DNNLOWP"),
        ]

        for op_type, engine in op_engine_list:
            net = core.Net("test_net")

            if engine == "DNNLOWP":
                for i in range(M):
                    quantize = core.CreateOperator(
                        "Quantize",
                        X_names[i],
                        X_q_names[i],
                        engine=engine,
                        device_option=gc,
                    )
                    net.Proto().op.extend([quantize])

            sum_relu = core.CreateOperator(
                op_type,
                X_q_names if engine == "DNNLOWP" else X_names,
                [X_q_names[0] if engine == "DNNLOWP" else X_names[0]],
                engine=engine,
                device_option=gc,
            )
            net.Proto().op.extend([sum_relu])

            if engine == "DNNLOWP":
                dequantize = core.CreateOperator(
                    "Dequantize",
                    [X_q_names[0]],
                    [X_names[0]],
                    engine=engine,
                    device_option=gc,
                )
                net.Proto().op.extend([dequantize])

            for i in range(M):
                self.ws.create_blob(X_names[i]).feed(X, device_option=gc)

            self.ws.run(net)
            outputs.append(
                Output(
                    Y=self.ws.blobs[X_names[0]].fetch(), op_type=op_type, engine=engine
                )
            )

        check_quantized_results_close(outputs)
