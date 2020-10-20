import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import time
from caffe2.python import core, dyndep, workspace
from caffe2.quantization.server import dnnlowp_pybind11
from hypothesis import given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")
workspace.GlobalInit(["caffe2", "--caffe2_omp_num_threads=11"])


class FusedMulInt8QuantizeOpTest(hu.HypothesisTestCase):

    @settings(max_examples=10, deadline=None)
    @given(m=st.integers(8192, 8192),
           n=st.integers(8192, 8192))
    def test_fused_mul_int8_quantize(self, m, n):
        # this test is to verfy the correctness of FusedMulInt8QuantizeOp
        # it should provide equivalent quantized result as separate Mul + Int8 Ops
        min_ = -10.0
        max_ = 20.0

        X = (np.random.rand(m, n) * (max_ - min_) + min_).astype(np.float32)
        X_min = 0 if X.size == 0 else X.min()
        X_max = 1 if X.size == 0 else X.max()
        X_scale = (max(X_max, 0) - min(X_min, 0)) / 255
        X_zero = np.round(-X_min / X_scale)

        # generate random equalization vector S in range [0,1]
        S = np.random.rand(X.shape[1]).astype(np.float32)

        X_dq, runtime = [], []

        for fused in (True, False):
            # test fused/unfused operators
            net = core.Net("test_net")
            dnnlowp_pybind11.CreateInt8QuantParamsBlob(
                "quant_param", float(X_scale), int(X_zero)
            )
            workspace.FeedBlob("X", X)
            workspace.FeedBlob("S", S)
            ops = []
            if fused:
                fuse = core.CreateOperator(  # fused broadcast mul S to X then int8 quant
                    "FusedEqualizeInt8Quantize",
                    ["X", "S", "quant_param"],
                    ["X_q"],
                    engine="DNNLOWP",
                )
                ops = [fuse]
            else:
                mul = core.CreateOperator(  # in-place broadcast mul S to each column of X
                    "Mul",
                    ["X", "S"],
                    ["X_equalized"],
                    broadcast=1,
                    axis=1,
                )
                quantize = core.CreateOperator(  # int8 quant on equalized X
                    "Quantize",
                    ["X_equalized", "quant_param"],
                    ["X_q"],
                    engine="DNNLOWP",
                )
                ops = [mul, quantize]

            net.Proto().op.extend(ops)
            workspace.CreateNet(net)
            start_time = time.time()
            workspace.RunNet(net)
            rt = time.time() - start_time
            # print(f"--- {fused} finished in {rt:.2f} sec ---")

            # runtimes = workspace.BenchmarkNet(net.Name(), 1, 1024, True)

            X_q = workspace.FetchInt8Blob("X_q")[0]

            # Dequantize fused and unfused results
            X_dq.append(X_scale * (X_q - X_zero))
            runtime.append(rt)

        # check fused/unfused X_dq are close
        np.testing.assert_allclose(X_dq[0], X_dq[1], atol=1e-3, rtol=1e-3)
