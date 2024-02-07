# mypy: ignore-errors

import numpy as np
import unittest
import caffe2.python.fakelowp.init_shared_libs  # noqa

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import datetime
from hypothesis import given, settings
import hypothesis.strategies as st
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])


class TestBatchMatMul(serial.SerializedTestCase):
    @given(
        C=st.integers(min_value=1, max_value=10),
        M=st.integers(min_value=1, max_value=50),
        K=st.integers(min_value=1, max_value=512),
        N=st.integers(min_value=1, max_value=50),
        rand_seed=st.integers(0, 65534),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        run_ints=st.booleans()
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_batch_matmul(self, M, K, N, C, rand_seed, trans_a, trans_b, run_ints):
        np.random.seed(rand_seed)
        workspace.ResetWorkspace()

        batch_dims = [C]

        if run_ints:
            X = np.random.randint(low=1, high=3, size=((C, M, K))).astype(np.float32)
        else:
            X = 100 * (np.random.rand(*(batch_dims + [M, K])).astype(np.float32) - 0.5)
        if trans_a:
            X = X.swapaxes(-1, -2)

        if run_ints:
            Y = np.random.randint(low=1, high=3, size=((C, K, N))).astype(np.float32)
        else:
            Y = 100 * (np.random.rand(*(batch_dims + [K, N])).astype(np.float32) - 0.5)
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "Y"])
        pred_net.external_output.append("out")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                'BatchMatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
            )
        )

        pred_net_ref = core.Net("pred_net_ref")

        # Reference updated to fp16 with fp32 accumulation
        pred_net_ref.BatchMatMulFP16Acc32Fake(
            ["X", "Y"], ['out'], trans_a=trans_a, trans_b=trans_b)

        print("dims", batch_dims, X.shape, Y.shape)
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                {"X": X.shape, "Y": Y.shape},
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)
        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net_ref)

        # Run Glow net
        workspace.RunNet(pred_net_onnxified.name)
        out_glow = workspace.FetchBlob('out')

        # Run caffe2 net
        workspace.RunNet(pred_net_ref)
        out_c2_fakefp16 = workspace.FetchBlob('out')

        diff = np.abs(out_c2_fakefp16 - out_glow)

        if not np.allclose(out_glow, out_c2_fakefp16):
            print_test_debug_info("bmm", {
                "seed": rand_seed,
                "m": M, "k": K,
                "n": N, "X": X.shape, "Y": Y.shape,
                "trans_a": trans_a,
                "trans_b": trans_b,
                "run_ints": run_ints,
                "out_glow": out_glow,
                "out_c2_fakefp16": out_c2_fakefp16,
                "diff": diff
            })
            assert(0)


if __name__ == "__main__":
    unittest.main()
