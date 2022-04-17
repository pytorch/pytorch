import numpy as np
import unittest

import caffe2.python.fakelowp.init_shared_libs  # noqa
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import datetime
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])

GLOW_MATMUL_RTOL = 0


class FCTest(serial.SerializedTestCase):
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_clip(self, seed):
        np.random.seed(seed)
        m, n, k = 8, 8, 8
        dtype = np.float32
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "W0", "b0", "W1", "b1"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "FC",
                ["X", "W0", "b0"],
                ["X1"],
            )
        )
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "FC",
                ["X1", "W1", "b1"],
                ["Y"],
            )
        )
        workspace.GlobalInit(
            ['caffe2', '--caffe2_log_level=0', '--glow_global_fp16=1',
             '--glow_clip_fp16', '--glow_global_fp16_constants=1'])
        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.ResetWorkspace()
        W0 = np.full((n, k), 65536.0, dtype)
        b0 = np.random.randint(low=1, high=3, size=(n)).astype(dtype)
        W1 = np.random.randint(low=1, high=3, size=(n, k)).astype(dtype)
        b1 = np.random.randint(low=1, high=3, size=(n)).astype(dtype)
        workspace.FeedBlob("W0", W0)
        workspace.FeedBlob("b0", b0)
        workspace.FeedBlob("W1", W1)
        workspace.FeedBlob("b1", b1)

        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": (m, k)},
            debug=True,
            adjust_batch=False,
            use_onnx=False
        )

        X = np.random.randint(low=1, high=3, size=(m, k)).astype(dtype)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(pred_net_onnxified)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")
        np.testing.assert_allclose(Y_glow, np.full((m, n), 65504.0, dtype))

    @given(
        m=st.integers(4, 50),
        k=st.integers(4, 50),
        n=st.integers(4, 50),
        seed=st.integers(0, 65534)
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_fc_exercise(self, m, k, n, seed):
        """ Test that the matmul engine is working, this doesn't test
            precision
        """
        np.random.seed(seed)
        dtype = np.float32
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "W0", "b0"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "FC",
                ["X", "W0", "b0"],
                ["Y"],
            )
        )

        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.ResetWorkspace()
        W0 = np.random.randint(low=1, high=3, size=(n, k)).astype(dtype)
        b0 = np.random.randint(low=1, high=3, size=(n)).astype(dtype)
        workspace.FeedBlob("W0", W0)
        workspace.FeedBlob("b0", b0)

        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                {"X": (m, k)},
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        X0 = np.random.randint(low=1, high=3, size=(m, k)).astype(dtype)
        workspace.FeedBlob("X", X0)
        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net)

        num_iterations = 2
        for _ in range(num_iterations):
            X0 = np.random.randint(low=1, high=3, size=(m, k)).astype(dtype)
            workspace.FeedBlob("X", X0)
            # Run Glow net
            workspace.RunNet(pred_net_onnxified.name)
            Y_glow = workspace.FetchBlob('Y')
            # Run caffe2 net
            workspace.RunNet(pred_net.name)
            Y_c2 = workspace.FetchBlob('Y')
            if not np.allclose(Y_c2, Y_glow):
                print_test_debug_info("fc", {
                    "seed": seed,
                    "m": m,
                    "k": k,
                    "n": n,
                    "X": X0,
                    "W0": W0,
                    "b0": b0,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "diff": np.abs((Y_c2 - Y_glow) / Y_c2)})
                assert(0)

    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_fc_numeric_cases(self, seed):
        """ Test numerics, use examples found from the unit test.
            Use Fp16FCAcc16NNPI as a reference.
        """
        np.random.seed(seed)
        m = 1
        k = 20
        n = 1
        dtype = np.float32
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", "W0", "b0"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "FC",
                ["X", "W0", "b0"],
                ["Y"],
            )
        )
        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred"
        pred_net_ref.external_input.extend(["X", "W0", "b0"])
        pred_net_ref.external_output.append("Y")
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "Fp16FCAcc32NNPI",
                ["X", "W0", "b0"],
                ["Y"],
            )
        )

        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.ResetWorkspace()

        W0 = np.array([[0.04882812, 0.21520996, 0.1027832, 0.04489136,
                        -0.07635498, 0.14587402,
                        -0.06240845, 0.3918457, 0.46362305, -0.11657715,
                        0.29174805, 0.02890015,
                        0.0680542, 0.4255371, -0.42895508, -0.4128418,
                        -0.47973633, 0.33251953,
                        0.27807617, 0.3701172]], dtype=np.float32)
        b0 = np.array([0.47851562], dtype=np.float32)

        workspace.FeedBlob("W0", W0)
        workspace.FeedBlob("b0", b0)

        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                {"X": (m, k)},
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        X_inputs = [
            np.array([[
                -2.94921875e-01, -3.58642578e-01, -1.92871094e-01,
                2.81250000e-01, -1.30126953e-01, 2.32696533e-02,
                -4.55566406e-01, -2.31811523e-01, -1.95190430e-01,
                -7.76977539e-02, -1.29394531e-01, 2.94677734e-01,
                8.96453857e-04, 4.97314453e-01, -6.07604980e-02,
                2.55371094e-01, 3.49853516e-01, -1.37695312e-01,
                2.95410156e-01, -3.67187500e-01]], dtype=np.float32),
            np.array([[
                -0.4494629, -0.22192383, -0.1640625, 0.11480713,
                -0.09851074, -0.02084351,
                0.19091797, -0.17468262, -0.47485352, 0.07489014,
                0.03897095, 0.00197601,
                0.02835083, -0.27294922, 0.26757812, -0.20996094,
                -0.31103516, -0.41601562,
                0.09918213, -0.07696533]], dtype=np.float32),
            np.array([[
                0.01150513, -0.20507812, 0.46704102, 0.00906372,
                0.19848633, 0.3720703,
                0.46557617, -0.47436523, -0.35107422, -0.0362854,
                -0.20812988, 0.41918945,
                0.09716797, 0.19897461, 0.3876953, -0.0165863,
                0.23535156, 0.29956055,
                0.24389648, -0.23486328]], dtype=np.float32)
        ]

        # keep onnxifi happy by feeding something with a shape
        workspace.FeedBlob("X", X_inputs[0])
        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net_ref)

        for i in range(len(X_inputs)):
            workspace.FeedBlob("X", X_inputs[i])
            # Run Glow net
            workspace.RunNet(pred_net_onnxified.name)
            Y_glow = workspace.FetchBlob('Y')
            workspace.RunNet(pred_net_ref.name)
            Y_c2 = workspace.FetchBlob('Y')

            diff = np.abs((Y_c2 - Y_glow) / (Y_c2 + 1e-8))
            rowdiff = np.max(diff, axis=1)

            n_offenders = np.count_nonzero(rowdiff[rowdiff > GLOW_MATMUL_RTOL])
            if n_offenders > 0:
                print_test_debug_info("fc", {
                    "seed": seed,
                    "iter": i,
                    "m": m,
                    "k": k,
                    "n": n,
                    "W0": W0,
                    "b0": b0,
                    "Y_glow": Y_glow,
                    "Y_c2": Y_c2,
                    "diff": diff,
                    "rowdiff": rowdiff})
                assert(0)

    @given(
        m=st.integers(1, 50),
        k=st.integers(1, 1000),
        n=st.integers(1, 50),
        seed=st.integers(0, 65534),
        use_packed=st.integers(0, 2)
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_fc_num0(self, seed, m, k, n, use_packed):
        """ Test numerics, fix a dimension and determine the ranges of error.
            Use Fp16FCAcc16 as a reference.
        """
        W = "W_packed" if use_packed else "W0"
        dtype = np.float32
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X", W, "b0"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "FbFCPacked" if use_packed else "FC",
                ["X", W, "b0"],
                ["Y"],
            )
        )
        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "pred"
        pred_net_ref.external_input.extend(["X", W, "b0"])
        pred_net_ref.external_output.append("Y")
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "Fp16FCAcc32NNPI",
                ["X", W, "b0"],
                ["Y"],
            )
        )

        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.ResetWorkspace()
        W0 = 10 * (np.random.rand(n, k) - 0.5).astype(np.float16).astype(np.float32)
        b0 = 1 * (np.random.rand(n) - 0.5).astype(np.float16).astype(np.float32)

        workspace.FeedBlob("W0", W0)
        workspace.FeedBlob("b0", b0)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FbGemmPack",
                ['W0'],
                ['W_packed'],
                no_packing=True,
            )
        )

        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                {"X": (m, k)},
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        X0 = np.random.rand(m, k).astype(dtype) - 0.5
        workspace.FeedBlob("X", X0)
        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(pred_net_ref)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob('Y')

        # Run caffe2 net
        workspace.RunNet(pred_net_ref.name)
        Y_c2 = workspace.FetchBlob('Y')

        diff = np.abs((Y_c2 - Y_glow) / (Y_c2 + 1e-8))
        rowdiff = np.max(diff, axis=1)

        n_offenders = np.count_nonzero(rowdiff[rowdiff > GLOW_MATMUL_RTOL])
        if n_offenders > 0:
            print_test_debug_info("fc", {
                "seed": seed,
                "use_packed": use_packed,
                "m": m,
                "k": k,
                "n": n,
                "X": X0.shape,
                "W0": W0.shape,
                "b0": b0.shape,
                "Y_glow": Y_glow,
                "Y_c2": Y_c2,
                "diff": diff,
                "rowdiff": rowdiff})
            assert(0)

if __name__ == '__main__':
    unittest.main()
