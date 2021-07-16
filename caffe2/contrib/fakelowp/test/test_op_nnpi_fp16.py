import numpy as np

import caffe2.python.fakelowp.init_shared_libs  # noqa
import datetime
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
from caffe2.python.fakelowp.test_utils import compute_ulp_error
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])

kEpsilon = 1e-8


class ArithmeticOpsTest(serial.SerializedTestCase):
    def _test_binary_op_graph(self, name, seed):
        np.random.seed(seed)
        workspace.ResetWorkspace()
        # First dimension is the batch size
        dims = np.concatenate((np.array([1]), np.random.randint(1, 20, size=3)))
        A = np.random.uniform(low=-100.0, high=100.0, size=dims).astype(np.float32)
        B = np.random.uniform(low=-100.0, high=100.0, size=dims).astype(np.float32)
        # Avoid dividing by 0
        B[np.abs(B) < 1e-3] = 1e-3
        print(A.shape, B.shape)
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["A", "B"])
        pred_net.external_output.append("C")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                name,
                ["A", "B"],
                ["C"]
            )
        )
        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "ref"
        pred_net_ref.external_input.extend(["A", "B"])
        pred_net_ref.external_output.append("C_ref")
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                name + "FakeFp16",
                ["A", "B"],
                ["C_ref"],
            )
        )

        shape_hints = {"A": A.shape, "B": B.shape}
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                shape_hints,
                                                debug=True,
                                                adjust_batch=True,
                                                use_onnx=False)
        print(pred_net_onnxified)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.FeedBlob("A", A)
        workspace.FeedBlob("B", B)

        workspace.CreateNet(pred_net_ref)
        workspace.CreateNet(pred_net_onnxified)
        num_iterations = 10
        for _ in range(num_iterations):
            A = np.random.uniform(low=-100.0, high=100.0, size=dims).astype(np.float32)
            B = np.random.uniform(low=-100.0, high=100.0, size=dims).astype(np.float32)
            # Avoid dividing by 0
            B[np.abs(B) < 1e-3] = 1e-3

            workspace.FeedBlob("A", A)
            workspace.FeedBlob("B", B)
            # Run caffe2 net
            workspace.RunNet(pred_net_ref.name)
            Y_c2 = workspace.FetchBlob("C_ref")

            # Run Glow net
            workspace.RunNet(pred_net_onnxified.name)
            Y_glow = workspace.FetchBlob("C")

            Y_glow[Y_glow == np.Inf] = np.finfo(np.float16).max
            Y_glow[Y_glow == np.NINF] = np.finfo(np.float16).min

            # Ignore mismatches solely due to difference in precision
            fp16_finite = np.isfinite(A.astype(np.float16) / B.astype(np.float16))

            # Results should be identical since we are comparing with the C2 emulation
            if not np.allclose(Y_c2[fp16_finite], Y_glow[fp16_finite]):
                diff = np.abs((Y_glow - Y_c2) / (Y_c2 + kEpsilon))
                print_test_debug_info(name, {
                    "dims": dims, "iter": _, "seed": seed, "A": A, "B": B,
                    "Y_glow": Y_glow, "Y_c2": Y_c2, "diff": diff})
                assert(0)

    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_add_graph(self, seed):
        self._test_binary_op_graph("Add", seed)

    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_sub_graph(self, seed):
        self._test_binary_op_graph("Sub", seed)

    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_mul_graph(self, seed):
        self._test_binary_op_graph("Mul", seed)

    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_div_graph(self, seed):
        self._test_binary_op_graph("Div", seed)


class UnaryOpTest(serial.SerializedTestCase):
    def _test_unary_op(self, opname, X, rtol=1e-5, atol=1e-8):
        workspace.ResetWorkspace()

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                opname,
                ['X'],
                ['Y'])
        )
        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.append("X")
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                opname + 'FakeFp16NNPI',
                ['X'],
                ['Y'])
        )
        print("REF NET = {}".format(ref_net))

        shape_hints = {"X": X.shape}
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                shape_hints,
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(ref_net)
        workspace.CreateNet(pred_net_onnxified)
        # Run Glow net
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob('Y')
        # Run caffe2 reference net
        workspace.RunNet(ref_net.name)
        Y_c2 = workspace.FetchBlob('Y')

        if not np.allclose(Y_c2, Y_glow, rtol=atol, atol=atol):
            diff = np.abs(Y_c2 - Y_glow)
            np.save('/tmp/' + opname + 'diff', diff)
            np.save('/tmp/' + opname + 'result', Y_c2)
            print_test_debug_info(opname, {
                "X": X,
                "Y_c2": Y_c2,
                "Y_glow": Y_glow,
                "diff": diff
            })
            assert(0)

        return Y_glow

    def _test_op_w_ulp_error(self, seed, opname, regions, atol=0, err_threshold=2):
        ulp_err = 0
        for x0, x1 in regions:
            X = np.linspace(x0, x1, num=1025, dtype=np.float16).astype(np.float32)
            Y_glow = self._test_unary_op(opname, X, atol=atol)
            region_err = compute_ulp_error(opname, X, Y_glow)
            ulp_err = max(np.max(np.abs(region_err)), ulp_err)
        if (ulp_err > err_threshold):
            print(r'{} Op detected ulp_err={}'.format(opname, ulp_err))
            assert(0)

    # These tests doesn't need to run multiple times given that it is a
    # linear sweep and it is deterministic.
    # Once hypothesis.testing version is updated, we can re-enable
    # testing with different hypothesis examples.
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=20))
    def test_sigmoid(self, seed):
        np.random.seed(seed)
        opname = "Sigmoid"
        regions = [[-8., -4.], [-4., -2.], [-2., -1.], [-1., -.5], [-.5, -.25],
                   [-.25, .25], [.25, .5], [.5, 1.], [1., 2.], [2., 4.],
                   [4., 8.]]
        self._test_op_w_ulp_error(seed, opname, regions, atol=0, err_threshold=2.5)

    # These tests doesn't need to run multiple times given that it is a
    # linear sweep and it is deterministic.
    # Once hypothesis.testing version is updated, we can re-enable
    # testing with different hypothesis examples.
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=20))
    def test_tanh(self, seed):
        np.random.seed(seed)
        opname = "Tanh"
        regions = [[2.**(-9), 2.**(-8)], [2.**(-8), 2.**(-7)],
                   [2.**(-7), 2.**(-6)], [2.**(-6), 2.**(-5)],
                   [2.**(-5), 2.**(-4)], [2.**(-4), 2.**(-3)],
                   [2.**(-3), 2.**(-2)], [2.**(-2), 2.**(-1)],
                   [2.**(-1), 1.], [1., 2.], [2., 4.], [4., 8.]]
        self._test_op_w_ulp_error(seed, opname, regions, atol=0, err_threshold=2)

    # These tests doesn't need to run multiple times given that it is a
    # linear sweep and it is deterministic.
    # Once hypothesis.testing version is updated, we can re-enable
    # testing with different hypothesis examples.
    # TODO: move atol to 1e-8 once we get a non-lowered swish implementation
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_swish(self, seed):
        np.random.seed(seed)
        opname = "Swish"
        regions = [[-20.5, -11.], [-11., -8.], [-8., -1.], [-1., -0.1],
                   [-1. / 8., 1. / 8.], [1. / 8, 5.], [5., 8.]]
        self._test_op_w_ulp_error(seed, opname, regions, atol=0.008, err_threshold=384)

    # These tests doesn't need to run multiple times given that it is a
    # linear sweep and it is deterministic.
    # Once hypothesis.testing version is updated, we can re-enable
    # testing with different hypothesis examples.
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_logit(self, seed):
        np.random.seed(seed)
        workspace.ResetWorkspace()
        n = 1
        m = 15361
        X = np.linspace(0, 1, num=m, dtype=np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                'Logit',
                ['X'],
                ['Y'],
                eps=1e-6)
        )
        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.append("X")
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                'LogitFakeFp16NNPI',
                ['X'],
                ['Y'],
                eps=1e-6)
        )
        print("REF NET = {}".format(ref_net))

        shape_hints = {"X": (n, m)}
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                shape_hints,
                                                debug=True,
                                                adjust_batch=False,
                                                use_onnx=False)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(ref_net)
        workspace.CreateNet(pred_net_onnxified)
        # Run Glow net
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob('Y')
        # Run caffe2 reference net
        workspace.RunNet(ref_net.name)
        Y_c2 = workspace.FetchBlob('Y')

        diff = np.abs(Y_c2 - Y_glow)
        if np.nanmax(diff) > 9e-3:
            np.save('/tmp/logit_diff', diff)
            np.save('/tmp/logit_result', Y_c2)
            print_test_debug_info('Logit', {
                "X": X,
                "Y_c2": Y_c2,
                "Y_glow": Y_glow,
                "diff": diff
            })
            assert(0)

class ReluTest(serial.SerializedTestCase):
    @given(seed=st.integers(0, 65534))
    @settings(deadline=datetime.timedelta(seconds=10))
    def relu_test(self, inputs, gc, dc, seed):
        np.random.seed(seed)
        inputs = np.random.rand(1).astype(np.float32)
        X = inputs[0]
        # First dimension is the batch size
        print(X.shape)
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["X"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Relu",
                ["X"],
                ["Y"]
            )
        )
        pred_net_ref = caffe2_pb2.NetDef()
        pred_net_ref.name = "ref"
        pred_net_ref.external_input.extend(["X"])
        pred_net_ref.external_output.append("Y_ref")
        pred_net_ref.op.add().CopyFrom(
            core.CreateOperator(
                "ReluFakeFp16",
                ["X"],
                ["Y_ref"],
            )
        )

        shape_hints = {"X": X.shape}
        pred_net_onnxified = onnxifi_caffe2_net(pred_net,
                                                shape_hints,
                                                debug=True,
                                                adjust_batch=True,
                                                use_onnx=False)
        print(pred_net_onnxified)
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.SwitchWorkspace("glow_test_ws", True)
        workspace.FeedBlob("X", X)

        workspace.CreateNet(pred_net_ref)
        workspace.CreateNet(pred_net_onnxified)
        workspace.FeedBlob("X", X)
        # Run caffe2 net
        workspace.RunNet(pred_net_ref.name)
        Y_c2 = workspace.FetchBlob("Y_ref")

        # Run Glow net
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        # Results should be identical since we are comparing with the C2 emulation
        if not np.allclose(Y_c2, Y_glow):
            diff = np.abs((Y_glow - Y_c2) / (Y_c2 + kEpsilon))
            print_test_debug_info("Relu", {
                "seed": seed, "X": X,
                "Y_glow": Y_glow, "Y_c2": Y_c2, "diff": diff})
            assert(0)
