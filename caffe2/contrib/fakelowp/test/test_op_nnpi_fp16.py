from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import numpy as np
import os

import caffe2.python.fakelowp.init_shared_libs  # noqa

from hypothesis import given
from hypothesis import strategies as st


from caffe2.proto import caffe2_pb2
from caffe2.python import dyndep
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

core.GlobalInit(["caffe2", "--caffe2_log_level=-3", "--glow_global_fp16=1"])

kEpsilon = 1e-8


class ArithmeticOpsTest(serial.SerializedTestCase):
    @given(seed=st.integers(0, 65534))
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
            workspace.FeedBlob("A", A)
            workspace.FeedBlob("B", B)
            # Run caffe2 net
            workspace.RunNet(pred_net_ref.name)
            Y_c2 = workspace.FetchBlob("C_ref")

            # Run Glow net
            workspace.RunNet(pred_net_onnxified.name)
            Y_glow = workspace.FetchBlob("C")

            # Results should be identical since we are comparing with the C2 emulation
            if not np.allclose(Y_c2, Y_glow):
                diff = np.abs((Y_glow - Y_c2) / (Y_c2 + kEpsilon))
                print_test_debug_info(name, {
                    "dims": dims, "A": A, "B": B,
                    "Y_glow": Y_glow, "Y_c2": Y_c2, "diff": diff})
                assert(0)

    def test_add_graph(self):
        self._test_binary_op_graph("Add")

    def test_sub_graph(self):
        self._test_binary_op_graph("Sub")

    def test_mul_graph(self):
        self._test_binary_op_graph("Mul")

    def test_div_graph(self):
        self._test_binary_op_graph("Div")


class UnaryOpTest(serial.SerializedTestCase):
    def _test_unary_op(self, opname):
        workspace.ResetWorkspace()
        n = 1
        m = 10001
        X = np.linspace(-25, 25, num=m, dtype=np.float32)
        assert 0.0 in X
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

        if not np.allclose(Y_c2, Y_glow):
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

    def test_sigmoid(self):
        self._test_unary_op("Sigmoid")

    def test_tanh(self):
        self._test_unary_op("Tanh")

    def _test_swish(self):
        self._test_unary_op("Swish")


class ReluTest(serial.SerializedTestCase):
    @given(seed=st.integers(0, 65534))
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
                "seed":seed, "X": X,
                "Y_glow": Y_glow, "Y_c2": Y_c2, "diff": diff})
            assert(0)
