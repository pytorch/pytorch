




from hypothesis import given, settings
import hypothesis.strategies as st
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util, workspace
from caffe2.python.core import CreateOperator, GradientRegistry, IR

import numpy as np


# First, we will set up a few gradient registry entries so that we can manually
# construct some test cases.


def NeedAll(op, g_output):
    """A sanity check to make sure that all the gradient are given."""
    for name, g in zip(op.output, g_output):
        if g is None:
            raise RuntimeError(
                'Need gradient for "%s" but it is not provided.' % name)
    return g_output


def GIS(op):
    """A test util function to generate the gradient name for input."""
    return [s + '_grad' for s in op.input]


def CopyDeviceOption(op, src_op):
    if src_op.HasField('device_option'):
        op.device_option.CopyFrom(src_op.device_option)
    return op


# First gradient: (in -> out) leading to (out_grad -> in_grad)
@GradientRegistry.RegisterGradient('Direct')
def AddDirectGradient(op, g_output):
    return (
        CopyDeviceOption(
            CreateOperator('DirectGradient', NeedAll(op, g_output), GIS(op)),
            op),
        GIS(op)
    )


# Second gradient: (in -> out) leading to (out, out_grad -> in_grad)
@GradientRegistry.RegisterGradient('UseOutput')
def AddUseOutputGradient(op, g_output):
    return (
        CopyDeviceOption(
            CreateOperator(
                'UseOutputGradient',
                list(op.output) + NeedAll(op, g_output), GIS(op)),
            op),
        GIS(op)
    )


@GradientRegistry.RegisterGradient('UseInput')
def AddUseInputGradient(op, g_output):
    return (
        CopyDeviceOption(
            CreateOperator(
                'UseInputGradient',
                list(op.input) + NeedAll(op, g_output), GIS(op)),
            op),
        GIS(op)
    )


@GradientRegistry.RegisterGradient('Nogradient')
def AddNogradient(op, g_output):
    return (
        [],
        [None for s in op.input]
    )


class TestGradientCalculation(test_util.TestCase):
    def assertOperatorListEqual(self, operatorDefList1, operatorDefList2):
        for op in operatorDefList1:
            op.debug_info = ""
            if op.device_option:
                del op.device_option.extra_info[:]
        for op in operatorDefList2:
            op.debug_info = ""
            if op.device_option:
                del op.device_option.extra_info[:]
        self.assertEqual(operatorDefList1, operatorDefList2)

    @given(device_option=st.sampled_from([
        None,
        core.DeviceOption(workspace.GpuDeviceType, 1)]))
    @settings(deadline=10000)
    def testDirect(self, device_option):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        if device_option:
            for op in operators:
                op.device_option.CopyFrom(device_option)
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'out_grad', 'hidden_grad'),
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        if device_option:
            for op in desired_grad_operators:
                op.device_option.CopyFrom(device_option)
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testDirectImplicitGradientSource(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator(
                "ConstantFill", 'out', "out_autogen_grad", value=1.0),
            CreateOperator(
                'DirectGradient', 'out_autogen_grad', 'hidden_grad'),
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        for op in desired_grad_operators:
            op.debug_info = ""
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, ['out'])
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testDoesNotGenerateUnnecessaryGradients(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        for op in desired_grad_operators:
            op.debug_info = ""
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'hidden': 'hidden_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testDirectButNoOutputGradientGiven(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {})
        self.assertOperatorListEqual(gradients, [])

    def testDirectInPlace(self):
        operators = [
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Direct', 'in', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'out_grad', 'in_grad'),
            CreateOperator('DirectGradient', 'in_grad', 'in_grad'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testVersionMismatch(self):
        operators = [
            CreateOperator('Direct', 'x', 'x'),
            CreateOperator('Direct', 'y', 'x'),
            CreateOperator('Direct', 'x', 'y'),
        ]
        try:
            gradients, _ = GradientRegistry.GetBackwardPass(
                operators, {'y': 'y_grad'})
            self.assertFalse(True, "Should raise exception of incorrect version")
        except RuntimeError as e:
            print(e)
            self.assertTrue("version" in str(e))
            pass

    def testUseOutput(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'hidden'),
            CreateOperator('UseOutput', 'hidden', 'out'),
            CreateOperator('Direct', 'out', 'sink'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'sink_grad', 'out_grad'),
            CreateOperator(
                'UseOutputGradient',
                ['out', 'out_grad'], 'hidden_grad'
            ),
            CreateOperator(
                'UseOutputGradient',
                ['hidden', 'hidden_grad'], 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'sink': 'sink_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testUseOutputInPlace(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'in'),
            CreateOperator('UseOutput', 'in', 'out'),
            CreateOperator('Direct', 'out', 'sink'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'sink_grad', 'out_grad'),
            CreateOperator(
                'UseOutputGradient',
                ['out', 'out_grad'], 'in_grad'
            ),
            CreateOperator(
                'UseOutputGradient',
                ['in', 'in_grad'], 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'sink': 'sink_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testUseOutputButOutputHasBeenChanged(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'hidden'),
            # Note here: we overwrite hidden, but hidden will be needed by the
            # gradient calculation of the first operator, so the gradient
            # registry should return an error.
            CreateOperator('Direct', 'hidden', 'hidden'),
            CreateOperator('UseOutput', 'hidden', 'out'),
            CreateOperator('Direct', 'out', 'sink'),
        ]
        with self.assertRaises(RuntimeError):
            gradients, _ = GradientRegistry.GetBackwardPass(
                operators, {'sink': 'sink_grad'})

    def testUseInput(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('UseInput', 'hidden', 'out'),
            CreateOperator('Direct', 'out', 'sink'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'sink_grad', 'out_grad'),
            CreateOperator(
                'UseInputGradient',
                ['hidden', 'out_grad'], 'hidden_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'sink': 'sink_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testUseInputButInputHasBeenChanged(self):
        """Test gradient for the following case:

        in -> out, with UseInput
        in -> in

        Since we overwrite in in op#1, but in will be needed by the gradient
        calculation of op#0, the gradient registry should raise an error.
        """
        operators = [
            CreateOperator('UseInput', 'in', 'out'),
            CreateOperator('Direct', 'in', 'in'),
        ]
        with self.assertRaises(RuntimeError):
            gradients, _ = GradientRegistry.GetBackwardPass(
                operators, {'out': 'out_grad'})

    @given(device_option=st.sampled_from([
        None,
        core.DeviceOption(workspace.GpuDeviceType, 1)]))
    @settings(deadline=10000)
    def testMultiUseInput(self, device_option):
        """Test gradient for the following case:

        in -> hidden1
        in -> hidden2
        hidden1, hidden2 -> out
        """
        operators = [
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Direct', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
        ]
        if device_option:
            for op in operators:
                op.device_option.CopyFrom(device_option)
        desired_grad_operators = [
            CreateOperator(
                'DirectGradient',
                'out_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden2_grad', 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'Sum',
                ['in_grad', '_in_grad_autosplit_0'], 'in_grad'
            ),
        ]
        if device_option:
            for op in desired_grad_operators:
                op.device_option.CopyFrom(device_option)
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {"out": "out_grad"})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testMultiUseInputButWithNoGradient(self):
        """Test gradient for the following case:

        in -> hidden1
        in -(no gradient)-> hidden2
        hidden1, hidden2 -> out
        """
        operators = [
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Nogradient', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
        ]
        desired_grad_operators = [
            CreateOperator(
                'DirectGradient',
                'out_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testMultiUseInputAndMultipleVersions(self):
        """Test gradient for the following case:

        in -> in
        in -> hidden1, hidden2
        hidden1, hidden2 -> out
        """
        operators = [
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Direct', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
        ]
        desired_grad_operators = [
            CreateOperator(
                'DirectGradient',
                'out_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden2_grad', 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'Sum',
                ['in_grad', '_in_grad_autosplit_0'], 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'in_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testMultiUseInputAutoGenSumDevice(self):
        parallel_tag = "parallelize:shard_by_1"
        split_op_device_option_clear_auto_gen_sum = core.DeviceOption(
            caffe2_pb2.CPU,
            extra_info=[
                parallel_tag,
                "{}:1".format(IR.ONLY_KEEP_IS_AUTO_GEN_SUM_OPS_TAG),
            ]
        )
        split_op_device_option_no_clear_auto_gen_sum = core.DeviceOption(
            caffe2_pb2.CPU,
            extra_info=[parallel_tag]
        )
        operators_clear_auto_gen_sum = [
            CreateOperator(
                'Direct', 'in', 'hidden1',
                device_option=split_op_device_option_clear_auto_gen_sum
            ),
            CreateOperator(
                'Direct', 'in', 'hidden2',
                device_option=split_op_device_option_clear_auto_gen_sum
            ),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
        ]
        gradients_clear_auto_gen_sum, _ = GradientRegistry.GetBackwardPass(
            operators_clear_auto_gen_sum, {'out': 'out_grad'})
        self.assertEqual(gradients_clear_auto_gen_sum[-1].type, "Sum")
        self.assertNotIn(
            parallel_tag,
            gradients_clear_auto_gen_sum[-1].device_option.extra_info
        )

        operators_no_clear_auto_gen_sum = [
            CreateOperator(
                'Direct', 'in', 'hidden1',
                device_option=split_op_device_option_no_clear_auto_gen_sum
            ),
            CreateOperator(
                'Direct', 'in', 'hidden2',
                device_option=split_op_device_option_no_clear_auto_gen_sum
            ),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
        ]
        gradients_no_clear_auto_gen_sum, _ = GradientRegistry.GetBackwardPass(
            operators_no_clear_auto_gen_sum, {'out': 'out_grad'})
        self.assertEqual(gradients_clear_auto_gen_sum[-1].type, "Sum")
        self.assertIn(
            parallel_tag,
            gradients_no_clear_auto_gen_sum[-1].device_option.extra_info
        )

    def testMultiUseInputAndMultipleVersionsBig(self):
        """Test gradient for the following case:

        in -> in
        in -> hidden1, hidden2
        hidden1, hidden2 -> in
        in -> hidden3, hidden4, hidden5
        hidden3, hidden4, hidden5 -> out
        """
        operators = [
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Direct', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'in'),
            CreateOperator('Direct', 'in', 'hidden3'),
            CreateOperator('Direct', 'in', 'hidden4'),
            CreateOperator('Direct', 'in', 'hidden5'),
            CreateOperator('Direct', ['hidden3', 'hidden4', 'hidden5'], 'out'),
        ]
        desired_grad_operators = [
            CreateOperator(
                'DirectGradient',
                'out_grad', ['hidden3_grad', 'hidden4_grad', 'hidden5_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden5_grad', 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden4_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden3_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator(
                'Sum',
                ['in_grad', '_in_grad_autosplit_0',
                 '_in_grad_autosplit_1'],
                'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'in_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden2_grad', 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'Sum',
                ['in_grad', '_in_grad_autosplit_0'],
                'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'in_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        for s in gradients:
            print(str(s))
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testGradientMappingUsingSumOp(self):
        """Since Sum is used in accumulating gradients, we will test if
        it is OK to also explicitly use it in the graph."""
        operators = [
            CreateOperator('FC', ['in', 'w', 'b'], 'fc'),
            CreateOperator('Sum', 'fc', 'agg'),
            CreateOperator('AveragedLoss', 'agg', 'loss'),
        ]
        # This should run correctly.
        gradient_ops, _ = GradientRegistry.GetBackwardPass(
            operators, {'loss': 'loss_grad'})
        for s in gradient_ops:
            print(str(s))

    def testGradientCalculationWithPrint(self):
        """Test a common use case where we have Print in the forward pass."""
        operators = [
            CreateOperator('FC', ['in', 'w', 'b'], 'fc'),
            CreateOperator('Print', 'fc', []),
            CreateOperator('AveragedLoss', 'fc', 'loss'),
        ]
        desired_grad_operators = [
            CreateOperator('AveragedLossGradient',
                           ['fc', 'loss_grad'], 'fc_grad'),
            CreateOperator('FCGradient', ['in', 'w', 'fc_grad'],
                           ['w_grad', 'b_grad', 'in_grad']),
        ]
        for g in desired_grad_operators:
            g.is_gradient_op = 1
        # This should run correctly.
        gradient_ops, _ = GradientRegistry.GetBackwardPass(
            operators, {'loss': 'loss_grad'})
        for s in gradient_ops:
            print(str(s))
        self.assertOperatorListEqual(gradient_ops, desired_grad_operators)

    def testStopGradient(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('StopGradient', 'hidden', 'hidden2'),
            CreateOperator('Direct', 'hidden2', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'out_grad', 'hidden2_grad'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)

    def testStopGradientOrphan(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('StopGradient', 'hidden', 'auto_blobx'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        with self.assertRaises(ValueError):
            # This should complain about incorrect use of StopGradient
            gradients, _ = GradientRegistry.GetBackwardPass(
                operators, {'out': 'out_grad'})

    def testStopGradientInplace(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('StopGradient', 'hidden', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'out_grad', 'hidden_grad'),
        ]
        gradients, grad_map = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)
        self.assertEqual(grad_map, {'out': 'out_grad'})

    def testStopGradientWithMultiUseOperators(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'hidden2'),
            CreateOperator('StopGradient', 'hidden', 'hidden3'),
            CreateOperator('Direct', ['hidden2', 'hidden3'], 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'out_grad',
                           ['hidden2_grad', 'hidden3_grad']),
            CreateOperator('DirectGradient', 'hidden2_grad', 'hidden_grad'),
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        gradients, grad_map = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertOperatorListEqual(gradients, desired_grad_operators)
        self.assertEqual(
            grad_map, {'out': 'out_grad', 'hidden2': 'hidden2_grad',
                       'hidden3': 'hidden3_grad', 'hidden': 'hidden_grad',
                       'in': 'in_grad'})

    def test_zero_gradient(self):
        net = core.Net("zero_grad_test")

        hidden_prev, cell, gates, seq_lengths, timestep =\
            net.AddExternalInput("h", "c", "g", "s", "t")
        hidden, cell = net.LSTMUnit(
            [hidden_prev, cell, gates, seq_lengths, timestep],
            ["hidden_t", "cell_t"])
        with self.assertRaises(Exception):
            net.AddGradientOperators([hidden])
        net.ZeroGradient(cell, [])
        net.AddGradientOperators([hidden])

    def test_two_grads(self):
        net = core.Net("test_two_grads")
        input, two, three = net.AddExternalInput("input", "two", "three")

        m1 = net.Mul([input, two], "mul_1")
        m2 = net.Mul([m1, three], "mul_2")
        grad_map = net.AddGradientOperators([m2, m1])
        workspace.ResetWorkspace()
        workspace.blobs[input] = np.array([1]).astype(np.float32)
        workspace.blobs[two] = np.array([2]).astype(np.float32)
        workspace.blobs[three] = np.array([3]).astype(np.float32)
        workspace.RunNetOnce(net)
        print(net.Proto())
        for blob in workspace.blobs:
            print(blob, workspace.blobs[blob])
        print("Input grad: ", workspace.blobs[grad_map[str(input)]])
        assert workspace.blobs[grad_map[str(input)]] == 8.0


# Skip if sparse operators are not available
@unittest.skipIf(not core.IsOperator('SparseFunHash'),
                 'Sparse operators not available')
class TestSparseGradientsAccumulation(test_util.TestCase):
    def testSparseAccumulationWithValues(self):
        # The gradient for "Gather" only computes values. indices are directly
        # passed from the input
        #
        # x1-->Gather-->x4-->
        #        |          |
        # x2-----+     DotProduct-->x6
        #        |          |
        # x3-->Gather-->x5-->
        net = core.Net("test_net")
        net.Gather(["x2", "x1"], "x4")
        net.Gather(["x2", "x3"], "x5")
        net.DotProduct(["x4", "x5"], "x6")
        net.AddGradientOperators(["x6"])
        sum_op_i = net.Proto().op[-2]
        sum_op_v = net.Proto().op[-1]
        self.assertEqual(sum_op_i.input[0], "x3")
        self.assertEqual(sum_op_i.input[1], "x1")
        self.assertEqual(sum_op_i.output[0], "x2_grad_indices_concat")
        self.assertEqual(sum_op_v.input[0], "x5_grad")
        self.assertEqual(sum_op_v.input[1], "x4_grad")
        self.assertEqual(sum_op_v.output[0], "x2_grad_values_concat")

    def testSparseGradientToDense(self):
        #
        #                                        x1-->Gather-->x4-->
        #                                                 |        |
        # x0, w, b-->FC-->x2-->EnsureDenseGradient-->x2---+  DotProduct-->x6
        #                                                 |        |
        #                                        x3-->Gather-->x5-->
        net = core.Net("test_net")
        net.FC(["x0", "w", "b"], "x2")
        net.EnsureDense(["x2"], "x2")
        net.Gather(["x2", "x1"], "x4")
        net.Gather(["x2", "x3"], "x5")
        net.DotProduct(["x4", "x5"], "x6")
        net.AddGradientOperators(["x6"])
        ensure_dense_op = net.Proto().op[-2]
        self.assertEqual(ensure_dense_op.input[0], "x2_grad_indices_concat")
        self.assertEqual(ensure_dense_op.input[1], "x2_grad_values_concat")
        self.assertEqual(ensure_dense_op.output[0], "x2_grad")

    def testSparseAccumulationWithIndicesAndValues(self):
        # The gradient for "SparseFunHash" computes both indices and values
        #
        # x1-------->
        #           |
        # x2---->   |
        #       |   |
        # x3---SparseFunHash-->x8
        #       /               \
        # x4---+            DotProduct-->x10
        #       \               /
        # x5---SparseFunHash-->x9
        #       |   |
        # x6---->   |
        #           |
        # x7-------->
        net = core.Net("test_net")
        net.SparseFunHash(["x1", "x2", "x3", "x4"], "x8")
        net.SparseFunHash(["x5", "x6", "x7", "x4"], "x9")
        net.DotProduct(["x8", "x9"], "x10")
        net.AddGradientOperators(["x10"])
        sum_op_i = net.Proto().op[-2]
        sum_op_v = net.Proto().op[-1]
        self.assertEqual(sum_op_i.input[0], "_x4_grad_indices_autosplit_0")
        self.assertEqual(sum_op_i.input[1], "_x4_grad_indices_autosplit_1")
        self.assertEqual(sum_op_i.output[0], "x4_grad_indices_concat")
        self.assertEqual(sum_op_v.input[0], "_x4_grad_values_autosplit_0")
        self.assertEqual(sum_op_v.input[1], "_x4_grad_values_autosplit_1")
        self.assertEqual(sum_op_v.output[0], "x4_grad_values_concat")


class TestGradientsAccumulationWithNoGradientOps(test_util.TestCase):
    def testNormalAccumulation(self):
        #  x1-->Relu--x2----------------->DotProduct-->x4
        #                |                 |
        #                 -->Softmax-->x3-->
        net = core.Net("test_net")
        net.Relu("x1", "x2")
        net.Softmax("x2", "x3")
        net.DotProduct(["x2", "x3"], "x4")
        net.AddGradientOperators(["x4"])
        sum_op = net.Proto().op[-2]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")

    def testAccumulationWithNoGradientBranch(self):
        #                 -->PRINT
        #                |
        #  x1-->Relu--x2----------------->DotProduct-->x4
        #                |                 |
        #                 -->Softmax-->x3-->
        net = core.Net("test_net")
        net.Relu("x1", "x2")
        net.Print("x2", [])
        net.Softmax("x2", "x3")
        net.DotProduct(["x2", "x3"], "x4")
        net.AddGradientOperators(["x4"])
        sum_op = net.Proto().op[-2]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")


class TestGradientsAccumulationWithPassThroughGradients(test_util.TestCase):
    def testAddOpInMiddle(self):
        #  x1-->Relu--x2----------------->Add-->x4
        #                |                 |
        #                 -->Softmax-->x3-->
        #
        # Expected gradient graph:
        #
        #  x1_g<--ReluG<--x2_g<--Sum<------------<---------x4_g
        #                          |                       |
        #                           <--_x2_g_split_0<--SoftmaxG
        net = core.Net("test_net")
        net.Relu("x1", "x2")
        net.Softmax("x2", "x3")
        net.Add(["x2", "x3"], "x4")
        input_to_grad = net.AddGradientOperators({"x4": "x4_grad"})
        sum_op = net.Proto().op[-2]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")

    def testAddAndDynamicConstant(self):
        net = core.Net("test_net")
        net.FC(["x1", "x1_w", "x1_b"], ["x2"])
        net.Relu("x2", "x2")
        net.ConstantFill(["x2"], ["x3"])
        net.Add(["x2", "x3"], "x4")
        net.FC(["x4", "x4_w", "x4_b"], ["x5"])
        net.SoftmaxWithLoss(["x5", "labels"], ["softmax", "loss"])
        input_to_grad = net.AddGradientOperators(["loss"])
        for op in net.Proto().op:
            self.assertFalse(op.type == 'Sum')

        self.assertTrue("x4" in input_to_grad)
        self.assertTrue("x1" in input_to_grad)
        self.assertEqual(input_to_grad["x1"], "x1_grad")

    def testAddAndStaticConstant(self):
        net = core.Net("test_net")
        net.FC(["x1", "x1_w", "x1_b"], ["x2"])
        net.Relu("x2", "x2")
        net.ConstantFill([], ["x3"], shape=[1])
        net.Add(["x2", "x3"], "x4", broadcast=1)
        net.FC(["x4", "x4_w", "x4_b"], ["x5"])
        net.SoftmaxWithLoss(["x5", "labels"], ["softmax", "loss"])
        input_to_grad = net.AddGradientOperators(["loss"])
        print(input_to_grad)

        self.assertTrue("x1" in input_to_grad)
        self.assertEqual(input_to_grad["x1"], "x1_grad")

    def testSubOpInMiddle(self):
        #  x1-->Relu--x2----------------->Sub-->x4
        #                |                 |
        #                 -->Softmax-->x3-->
        #
        # Expected gradient graph:
        #
        #  x1_g<--ReluG<--x2_g<--Sum<------------<-----------------------x4_g
        #                          |                                      |
        #                           <--_x2_g_split_0<--SoftmaxG<--x3_g<--neg
        net = core.Net("test_net")
        net.Relu("x1", "x2")
        net.Softmax("x2", "x3")
        net.Sub(["x2", "x3"], "x4")
        input_to_grad = net.AddGradientOperators({"x4": "x4_grad"})
        print(str(net.Proto()))
        sum_op = net.Proto().op[-2]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")

    def testAddOpAtLeaf(self):
        # x1
        #   \
        #    -->Add-->x4
        #   /           \
        # x2             -->DotProduct-->x6
        #   \           /
        #    -->Add-->x5
        #   /
        # x3
        #
        # Expected gradient graph:
        #
        #  x2_g<--Sum<--x4_g<--DotProductG<--x6_g
        #          |                |                       |
        #           <---x5_g<-------
        net = core.Net("test_net")
        net.Add(["x1", "x2"], "x4")
        net.Add(["x2", "x3"], "x5")
        net.DotProduct(["x4", "x5"], "x6")
        input_to_grad = net.AddGradientOperators({"x6": "x6_grad"})
        sum_op = net.Proto().op[-1]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")
        self.assertEqual(input_to_grad["x2"], "x2_grad")
        self.assertEqual(input_to_grad["x3"], "x3_grad")

    def testSubOpAtLeaf(self):
        # x1
        #   \
        #    -->Sub-->x4
        #   /           \
        # x2             -->DotProduct-->x6
        #   \           /
        #    -->Sub-->x5
        #   /
        # x3
        #
        # Expected gradient graph:
        #
        #  x2_g<-------Sum<--x2_g_split_0<--neg<--x4_g<--DotProductG<--x6_g
        #               |                                       |
        #  x3_g<--neg<--<--x5_g<--------------------------------
        net = core.Net("test_net")
        net.Sub(["x1", "x2"], "x4")
        net.Sub(["x2", "x3"], "x5")
        net.DotProduct(["x4", "x5"], "x6")
        input_to_grad = net.AddGradientOperators({"x6": "x6_grad"})
        sum_op = net.Proto().op[-1]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")
        self.assertEqual(input_to_grad["x2"], "x2_grad")
        self.assertEqual(input_to_grad["x3"], "x3_grad")

    def testMultiLayerAddOps(self):
        # x1
        #   \
        #    -->Add-->x4
        #   /           \
        # x2             -->Add-->x6
        #   \           /
        #    -->Add-->x5
        #   /
        # x3
        #
        # Expected gradient graph:
        #
        #  x2_g<--Sum<-----x6_g
        #          |         |
        #           <--------
        net = core.Net("test_net")
        net.Add(["x1", "x2"], "x4")
        net.Add(["x2", "x3"], "x5")
        net.Add(["x4", "x5"], "x6")
        input_to_grad = net.AddGradientOperators({"x6": "x6_grad"})
        sum_op = net.Proto().op[-1]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")
        self.assertEqual(input_to_grad["x2"], "x2_grad")
        self.assertEqual(input_to_grad["x3"], "x3_grad")

    def testMultiLayerSubOps(self):
        # x1
        #   \
        #    -->Sub-->x4
        #   /           \
        # x2             -->Sub-->x6
        #   \           /
        #    -->Sub-->x5
        #   /
        # x3
        #
        # Expected gradient graph:
        #
        #  x2_g<--Sum<-----x6_g
        #          |         |
        #           <--------
        net = core.Net("test_net")
        net.Sub(["x1", "x2"], "x4")
        net.Sub(["x2", "x3"], "x5")
        net.Sub(["x4", "x5"], "x6")
        input_to_grad = net.AddGradientOperators({"x6": "x6_grad"})
        sum_op = net.Proto().op[-1]
        self.assertEqual(sum_op.input[0], "x2_grad")
        self.assertEqual(sum_op.input[1], "_x2_grad_autosplit_0")
        self.assertEqual(sum_op.output[0], "x2_grad")
        self.assertEqual(input_to_grad["x1"], "x1_grad")
        self.assertEqual(input_to_grad["x2"], "x2_grad")
        self.assertEqual(input_to_grad["x3"], "x3_grad")

    def testAccumulationRuns(self):
        net = core.Net("test_net")
        input, one, two, three = net.AddExternalInput(
            "input", "one", "two", "three")

        m1 = net.Mul([input, two], "mul_1")
        m2 = net.Mul([input, three], "mul_2")
        sub = net.Sub([m1, one])
        grad_map = net.AddGradientOperators([m2, sub])

        workspace.ResetWorkspace()
        workspace.blobs[one] = np.array([1]).astype(np.float32)
        workspace.blobs[input] = np.array([1]).astype(np.float32)
        workspace.blobs[two] = np.array([2]).astype(np.float32)
        workspace.blobs[three] = np.array([3]).astype(np.float32)
        workspace.RunNetOnce(net)
        print("Input grad: ", workspace.blobs[grad_map[str(input)]])
        assert workspace.blobs[grad_map[str(input)]] == 5.0

    def testIncorrectOperator(self):
        net = core.Net("test_net")
        a, b, one = net.AddExternalInput("a", "b", "one")
        m1 = net.Mul(a, b)  # does not have second output
        sub = net.Sub([m1, one])
        try:
            net.AddGradientOperators([sub])
            self.assertFalse(True, "Did not throw exception")
        except Exception as e:
            self.assertTrue("schema" in str(e))

    def testDeviceOptionsPropagation(self):
        '''
        Test verifies that aggregation operators in a backward path will be in
        the same device as the parameter.
        '''
        device_0 = 'node:0'

        # init_net.
        init_net = core.Net("init_net")
        with core.DeviceScope(0, node_name=device_0):
            w = init_net.UniformFill([], 'w', shape=[10000, 64])
            ids = init_net.GivenTensorFill(
                [],
                'ids',
                values=np.random.random_integers(low=0, high=10000, size=10),
            )
            ids_2 = init_net.GivenTensorFill(
                [],
                'ids_2',
                values=np.random.random_integers(low=0, high=10000, size=10),
            )

        # train_net.
        train_net = core.Net("train_net")
        with core.DeviceScope(0, node_name=device_0):
            vals = train_net.Gather([w, ids], "gathered")
            r_vals = train_net.ReduceSum([vals], 1, axes=0)

            vals_2 = train_net.Gather([w, ids_2], "gathered_2")
            r_vals_2 = train_net.ReduceSum([vals_2], 1, axes=0)

        loss = train_net.Sum([r_vals, r_vals_2], 1)
        train_net.AddGradientOperators([loss])
        # All concat operators should be on device_0
        for op in train_net.Proto().op:
            if op.type == 'Concat':
                self.assertEqual(op.device_option.node_name, device_0)


if __name__ == '__main__':
    unittest.main()
