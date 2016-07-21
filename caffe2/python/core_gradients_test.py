from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util
from caffe2.python.core import CreateOperator, GradientRegistry


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
    @given(device_option=st.sampled_from([
        None,
        core.DeviceOption(caffe2_pb2.CUDA, 1)]))
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
        self.assertEqual(gradients, desired_grad_operators)

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
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, ['out'])
        self.assertEqual(gradients, desired_grad_operators)

    def testDoesNotGenerateUnnecessaryGradients(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        desired_grad_operators = [
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'hidden': 'hidden_grad'})
        self.assertEqual(gradients, desired_grad_operators)

    def testDirectButNoOutputGradientGiven(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {})
        self.assertEqual(gradients, [])

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
        self.assertEqual(gradients, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)

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
        core.DeviceOption(caffe2_pb2.CUDA, 1)]))
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
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator(
                'Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'], 'in_grad'
            ),
        ]
        if device_option:
            for op in desired_grad_operators:
                op.device_option.CopyFrom(device_option)
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {"out": "out_grad"})
        self.assertEqual(gradients, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)

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
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator(
                'Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'], 'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'in_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(
            operators, {'out': 'out_grad'})
        self.assertEqual(gradients, desired_grad_operators)

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
                'hidden5_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden4_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden3_grad', '_in_grad_autosplit_2'
            ),
            CreateOperator(
                'Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1',
                 '_in_grad_autosplit_2'],
                'in_grad'
            ),
            CreateOperator(
                'DirectGradient',
                'in_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator(
                'DirectGradient',
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator(
                'DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator(
                'Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'],
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
        self.assertEqual(gradients, desired_grad_operators)

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
        # This should run correctly.
        gradient_ops, _ = GradientRegistry.GetBackwardPass(
            operators, {'loss': 'loss_grad'})
        for s in gradient_ops:
            print(str(s))
        self.assertEqual(gradient_ops, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)

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
        self.assertEqual(gradients, desired_grad_operators)
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
        self.assertEqual(gradients, desired_grad_operators)
        self.assertEqual(
            grad_map, {'out': 'out_grad', 'hidden2': 'hidden2_grad',
                       'hidden3': 'hidden3_grad', 'hidden': 'hidden_grad',
                       'in': 'in_grad'})


if __name__ == '__main__':
    unittest.main()
