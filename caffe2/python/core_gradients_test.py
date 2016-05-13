# This a large test that goes through the translation of the bvlc caffenet
# model, runs an example through the whole model, and verifies numerically
# that all the results look right. In default, it is disabled unless you
# explicitly want to run it.

import unittest

import caffe2.python
from caffe2.python.core import *
from caffe2.python import test_util

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



# First gradient: (in -> out) leading to (out_grad -> in_grad)
@GradientRegistry.RegisterGradient('Direct')
def AddDirectGradient(op, g_output):
    return (
        CreateOperator('DirectGradient', NeedAll(op, g_output), GIS(op)),
        GIS(op)
    )


# Second gradient: (in -> out) leading to (out, out_grad -> in_grad)
@GradientRegistry.RegisterGradient('UseOutput')
def AddUseOutputGradient(op, g_output):
    return (
        CreateOperator('UseOutputGradient',
            list(op.output) + NeedAll(op, g_output), GIS(op)),
        GIS(op)
    )


@GradientRegistry.RegisterGradient('UseInput')
def AddUseInputGradient(op, g_output):
    return (
        CreateOperator('UseInputGradient',
            list(op.input) + NeedAll(op, g_output), GIS(op)),
        GIS(op)
    )


@GradientRegistry.RegisterGradient('Sink')
def AddSinkGradient(op, g_output):
    return (
        CreateOperator('SinkGradient', [], GIS(op)),
        GIS(op)
    )


class TestGradientCalculation(test_util.TestCase):
    def testDirect(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('DirectGradient', 'out_grad', 'hidden_grad'),
            CreateOperator('DirectGradient', 'hidden_grad', 'in_grad'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testDirectButNoOutputGradient(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('Direct', 'hidden', 'out'),
        ]
        try:
            gradients, _ = GradientRegistry.GetBackwardPass(operators)
        except RuntimeError as e:
            self.assertEqual(
                str(e),
                'Need gradient for "out" but it is not provided.'
            )
        else:
            self.assertTrue(
                False, "Failed to identify an impossible gradient case."
            )

    def testDirectInPlace(self):
        operators = [
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Direct', 'in', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('DirectGradient', 'out_grad', 'in_grad'),
            CreateOperator('DirectGradient', 'in_grad', 'in_grad'),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testUseOutput(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'hidden'),
            CreateOperator('UseOutput', 'hidden', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('UseOutputGradient',
                ['out', 'out_grad'], 'hidden_grad'
            ),
            CreateOperator('UseOutputGradient',
                ['hidden', 'hidden_grad'], 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testUseOutputInPlace(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'in'),
            CreateOperator('UseOutput', 'in', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('UseOutputGradient',
                ['out', 'out_grad'], 'in_grad'
            ),
            CreateOperator('UseOutputGradient',
                ['in', 'in_grad'], 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testUseOutputButOutputHasBeenChanged(self):
        operators = [
            CreateOperator('UseOutput', 'in', 'hidden'),
            # Note here: we overwrite hidden, but hidden will be needed by the
            # gradient calculation of the first operator, so the gradient registry
            # should return an error.
            CreateOperator('Direct', 'hidden', 'hidden'),
            CreateOperator('UseOutput', 'hidden', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        try:
            gradients, _ = GradientRegistry.GetBackwardPass(operators)
        except RuntimeError as e:
            self.assertEqual(
                str(e),
                'Gradient operator needs output "hidden" at version 0, but currently '
                'we have version 1.'
            )
        else:
            self.assertTrue(
                False, "Failed to identify an impossible gradient case."
            )

    def testUseInput(self):
        operators = [
            CreateOperator('Direct', 'in', 'hidden'),
            CreateOperator('UseInput', 'hidden', 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('UseInputGradient',
                ['hidden', 'out_grad'], 'hidden_grad'
            ),
            CreateOperator('DirectGradient',
                'hidden_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testUseInputButInputHasBeenChanged(self):
        """Test gradient for the following case:

        in -> hidden, with UseInput
        in -> in
        out -> (sink)
        in -> (sink)

    Since we overwrite in in op#1, but in will be needed by the gradient
    calculation of op#0, the gradient registry should raise an error.
    """
        operators = [
            CreateOperator('UseInput', 'in', 'hidden'),
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Sink', 'out', []),
            CreateOperator('Sink', 'in', []),
            CreateOperator('Sink', 'hidden', []),
        ]
        try:
            gradients, _ = GradientRegistry.GetBackwardPass(operators)
        except RuntimeError as e:
            self.assertEqual(
                str(e),
                'Gradient operator needs input "in" at version 0, but currently we '
                'have version 1.'
            )
        else:
            self.assertTrue(
                False, "Failed to identify an impossible gradient case."
            )

    def testMultiUseInput(self):
        """Test gradient for the following case:

        in -> hidden1
        in -> hidden2
        hidden1, hidden2 -> out
        out -> (sink)
    """
        operators = [
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Direct', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('DirectGradient',
                'out_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator('DirectGradient',
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator('DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator('Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'], 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testMultiUseInputAndMultipleVersions(self):
        """Test gradient for the following case:

        in -> in
        in -> hidden1, hidden2
        hidden1, hidden2 -> out
        out -> (sink)
    """
        operators = [
            CreateOperator('Direct', 'in', 'in'),
            CreateOperator('Direct', 'in', 'hidden1'),
            CreateOperator('Direct', 'in', 'hidden2'),
            CreateOperator('Direct', ['hidden1', 'hidden2'], 'out'),
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('DirectGradient',
                'out_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator('DirectGradient',
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator('DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator('Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'], 'in_grad'
            ),
            CreateOperator('DirectGradient',
                'in_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        self.assertTrue(gradients == desired_grad_operators)

    def testMultiUseInputAndMultipleVersionsBig(self):
        """Test gradient for the following case:

        in -> in
        in -> hidden1, hidden2
        hidden1, hidden2 -> in
        in -> hidden3, hidden4, hidden5
        hidden3, hidden4, hidden5 -> out
        out -> (sink)
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
            CreateOperator('Sink', 'out', []),
        ]
        desired_grad_operators = [
            CreateOperator('SinkGradient', [], 'out_grad'),
            CreateOperator('DirectGradient',
                'out_grad', ['hidden3_grad', 'hidden4_grad', 'hidden5_grad']
            ),
            CreateOperator('DirectGradient',
                'hidden5_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator('DirectGradient',
                'hidden4_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator('DirectGradient',
                'hidden3_grad', '_in_grad_autosplit_2'
            ),
            CreateOperator('Sum',
                [
                    '_in_grad_autosplit_0', '_in_grad_autosplit_1',
                    '_in_grad_autosplit_2'
                ], 'in_grad'
            ),
            CreateOperator('DirectGradient',
                'in_grad', ['hidden1_grad', 'hidden2_grad']
            ),
            CreateOperator('DirectGradient',
                'hidden2_grad', '_in_grad_autosplit_0'
            ),
            CreateOperator('DirectGradient',
                'hidden1_grad', '_in_grad_autosplit_1'
            ),
            CreateOperator('Sum',
                ['_in_grad_autosplit_0', '_in_grad_autosplit_1'], 'in_grad'
            ),
            CreateOperator('DirectGradient',
                'in_grad', 'in_grad'
            ),
        ]
        gradients, _ = GradientRegistry.GetBackwardPass(operators)
        for s in gradients:
            print(str(s))
        self.assertTrue(gradients == desired_grad_operators)


if __name__ == '__main__':
    unittest.main()
