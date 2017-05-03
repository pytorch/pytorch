## @package gradient_checker
# Module caffe2.python.gradient_checker
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2


class NetGradientChecker(object):
    @staticmethod
    def Check(net, outputs_with_grad, input_values,
              input_to_check, step_size=0.0001,
              threshold=0.05, print_net=True):
        assert input_to_check in input_values.keys()

        net_copy = net.Clone(net.Name() + "_copy")

        grad_map = net_copy.AddGradientOperators(outputs_with_grad)
        assert input_to_check in grad_map, (
            '{} has no gradient, cannot check net gradient.'.format(
                input_to_check))

        for name, value in input_values.items():
            workspace.blobs[name] = value

        def GetLoss(new_value):
            workspace.blobs[input_to_check] = new_value
            workspace.RunNetOnce(net_copy)
            return sum([
                workspace.blobs[output]
                for output in outputs_with_grad
            ]).sum()

        def GetValue(dim, delta):
            input_value = input_values[input_to_check].copy()
            input_value.flat[dim] += delta
            return input_value

        workspace.RunNetOnce(net_copy)
        grad_blob = grad_map[input_to_check]

        def get_analytic_grad(grad_blob):
            if isinstance(grad_blob, core.BlobReference):
                return workspace.blobs[grad_blob]

            # If grad_blob is not a single blob, it should be a gradient slice.
            # To make it comparable with the estimiated gradient which is dense,
            # we need to first convert grad_blob to dense gradient.
            assert isinstance(grad_blob, core.GradientSlice)
            dense_grad = 'tmp_dense_grad'
            sparse_to_dense_op = core.CreateOperator(
                'SparseToDense',
                [grad_blob.indices, grad_blob.values, input_to_check],
                dense_grad,
            )
            workspace.RunOperatorOnce(sparse_to_dense_op)
            return workspace.blobs[dense_grad]

        analytic_grad = get_analytic_grad(grad_blob)

        grad_estimate = np.zeros_like(input_values[input_to_check])
        for dim in range(input_values[input_to_check].size):
            pos_loss = GetLoss(GetValue(dim, step_size))
            neg_loss = GetLoss(GetValue(dim, -step_size))
            grad_estimate.flat[dim] = (pos_loss - neg_loss) / step_size / 2

        err_msg = "Error in gradient check for net_copy {}".format(
            net.Name())
        if print_net:
            err_msg += ": {}".format(net.Proto())

        np.testing.assert_allclose(
            analytic_grad, grad_estimate,
            atol=threshold, rtol=threshold,
            err_msg=err_msg,
        )

        delta = np.abs(grad_estimate - analytic_grad).flatten()
        return np.mean(delta), max(delta)


class GradientChecker:
    """A gradient checker in Python.

    This is not the most efficient way to check gradients, as the Python
    interface will involve a lot of copy back and forth operations. Use at your
    own risk.
    """

    def __init__(
        self,
        stepsize,
        threshold,
        device_option=caffe2_pb2.DeviceOption(),
        workspace_name="gradient_check"
    ):
        self._stepsize = stepsize
        self._threshold = threshold
        self._device_option = device_option
        self._workspace_name = workspace_name

    def GetLossAndGrad(
        self, op, grad_ops, x, input_name, grad_name, outputs_with_grads
    ):
        # First, feed in the current input. Note that we are not changing
        # anything else, so we don't need to feed in others.
        workspace.FeedBlob(input_name, x, self._device_option)
        # Run.
        workspace.RunOperatorOnce(op)
        loss = 0.
        # Get Loss and feed in the gradients, run gradient ops.
        for idx in outputs_with_grads:
            name = op.output[idx]
            arr = workspace.FetchBlob(name)
            loss += (arr**2).sum()
            workspace.FeedBlob(name + '_grad', arr, self._device_option)
        loss /= 2.
        # Run gradient ops
        workspace.RunOperatorsOnce(grad_ops)
        # Get gradients
        if isinstance(grad_name, core.GradientSlice):
            workspace.FeedBlob('zeros', np.zeros_like(x, dtype=np.float32))
            workspace.FeedBlob('ones', np.ones(1, dtype=np.float32))
            gv_cpu_op = core.CreateOperator(
                'EnsureCPUOutput', grad_name.values, grad_name.values + '_cpu',
                device_option=self._device_option
            )
            gi_cpu_op = core.CreateOperator(
                'EnsureCPUOutput', grad_name.indices, grad_name.indices + '_cpu',
                device_option=self._device_option
            )
            sparse_to_dense_op = core.CreateOperator(
                'ScatterWeightedSum',
                [
                    'zeros', 'ones', grad_name.indices + '_cpu',
                    grad_name.values + '_cpu', 'ones'
                ],
                'zeros',
            )
            workspace.RunOperatorOnce(gv_cpu_op)
            workspace.RunOperatorOnce(gi_cpu_op)
            workspace.RunOperatorOnce(sparse_to_dense_op)
            grad = workspace.FetchBlob('zeros')
        else:
            grad = workspace.FetchBlob(grad_name)
        return loss, grad

    def CheckSimple(
        self,
        op,
        inputs,
        input_to_check,
        outputs_with_grads,
        grad_ops=None,
        input_device_options=None
    ):
        """Checks the operator in a very simple fashion by stacking a sum of
        squares on the top.

        Inputs:
          op: the operator to be checked.
          inputs: the input data in numpy arrays.
          input_to_check: an index specifying which input blob we should
              check.
          outputs_with_grads: indices specifying which output blobs will we
              need to check gradients with. For these outputs, we will collect a
              squared sum and also feed in their gradients.
          grad_operator: the gradient operator. If not given, we will get the
              gradient operator from the gradient registry.
          input_device_options: an optional mapping from input names to
              DeviceOptions (to override the default DeviceOption)
        Outputs:
          boolean: True if it passes, False if it does not pass.
        """
        if input_device_options is None:
            input_device_options = {}
        # Entering the checker workspace
        old_ws_name = workspace.CurrentWorkspace()
        if self._workspace_name != old_ws_name:
            workspace.SwitchWorkspace(self._workspace_name, True)

        op.device_option.CopyFrom(self._device_option)
        if grad_ops is None:
            # TODO(jiayq): use the gradient registration instead of the old
            # hack.
            grad_ops, g_input = core.GradientRegistry.GetGradientForOp(
                op, [s + '_grad' for s in op.output])

        dims_to_check = inputs[input_to_check].size
        # First, feed in the input.
        for i, arr in enumerate(inputs):
            workspace.FeedBlob(
                op.input[i], arr,
                input_device_options.get(
                    op.input[i], self._device_option))

        # Get the loss and gradient for the original.
        input_name = op.input[input_to_check]
        grad_name = g_input[input_to_check]
        loss, grad = self.GetLossAndGrad(
            op, grad_ops, inputs[input_to_check], input_name, grad_name,
            outputs_with_grads
        )
        grad_estimate = np.zeros_like(inputs[input_to_check])
        if grad_estimate.shape != grad.shape:
            raise Exception(
                "Mismatched gradient shapes: estimated ({}), grad ({})".format(
                    grad_estimate.shape, grad.shape))

        for current_dim in range(dims_to_check):
            # Positive gradient
            inputs[input_to_check].flat[current_dim] += self._stepsize
            pos_loss, _ = self.GetLossAndGrad(
                op, grad_ops, inputs[input_to_check], input_name,
                grad_name, outputs_with_grads
            )
            # Negative gradient
            inputs[input_to_check].flat[current_dim] -= self._stepsize * 2
            neg_loss, _ = self.GetLossAndGrad(
                op, grad_ops, inputs[input_to_check], input_name,
                grad_name, outputs_with_grads
            )
            # Recover the value
            inputs[input_to_check].flat[current_dim] += self._stepsize
            grad_estimate.flat[current_dim] = (
                pos_loss - neg_loss) / self._stepsize / 2
        # Now, check correctness
        fail_mat = ~np.isclose(
            grad, grad_estimate, atol=self._threshold, rtol=self._threshold)
        if np.any(fail_mat):
            idx = np.flatnonzero(fail_mat)
            print('Failed. [idx, grad, grad_estimate] are:')
            print(np.vstack([idx, grad.flat[idx], grad_estimate.flat[idx]]).T)
            ret = False
        else:
            ret = True
        # After finishing, cleaning up things.
        if self._workspace_name != old_ws_name:
            # We reset the workspace to make sure everything intermediate is
            # cleaned up. Note that there is no need to delete a workspace -
            # when empty it takes a very limited amount of memory.
            workspace.ResetWorkspace()
            workspace.SwitchWorkspace(old_ws_name)
        return ret, grad, grad_estimate
