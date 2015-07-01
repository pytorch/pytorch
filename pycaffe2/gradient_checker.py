import numpy as np
from pycaffe2 import core, workspace
from caffe2.proto import caffe2_pb2

class GradientChecker:
  """A gradient checker in Python.

  This is not the most efficient way to check gradients, as the Python interface
  will involve a lot of copy back and forth operations. Use at your own risk.
  """
  def __init__(self, stepsize, threshold,
               device_option=caffe2_pb2.DeviceOption(),
               workspace_name="gradient_check"):
    self._stepsize = stepsize
    self._threshold = threshold
    self._device_option = device_option
    self._workspace_name = workspace_name

  def GetLossAndGrad(self, op, grad_ops, x, input_name, outputs_with_grads):
    # First, feed in the current input. Note that we are not changing anything
    # else, so we don't need to feed in others.
    workspace.FeedBlob(input_name, x, self._device_option)
    # Run.
    workspace.RunOperatorOnce(op)
    loss = 0.
    # Get Loss and feed in the gradients, run gradient ops.
    for idx in outputs_with_grads:
      name = op.output[idx]
      arr = workspace.FetchBlob(name)
      loss += (arr ** 2).sum()
      workspace.FeedBlob(core.GetGradientName(name), arr, self._device_option)
    loss /= 2.
    # Run gradient ops
    workspace.RunOperatorsOnce(grad_ops)
    # Get gradients
    grad = workspace.FetchBlob(core.GetGradientName(input_name))
    return loss, grad


  def CheckSimple(self, op, inputs, input_to_check,
                  outputs_with_grads, grad_ops=None):
    """Checks the operator in a very simple fashion by stacking a sum of squares
    on the top.

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
    Outputs:
      boolean: True if it passes, False if it does not pass.
    """
    # Entering the checker workspace
    old_ws_name = workspace.CurrentWorkspace()
    if self._workspace_name != old_ws_name:
      workspace.SwitchWorkspace(self._workspace_name, True)

    op.device_option.CopyFrom(self._device_option)
    if grad_ops is None:
      grad_ops = core.GradientRegistry.GetGradient(op)

    dims_to_check = inputs[input_to_check].size
    # First, feed in the input.
    for i, arr in enumerate(inputs):
      workspace.FeedBlob(op.input[i], arr, self._device_option)

    # Get the loss and gradient for the original.
    input_name = op.input[input_to_check]
    loss, grad = self.GetLossAndGrad(op, grad_ops, inputs[input_to_check],
                                     input_name, outputs_with_grads)
    grad_estimate = np.zeros_like(inputs[input_to_check])
    for current_dim in range(dims_to_check):
      # Positive gradient
      inputs[input_to_check].flat[current_dim] += self._stepsize
      pos_loss, _ = self.GetLossAndGrad(op, grad_ops, inputs[input_to_check],
                                     input_name, outputs_with_grads)
      # Negative gradient
      inputs[input_to_check].flat[current_dim] -= self._stepsize * 2
      neg_loss, _ = self.GetLossAndGrad(op, grad_ops, inputs[input_to_check],
                                     input_name, outputs_with_grads)
      # Recover the value
      inputs[input_to_check].flat[current_dim] += self._stepsize
      grad_estimate.flat[current_dim] = (pos_loss - neg_loss) / self._stepsize / 2
    # Now, check correctness
    scale = np.maximum(np.maximum(np.abs(grad), np.abs(grad_estimate)), 1)
    fail_mat = (np.abs(grad - grad_estimate) > scale * self._threshold)
    if np.any(fail_mat):
      idx = np.flatnonzero(fail_mat)
      #print 'Failed. [idx, grad, grad_estimate] are:'
      #print np.vstack([idx, grad.flat[idx], grad_estimate.flat[idx]]).T
      ret = False
    else:
      ret = True
    # After finishing, cleaning up things.
    if self._workspace_name != old_ws_name:
      # We reset the workspace to make sure everything intermediate is cleaned
      # up. Note that there is no need to delete a workspace - when empty it
      # takes a very limited amount of memory.
      workspace.ResetWorkspace()
      workspace.SwitchWorkspace(old_ws_name)
    return ret, grad, grad_estimate