import numpy as np
from pycaffe2 import core, workspace

class DeviceChecker(object):
  """A gradient checker in Python.

  This is not the most efficient way to check gradients, as the Python interface
  will involve a lot of copy back and forth operations. Use at your own risk.
  """
  def __init__(self, threshold, device_options):
    self._threshold = threshold
    self._device_options = device_options

  def CheckSimple(self, op, inputs, outputs_to_check):
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
    results = []
    workspace.SwitchWorkspace("_device_check_", True)
    for i, device_option in enumerate(self._device_options):
      for i, arr in enumerate(inputs):
        workspace.FeedBlob(op.inputs[i], arr, device_option)
      op.device_option.CopyFrom(device_option)
      workspace.RunOperatorOnce(op)
      results.append(
          [workspace.FetchBlob(op.outputs[idx]) for idx in outputs_to_check])
      # Everything is done, reset the workspace.
      workspace.ResetWorkspace()
    # After running on all devices, check correctness
    success = True
    for i in range(1, len(self._device_options)):
      for j in range(len(outputs_to_check)):
        x = results[i][j]
        y = results[0][j]
        if np.any(np.abs(x - y) > self._threshold):
          print 'Failure in checking device option', i, 'and output ',
          print op.outputs[j], '. The outputs are:'
          print x.flatten()
          print y.flatten()
          success = False
          continue
    workspace.SwitchWorkspace(old_ws_name)
    return success

  def CheckNet(self, net, inputs={}, ignore=set()):
    """Checks a network by inspecting all of its intermediate results, and see
    if things match.
    """
    old_ws_name = workspace.CurrentWorkspace()
    results = []
    blobs_to_check = sum([list(op.outputs) for op in net.operators], [])
    blobs_to_check = [b for b in blobs_to_check if b not in ignore]
    workspace.SwitchWorkspace("_device_check_", True)
    for i, device_option in enumerate(self._device_options):
      for name, arr in inputs.iteritems():
        workspace.FeedBlob(name, arr, device_option)
      for op in net.operators:
        op.device_option.CopyFrom(device_option)
      workspace.RunNetOnce(net)
      results.append(
          [workspace.FetchBlob(name) for name in blobs_to_check])
    # After running on all devices, check correctness
    success = True
    for i in range(1, len(results)):
      for j in range(len(blobs_to_check)):
        x = results[i][j]
        y = results[0][j]
        if np.any(np.abs(x - y) > self._threshold):
          print 'Failure in checking device option', i, 'and blob ',
          print blobs_to_check[j], '. The outputs are:'
          print x.flatten()
          print y.flatten()
          success = False
          continue
    workspace.SwitchWorkspace(old_ws_name)
    return success
