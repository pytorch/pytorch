## @package device_checker
# Module caffe2.python.device_checker
import numpy as np
import copy
from caffe2.python import workspace
from future.utils import viewitems


class DeviceChecker(object):
    """A device checker in Python to check consistency across multiple devices.

    This is not the most efficient way to check devices, as the Python interface
    will involve a lot of copies back and forth operations. Use at your own risk.
    """

    def __init__(self, threshold, device_options):
        self._threshold = threshold
        self._device_options = device_options

    def CheckSimple(self, op, inputs, outputs_to_check,
                    input_device_options=None):
        """Checks the operator with different device implementations.

        Inputs:
          op: the operator to be checked.
          inputs: the input data in numpy arrays.
          outputs_to_check: the outputs to check between devices.
          input_device_options: a mapping from input name to a device to use
            (instead of self._device_options)
        Outputs:
          boolean: True if it passes, False if it does not pass.
        """
        op = copy.deepcopy(op)
        input_device_options = input_device_options or {}
        # Entering the checker workspace
        old_ws_name = workspace.CurrentWorkspace()
        results = []
        workspace.SwitchWorkspace("_device_check_", True)
        for i, device_option in enumerate(self._device_options):
            for i, arr in enumerate(inputs):
                workspace.FeedBlob(
                    op.input[i], np.array(arr),
                    input_device_options.get(op.input[i], device_option))
            op.device_option.CopyFrom(device_option)
            workspace.RunOperatorOnce(op)
            results.append(
                [workspace.FetchBlob(op.output[idx])
                 for idx in outputs_to_check])
            # Everything is done, reset the workspace.
            workspace.ResetWorkspace()
        # After running on all devices, check correctness
        success = True
        for i in range(1, len(self._device_options)):
            for j in range(len(outputs_to_check)):
                x = results[i][j]
                y = results[0][j]
                if not np.allclose(x, y,
                                   atol=self._threshold, rtol=self._threshold):
                    print('Failure in checking device option {}'
                          ' and output {}. The outputs are:'
                          .format(i, op.output[outputs_to_check[j]]))
                    print(x.flatten())
                    print(y.flatten())
                    print(np.max(np.abs(x - y)))
                    success = False
                # else:
                #     print ('Passed device pair (0, %d), %s %s' %
                #            (i, outputs_to_check[j], y.shape))
        workspace.SwitchWorkspace(old_ws_name)
        return success

    def CheckNet(self, net, inputs=None, blobs_to_check=None, ignore=None):
        """Checks a network by inspecting all of its intermediate results, and
        see if things match.
        """
        if inputs is None:
            inputs = {}
        if ignore is None:
            ignore = set()
        old_ws_name = workspace.CurrentWorkspace()
        results = []
        if blobs_to_check is None:
            blobs_to_check = sum([list(op.output) for op in net.op], [])
        blobs_to_check = [b for b in blobs_to_check if b not in ignore]
        workspace.SwitchWorkspace("_device_check_", True)
        for device_option in self._device_options:
            for name, arr in viewitems(inputs):
                # print 'feeding', name
                workspace.FeedBlob(name, arr, device_option)
            for op in net.op:
                op.device_option.CopyFrom(device_option)
            workspace.RunNetOnce(net)
            results.append(
                [workspace.FetchBlob(name) for name in blobs_to_check]
            )
        # After running on all devices, check correctness
        success = True
        for i in range(1, len(results)):
            for j in range(len(blobs_to_check)):
                x = results[i][j]
                y = results[0][j]
                if not np.allclose(x, y,
                                   atol=self._threshold, rtol=self._threshold):
                    print('Failure in checking device option {}'
                          ' and output {}. The outputs are:'
                          .format(i, blobs_to_check[j]))
                    print(x.flatten())
                    print(y.flatten())
                    print(np.max(np.abs(x - y)))
                    success = False
                # else:
                #     print ('Passed device pair (%d, %d), %s %s: %s' %
                #            (i, j, blobs_to_check[j], y.shape,
                #             str(y.flatten())))
        workspace.SwitchWorkspace(old_ws_name)
        return success
