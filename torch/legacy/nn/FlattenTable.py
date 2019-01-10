import torch
from .Module import Module


class FlattenTable(Module):

    def __init__(self):
        super(FlattenTable, self).__init__()

        self.output = []
        self.input_map = []
        self.gradInput = []

    def _flatten(self, output, input):
        if isinstance(input, list):
            input_map = []
            # forward DFS order
            for i in range(len(input)):
                input_map.append(self._flatten(output, input[i]))
        else:
            input_map = len(output)
            output.append(input)

        return input_map

    def _checkMapping(self, output, input, input_map):
        if isinstance(input, list):
            if len(input) != len(input_map):
                return False

            # forward DFS order
            for i in range(len(input)):
                if not self._checkMapping(output, input[i], input_map[i]):
                    return False

            return True
        else:
            return output[input_map] is input

    # During BPROP we have to build a gradInput with the same shape as the
    # input.  This is a recursive function to build up a gradInput
    def _inverseFlatten(self, gradOutput, input_map):
        if isinstance(input_map, list):
            gradInput = []
            for i in range(len(input_map)):
                gradInput.append(self._inverseFlatten(gradOutput, input_map[i]))

            return gradInput
        else:
            return gradOutput[input_map]

    def updateOutput(self, input):
        assert isinstance(input, list)
        # to avoid updating rebuilding the flattened table every updateOutput call
        # we will: a DFS pass over the existing output table and the inputs to
        # see if it needs to be rebuilt.
        if not self._checkMapping(self.output, input, self.input_map):
            self.output = []
            self.input_map = self._flatten(self.output, input)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert isinstance(input, list)
        assert isinstance(gradOutput, list)
        # If the input changes between the updateOutput and updateGradInput call,
        #: we may have to rebuild the input_map!  However, let's assume that
        # the input_map is valid and that forward has already been called.

        # However, we should check that the gradInput is valid:
        if not self._checkMapping(gradOutput, self.gradInput, self.input_map):
            self.gradInput = self._inverseFlatten(gradOutput, self.input_map)

        return self.gradInput

    def type(self, type=None, tensorCache=None):
        if not type:
            return self._type
        # This function just stores references so we don't need to do any type
        # conversions. Just force the tables to be empty.
        self.clearState()

    def clearState(self):
        self.input_map = []
        return super(FlattenTable, self).clearState()
