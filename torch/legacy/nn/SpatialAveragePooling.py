import torch
from .Module import Module


class SpatialAveragePooling(Module):

    def __init__(self, kW, kH, dW=1, dH=1, padW=0, padH=0):
        super(SpatialAveragePooling, self).__init__()

        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH
        self.ceil_mode = False
        self.count_include_pad = True
        self.divide = True

    def ceil(self):
        self.ceil_mode = True
        return self

    def floor(self):
        self.ceil_mode = False
        return self

    def setCountIncludePad(self):
        self.count_include_pad = True
        return self

    def setCountExcludePad(self):
        self.count_include_pad = False
        return self

    def updateOutput(self, input):
        self._backend.SpatialAveragePooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.ceil_mode,
            self.count_include_pad
        )
        # for backward compatibility with saved models
        # which are not supposed to have "divide" field
        if not self.divide:
            self.output.mul_(self.kW * self.kH)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is not None:
            self._backend.SpatialAveragePooling_updateGradInput(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradInput,
                self.kW, self.kH,
                self.dW, self.dH,
                self.padW, self.padH,
                self.ceil_mode,
                self.count_include_pad
            )
            # for backward compatibility
            if not self.divide:
                self.gradInput.mul_(self.kW * self.kH)

            return self.gradInput

    def __repr__(self):
        s = super(SpatialAveragePooling, self).__repr__()
        s += '({}x{}, {}, {}'.format(self.kW, self.kH, self.dW, self.dH)
        if (self.padW or self.padH) and (self.padW != 0 or self.padH != 0):
            s += ', {}, {}'.format(self.padW, self.padH)
        s += ')'
        return s
