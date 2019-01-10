import torch
from .Module import Module


class VolumetricAveragePooling(Module):

    def __init__(self, kT, kW, kH, dT=None, dW=None, dH=None,
                 padT=0, padW=0, padH=0,
                 ceil_mode=False, count_include_pad=True):
        super(VolumetricAveragePooling, self).__init__()
        self.kT = kT
        self.kH = kH
        self.kW = kW
        self.dT = dT or kT
        self.dW = dW or kW
        self.dH = dH or kH
        self.padT = padT
        self.padW = padW
        self.padH = padH
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault('padT', 0)
        self.__dict__.setdefault('padH', 0)
        self.__dict__.setdefault('padW', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)

    def updateOutput(self, input):
        self._backend.VolumetricAveragePooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            self.ceil_mode, self.count_include_pad
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.VolumetricAveragePooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            self.ceil_mode, self.count_include_pad
        )
        return self.gradInput

    def __repr__(self):
        s = super(VolumetricAveragePooling, self).__repr__()
        s += '({}x{}x{}, {}x{}x{}, {}x{}x{}, {}, {}'.format(
            self.kT, self.kW, self.kH, self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            self.ceil_mode, self.count_include_pad)
        s += ')'
        return s
