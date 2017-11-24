import torch
from .Module import Module


class SpatialZeroPadding(Module):

    def __init__(self, pad_l, pad_r=None, pad_t=None, pad_b=None):
        super(SpatialZeroPadding, self).__init__()
        self.pad_l = pad_l
        self.pad_r = pad_r if pad_r is not None else pad_l
        self.pad_t = pad_t if pad_t is not None else pad_l
        self.pad_b = pad_b if pad_b is not None else pad_l

    def updateOutput(self, input):
        assert input.dim() == 4

        # sizes
        h = input.size(2) + self.pad_t + self.pad_b
        w = input.size(3) + self.pad_l + self.pad_r
        if w < 1 or h < 1:
            raise RuntimeError('input is too small (feature map size: {}x{})'.format(h, w))
        self.output.resize_(input.size(0), input.size(1), h, w)
        self.output.zero_()
        # crop input if necessary
        c_input = input
        if self.pad_t < 0:
            c_input = c_input.narrow(2, 0 - self.pad_t, c_input.size(2) + self.pad_t)
        if self.pad_b < 0:
            c_input = c_input.narrow(2, 0, c_input.size(2) + self.pad_b)
        if self.pad_l < 0:
            c_input = c_input.narrow(3, 0 - self.pad_l, c_input.size(3) + self.pad_l)
        if self.pad_r < 0:
            c_input = c_input.narrow(3, 0, c_input.size(3) + self.pad_r)
        # crop output if necessary
        c_output = self.output
        if self.pad_t > 0:
            c_output = c_output.narrow(2, 0 + self.pad_t, c_output.size(2) - self.pad_t)
        if self.pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.size(2) - self.pad_b)
        if self.pad_l > 0:
            c_output = c_output.narrow(3, 0 + self.pad_l, c_output.size(3) - self.pad_l)
        if self.pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.size(3) - self.pad_r)
        # copy input to output
        c_output.copy_(c_input)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 4

        self.gradInput.resize_as_(input).zero_()
        # crop gradInput if necessary
        cg_input = self.gradInput
        if self.pad_t < 0:
            cg_input = cg_input.narrow(2, 0 - self.pad_t, cg_input.size(2) + self.pad_t)
        if self.pad_b < 0:
            cg_input = cg_input.narrow(2, 0, cg_input.size(2) + self.pad_b)
        if self.pad_l < 0:
            cg_input = cg_input.narrow(3, 0 - self.pad_l, cg_input.size(3) + self.pad_l)
        if self.pad_r < 0:
            cg_input = cg_input.narrow(3, 0, cg_input.size(3) + self.pad_r)
        # crop gradOutput if necessary
        cg_output = gradOutput
        if self.pad_t > 0:
            cg_output = cg_output.narrow(2, 0 + self.pad_t, cg_output.size(2) - self.pad_t)
        if self.pad_b > 0:
            cg_output = cg_output.narrow(2, 0, cg_output.size(2) - self.pad_b)
        if self.pad_l > 0:
            cg_output = cg_output.narrow(3, 0 + self.pad_l, cg_output.size(3) - self.pad_l)
        if self.pad_r > 0:
            cg_output = cg_output.narrow(3, 0, cg_output.size(3) - self.pad_r)
        # copy gradOutput to gradInput
        cg_input.copy_(cg_output)

        return self.gradInput

    def __tostring__(self, ):
        s = super(SpatialZeroPadding, self).__repr__()
        s += '({}, {}, {}, {})'.foramat(self.pad_l, self.pad_r, self.pad_t, self.pad_b)
        return s
