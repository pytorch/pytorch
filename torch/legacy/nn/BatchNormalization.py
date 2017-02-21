"""
        This file implements Batch Normalization as described in the paper:
        "Batch Normalization: Accelerating Deep Network Training
                              by Reducing Internal Covariate Shift"
                        by Sergey Ioffe, Christian Szegedy

        This implementation is useful for inputs NOT coming from convolution layers.
        For convolution layers, use nn.SpatialBatchNormalization.

        The operation implemented is:
        y =     ( x - mean(x) )
             ########## * gamma + beta
             standard-deviation(x)
        where gamma and beta are learnable parameters.

        The learning of gamma and beta is optional.

        Usage:
        with    learnable parameters: nn.BatchNormalization(N [, eps] [, momentum])
                                      where N = dimensionality of input
        without learnable parameters: nn.BatchNormalization(N [, eps] [, momentum], False)

        eps is a small value added to the standard-deviation to avoid divide-by-zero.
            Defaults to 1e-5

        In training time, this layer keeps a running estimate of it's computed mean and std.
        The running sum is kept with a default momentum of 0.1 (unless over-ridden)
        In test time, this running mean/std is used to normalize.
"""

import torch
from .Module import Module
from .utils import clear


class BatchNormalization(Module):
    # expected dimension of input
    nDim = 2

    def __init__(self, nOutput, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNormalization, self).__init__()
        assert nOutput != 0

        self.affine = affine
        self.eps = eps
        self.train = True
        self.momentum = momentum
        self.running_mean = torch.zeros(nOutput)
        self.running_var = torch.ones(nOutput)

        self.save_mean = None
        self.save_std = None
        self._gradOutput = None

        if self.affine:
            self.weight = torch.Tensor(nOutput)
            self.bias = torch.Tensor(nOutput)
            self.gradWeight = torch.Tensor(nOutput)
            self.gradBias = torch.Tensor(nOutput)
            self.reset()
        else:
            self.weight = None
            self.bias = None
            self.gradWeight = None
            self.gradBias = None

    def reset(self):
        if self.weight is not None:
            self.weight.uniform_()

        if self.bias is not None:
            self.bias.zero_()

        self.running_mean.zero_()
        self.running_var.fill_(1)

    def _checkInputDim(self, input):
        if input.dim() != self.nDim:
            raise RuntimeError(
                'only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.nDim, input.dim()))
        if input.size(1) != self.running_mean.nelement():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.nelement()))

    def _makeContiguous(self, input, gradOutput=None):
        if not input.is_contiguous():
            if self._input is None:
                self._input = input.new()
            self._input.resize_as_(input).copy_(input)
            input = self._input

        if gradOutput is not None:
            if not gradOutput.is_contiguous():
                if self._gradOutput is None:
                    self._gradOutput = gradOutput.new()
                self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
                gradOutput = self._gradOutput

        return input, gradOutput

    def updateOutput(self, input):
        self._checkInputDim(input)

        input = self._makeContiguous(input)[0]

        self.output.resize_as_(input)
        if self.save_mean is None:
            self.save_mean = input.new()
        self.save_mean.resize_as_(self.running_mean)
        if self.save_std is None:
            self.save_std = input.new()
        self.save_std.resize_as_(self.running_var)

        self._backend.BatchNormalization_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.save_mean,
            self.save_std,
            self.train,
            self.momentum,
            self.eps
        )

        return self.output

    def _backward(self, input, gradOutput, scale, gradInput=None, gradWeight=None, gradBias=None):
        self._checkInputDim(input)
        self._checkInputDim(gradOutput)
        if not hasattr(self, 'save_mean') or not hasattr(self, 'save_std'):
            raise RuntimeError('you have to call updateOutput() at least once before backward()')

        input, gradOutput = self._makeContiguous(input, gradOutput)

        scale = scale or 1.
        if gradInput is not None:
            gradInput.resize_as_(gradOutput)

        self._backend.BatchNormalization_backward(
            self._backend.library_state,
            input,
            gradOutput,
            gradInput,
            gradWeight,
            gradBias,
            self.weight,
            self.running_mean,
            self.running_var,
            self.save_mean,
            self.save_std,
            self.train,
            scale,
            self.eps
        )

        return self.gradInput

    def backward(self, input, gradOutput, scale=1.):
        return self._backward(input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)

    def updateGradInput(self, input, gradOutput):
        return self._backward(input, gradOutput, 1., self.gradInput)

    def accGradParameters(self, input, gradOutput, scale=1.):
        return self._backward(input, gradOutput, scale, None, self.gradWeight, self.gradBias)

    def read(self, file, version):
        super(BatchNormalization, self).read(self, file)
        if version < 2:
            if self.running_std:
                self.running_var = self.running_std.pow_(-2).add_(-self.eps)
                self.running_std = None

    def clearState(self):
        # first 5 buffers are not present in the current implementation,
        # but we keep them for cleaning old saved models
        clear(self, [
            'buffer',
            'buffer2',
            'centered',
            'std',
            'normalized',
            '_input',
            '_gradOutput',
            'save_mean',
            'save_std',
        ])
        return super(BatchNormalization, self).clearState()
