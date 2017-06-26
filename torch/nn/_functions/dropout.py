import torch
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Dropout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    # classmethod so we can override _make_noise
    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if p > 0 and train:
            ctx.noise = cls._make_noise(input)
            if p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - p).div_(1 - p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None


class FeatureDropout(Dropout):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))
