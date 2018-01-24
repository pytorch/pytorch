import torch
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
from itertools import repeat


class Dropout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
        return r

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if ctx.p == 1:
            ctx.noise.fill_(0)
        else:
            ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
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
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("FeatureDropout", "training mode")
        return input

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))


class Zoneout(InplaceFunction):

    @staticmethod
    def _make_mask(input):
        return input.new().resize_as_(input).expand_as(input)

    @classmethod
    def forward(cls, ctx, current_input, previous_input, p=None, mask=None,
                train=False, inplace=False):
        if p is None and mask is None:
            raise ValueError('Either p or mask must be provided')
        if p is not None and mask is not None:
            raise ValueError('Only one of p and mask can be provided')
        if p is not None and (p < 0 or p > 1):
            raise ValueError('zoneout probability has to be between 0 and 1, '
                             'but got {}'.format(p))
        if mask is not None and \
                not isinstance(mask, torch.ByteTensor) and \
                not isinstance(mask, torch.cuda.ByteTensor):
            raise ValueError("mask must be a ByteTensor")
        if current_input.size() != previous_input.size():
            raise ValueError(
                'Current and previous inputs must be of the same '
                'size, but current has size {current} and '
                'previous has size {previous}.'.format(
                    current='x'.join(str(size) for size in current_input.size()),
                    previous='x'.join(str(size) for size in previous_input.size()))
            )
        if type(current_input) != type(previous_input):
            raise ValueError('Current and previous inputs must be of the same '
                             'type, but current is {current} and previous is '
                             '{previous}'.format(current=type(current_input),
                                                 previous=type(previous_input))
                             )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(current_input)
            output = current_input
        else:
            output = current_input.clone()

        if not ctx.train:
            return output

        ctx.current_mask = cls._make_mask(current_input)
        ctx.previous_mask = cls._make_mask(current_input)
        if mask is None:
            ctx.current_mask.bernoulli_(1 - ctx.p)
        else:
            ctx.current_mask.fill_(0).masked_fill_(mask, 1)
        ctx.previous_mask.fill_(1).sub_(ctx.current_mask)
        output.mul_(ctx.current_mask).add_(previous_input.mul(ctx.previous_mask))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * Variable(ctx.current_mask), grad_output * Variable(ctx.previous_mask), \
                None, None, None, None
        else:
            return grad_output, None, None, None, None, None
