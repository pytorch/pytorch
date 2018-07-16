import torch
from torch.autograd.function import InplaceFunction
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
            return grad_output * ctx.noise, None, None, None
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


class AlphaDropout(Dropout):

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("AlphaDropout", "training mode")
        return input

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
            a = 0
            b = ctx.noise
        else:
            ctx.noise.bernoulli_(1 - ctx.p)
            alpha = 1.7580993408473766
            a = ((alpha ** 2 * ctx.p + 1) * (1 - ctx.p)) ** (-0.5)
            b = ctx.noise.add(-1).mul_(alpha * a).add_(alpha * a * ctx.p)
        ctx.noise = ctx.noise.mul_(a).expand_as(input)
        b = b.expand_as(input)
        output.mul_(ctx.noise).add_(b)

        return output


class FeatureAlphaDropout(AlphaDropout):

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        # NB: In inference mode, FeatureDropout is exported as an identity op.
        from torch.onnx.symbolic import _unimplemented
        if train:
            return _unimplemented("FeatureAlphaDropout", "training mode")
        return input

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))
