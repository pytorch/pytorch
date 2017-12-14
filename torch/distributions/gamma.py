from numbers import Number

import torch
from torch.autograd import Variable, Function
from torch.autograd.function import once_differentiable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import expand_n, broadcast_all


class _StandardGamma(Function):
    @staticmethod
    def forward(ctx, alpha):
        x = torch._C._standard_gamma(alpha)
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad = torch._C._standard_gamma_grad(x, alpha)
        return grad_output * grad


def _standard_gamma(alpha):
    if not isinstance(alpha, Variable):
        return torch._C._standard_gamma(alpha)
    return _StandardGamma.apply(alpha)


class Gamma(Distribution):
    r"""
    Creates a Gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> m = Gamma(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # Gamma distributed with shape alpha=1 and rate beta=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        alpha (float or Tensor or Variable): shape parameter of the distribution
        beta (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    has_rsample = True

    def __init__(self, alpha, beta):
        self.alpha, self.beta = broadcast_all(alpha, beta)

    def rsample(self, sample_shape=()):
        if len(sample_shape) == 0:
            return _standard_gamma(self.alpha) / self.beta
        elif len(sample_shape) == 1:
            return _standard_gamma(expand_n(self.alpha, sample_shape[0])) / self.beta
        else:
            raise NotImplementedError("rsample is not implemented for len(sample_shape)>1")

    def log_prob(self, value):
        return (self.alpha * torch.log(self.beta) +
                (self.alpha - 1) * torch.log(value) -
                self.beta * value - torch.lgamma(self.alpha))
