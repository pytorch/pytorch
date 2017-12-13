from numbers import Number

import torch
from torch.autograd import Variable, Function
from torch.autograd.function import once_differentiable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import expand_n


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
        # TODO handle (Variable, Number) cases
        alpha_num = isinstance(alpha, Number)
        beta_num = isinstance(beta, Number)
        if alpha_num and not beta_num:
            alpha = beta.new(beta.size()).fill_(alpha)
        elif not alpha_num and beta_num:
            beta = alpha.new(alpha.size()).fill_(beta)
        elif alpha_num and beta_num:
            alpha, beta = torch.Tensor([alpha]), torch.Tensor([beta])
        elif alpha.size() != beta.size():
            raise ValueError('Expected alpha.size() == beta.size(), actual {} vs {}'.format(
                alpha.size(), beta.size()))
        self.alpha = alpha
        self.beta = beta

    def sample(self):
        return _standard_gamma(self.alpha) / self.beta

    def sample_n(self, n):
        return _standard_gamma(expand_n(self.alpha, n)) / self.beta

    def log_prob(self, value):
        return (self.alpha * torch.log(self.beta) +
                (self.alpha - 1) * torch.log(value) -
                self.beta * value - torch.lgamma(self.alpha))
