from numbers import Number

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _finfo, broadcast_all


def _dirichlet_sample_nograd(concentration):
    probs = torch._standard_gamma(concentration)
    probs /= probs.sum(-1, True)
    eps = _finfo(probs).eps
    return probs.clamp_(min=eps, max=1 - eps)


# This helper is exposed for testing.
def _Dirichlet_backward(x, concentration, grad_output):
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    @staticmethod
    def forward(ctx, concentration):
        x = _dirichlet_sample_nograd(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    r"""
    Creates a Dirichlet distribution parameterized by concentration `concentration`.

    Example::

        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentrarion concentration
         0.1046
         0.8954
        [torch.FloatTensor of size 2]

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """
    arg_constraints = {'concentration': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        self.concentration, = broadcast_all(concentration)
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(Dirichlet, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        if isinstance(concentration, torch.Tensor):
            return _Dirichlet.apply(concentration)
        return _dirichlet_sample_nograd(concentration)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((torch.log(value) * (self.concentration - 1.0)).sum(-1) +
                torch.lgamma(self.concentration.sum(-1)) -
                torch.lgamma(self.concentration).sum(-1))

    @property
    def mean(self):
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def variance(self):
        con0 = self.concentration.sum(-1, True)
        return self.concentration * (con0 - self.concentration) / (con0.pow(2) * (con0 + 1))

    def entropy(self):
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (torch.lgamma(self.concentration).sum(-1) - torch.lgamma(a0) -
                (k - a0) * torch.digamma(a0) -
                ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1))

    @property
    def _natural_params(self):
        return (self.concentration, )

    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))
