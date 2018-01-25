from numbers import Number

import math

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all, lazy_property


class Pareto(ExponentialFamily):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
         1.5623
        [torch.FloatTensor of size 1]

    Args:
        scale (float or Tensor or Variable): Scale parameter of the distribution
        alpha (float or Tensor or Variable): Shape parameter of the distribution
    """
    has_rsample = True
    params = {'alpha': constraints.positive, 'scale': constraints.positive}
    _zero_carrier_measure = True

    def __init__(self, scale, alpha):
        self.scale, self.alpha = broadcast_all(scale, alpha)
        if isinstance(scale, Number) and isinstance(alpha, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super(Pareto, self).__init__(batch_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.greater_than(self.scale)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        exp_dist = self.alpha.new(shape).exponential_()
        return self.scale * torch.exp(exp_dist / self.alpha)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return torch.log(self.alpha / value) + self.alpha * (self.scale / value).log()

    def entropy(self):
        return ((self.scale / self.alpha).log() + (1 + self.alpha.reciprocal()))

    def natural_params(self):
        return self._natural_params

    @lazy_property
    def _natural_params(self):
        # note that self.scale is fixed, meaning that it is not a natural parameter
        if isinstance(self.alpha, Variable):
            V1 = Variable(-self.alpha.data - 1, requires_grad=True)
            V2 = Variable(self.scale.data)
        else:
            V1 = Variable(-self.alpha - 1, requires_grad=True)
            V2 = Variable(self.scale)
        return (V1, V2)

    def log_normalizer(self):
        x, y = self._natural_params
        term = x + 1
        return -torch.log(-term) + term * torch.log(y)
