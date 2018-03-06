import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Uniform(Distribution):
    r"""
    Generates uniformly distributed random samples from the half-open interval
    `[low, high)`.

    Example::

        >>> m = Uniform(torch.Tensor([0.0]), torch.Tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
         2.3418
        [torch.FloatTensor of size 1]

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """
    # TODO allow (loc,scale) parameterization to allow independent constraints.
    params = {'low': constraints.dependent, 'high': constraints.dependent}
    has_rsample = True

    @property
    def mean(self):
        return (self.high + self.low) / 2

    @property
    def stddev(self):
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self):
        return (self.high - self.low).pow(2) / 12

    def __init__(self, low, high):
        self.low, self.high = broadcast_all(low, high)
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super(Uniform, self).__init__(batch_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = self.low.new(shape).uniform_()
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        lb = value.ge(self.low).type_as(self.low)
        ub = value.lt(self.high).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value):
        self._validate_log_prob_arg(value)
        result = (value - self.low) / (self.high - self.low)
        return result

    def icdf(self, value):
        self._validate_log_prob_arg(value)
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        return torch.log(self.high - self.low)
