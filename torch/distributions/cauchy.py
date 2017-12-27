from numbers import Number

import math

import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Cauchy(Distribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> m = Cauchy(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a cauchy distribution with loc=0 and scale=1
         2.3214
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mode or median of the distribution.
        scale (float or Tensor or Variable): half width at half maximum.
    """
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Cauchy, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return -math.log(math.pi) - self.scale.log() - (1 + ((value - self.loc) / self.scale)**2).log()

    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
