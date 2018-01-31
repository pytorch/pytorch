import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Normal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    `loc` and `scale`.

    Example::

        >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of the distribution (often referred to as mu)
        scale (float or Tensor or Variable): standard deviation of the distribution
            (often referred to as sigma)
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).normal_()
        return self.loc + eps * self.scale

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
