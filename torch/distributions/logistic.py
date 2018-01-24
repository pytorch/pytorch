import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, _finfo


class Logistic(Distribution):
    r"""
    Creates a logistic distribution parameterized by
    `loc` and `scale`.

    Example::

        >>> m = Logistic(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # logistic distributed with loc=0 and scale=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of the distribution (often referred to as mu)
        scale (float or Tensor or Variable): approximately standard deviation of the distribution
            (often referred to as sigma)
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Logistic, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _finfo(self.scale).eps
        U = self.loc.new(shape).uniform_(eps, 1 - eps)
        return self.loc + self.scale * (U.log() - (-U).log1p())

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        x = -(value - self.loc) / self.scale
        return x - self.scale.log() - 2 * x.exp().log1p()

    def entropy(self):
        return self.scale.log() + 2
