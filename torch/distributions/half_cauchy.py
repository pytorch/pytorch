import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions import Cauchy
from torch.distributions.utils import broadcast_all


class HalfCauchy(Distribution):
    r"""
    Half-Cauchy distribution.

    This is a continuous distribution with lower-bounded domain (`x > mu`).
    See also the `Cauchy` distribution.

    Example::

        >>> m = HalfCauchy(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        >>> m.sample()  # sample from HalfCauchy distributed with loc=0 and scale=1
         0.3141
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): lower bound of the distribution.
        scale (float or Tensor or Variable): half width at half maximum.
    """

    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        self._cauchy = Cauchy(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(HalfCauchy, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        sample = self._cauchy.rsample(sample_shape)
        return sample.abs()

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        x_0 = torch.pow((value - self.loc) / self.scale, 2)
        px = 2 / (math.pi * self.scale * (1 + x_0))
        return torch.log(px)

    def entropy(self):
        return math.log(2 * math.pi) + self.scale.log()
