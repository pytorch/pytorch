import math
from numbers import Number

import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import expand_n, broadcast_all


class Normal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    `mean` and `std`.

    Example::

        >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normally distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        mean (float or Tensor or Variable): mean of the distribution
        std (float or Tensor or Variable): standard deviation of the distribution
    """

    def __init__(self, mean, std):
        self.mean, self.std = broadcast_all(mean, std)

    def sample(self):
        return torch.normal(self.mean, self.std)

    def sample_n(self, n):
        return torch.normal(expand_n(self.mean, n), expand_n(self.std, n))

    def log_prob(self, value):
        # compute the variance
        var = (self.std ** 2)
        log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
        return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
