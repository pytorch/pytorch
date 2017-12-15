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
    has_rsample = True

    def __init__(self, mean, std):
        self.mean, self.std = broadcast_all(mean, std)

    def sample(self, sample_shape=()):
        if len(sample_shape) == 0:
            return torch.normal(self.mean, self.std)
        elif len(sample_shape) == 1:
            return torch.normal(expand_n(self.mean, sample_shape[0]), expand_n(self.std, sample_shape[0]))
        else:
            raise NotImplementedError("sample is not implemented for len(sample_shape)>1")

    def rsample(self, sample_shape=()):
        if len(sample_shape) == 0:
            eps = self.mean.new((self.mean + self.std).size())
            eps.normal_()
            return self.mean + self.std * eps
        elif len(sample_shape) == 1:
            expanded_mean = expand_n(self.mean, sample_shape[0])
            expanded_std = expand_n(self.std, sample_shape[0])
            eps = expanded_mean.new()
            eps.normal_()
            return expanded_mean + expanded_std * eps
        else:
            raise NotImplementedError("rsample is not implemented for len(sample_shape)>1")

    def log_prob(self, value):
        # compute the variance
        var = (self.std ** 2)
        log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
        return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
