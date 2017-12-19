import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


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
        if isinstance(mean, Number) and isinstance(std, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mean.size()
        super(Normal, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.normal(self.mean.expand(shape), self.std.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.mean.new(shape).normal_()
        return self.mean + eps * self.std

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        # compute the variance
        var = (self.std ** 2)
        log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
        return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.std)
