from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all


class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by `rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by
    $rate^k e^{-rate}/k!$

    Example::

        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
         3
        [torch.LongTensor of size 1]

    Args:
        rate (Number, Tensor): the rate parameter
    """
    arg_constraints = {'rate': constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def __init__(self, rate, validate_args=None):
        self.rate, = broadcast_all(rate)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super(Poisson, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        return (rate.log() * value) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self):
        return (torch.log(self.rate), )

    def _log_normalizer(self, x):
        return torch.exp(x)
