from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by `rate`.

    Example::

        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """
    arg_constraints = {'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.rate.reciprocal()

    @property
    def stddev(self):
        return self.rate.reciprocal()

    @property
    def variance(self):
        return self.rate.pow(-2)

    def __init__(self, rate, validate_args=None):
        self.rate, = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super(Exponential, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(1 - value) / self.rate

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self):
        return (-self.rate, )

    def _log_normalizer(self, x):
        return -torch.log(-x)
