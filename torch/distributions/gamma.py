from numbers import Number

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _finfo, broadcast_all, lazy_property


def _standard_gamma(concentration):
    return concentration._standard_gamma()


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape `concentration` and `rate`.

    Example::

        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / self.rate.pow(2)

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super(Gamma, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(shape)
        value.data.clamp_(min=_finfo(value).tiny)  # do not record in autograd graph
        return value

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self.concentration * torch.log(self.rate) +
                (self.concentration - 1) * torch.log(value) -
                self.rate * value - torch.lgamma(self.concentration))

    def entropy(self):
        return (self.concentration - torch.log(self.rate) + torch.lgamma(self.concentration) +
                (1.0 - self.concentration) * torch.digamma(self.concentration))

    @property
    def _natural_params(self):
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())
