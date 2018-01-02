from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


def _poisson(rate):
    if not isinstance(rate, Variable):
        return torch._C._VariableFunctions.poisson(Variable(rate)).data
    return torch._C._VariableFunctions.poisson(rate)


class Poisson(Distribution):
    r"""
    Creates a Poisson distribution parameterized by `rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by
    $rate^k e^{-rate}/k!$

    Example::

        >>> m = Poisson(torch.Tensor([4]))
        >>> m.sample()
         3
        [torch.LongTensor of size 1]

    Args:
        rate (Number, Tensor or Variable): the rate parameter
    """
    params = {'rate': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, rate):
        self.rate, = broadcast_all(rate)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super(Poisson, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return _poisson(self.rate.expand(shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        param_shape = value.size()
        rate = self.rate.expand(param_shape)
        return (rate.log() * value) - rate - (value + 1).lgamma()
