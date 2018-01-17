from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Beta(Distribution):
    r"""
    Beta distribution parameterized by `concentration1` and `concentration0`.

    Example::

        >>> m = Beta(torch.Tensor([0.5]), torch.Tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        concentration1 (float or Tensor or Variable): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor or Variable): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    params = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0):
        if isinstance(concentration1, Number) and isinstance(concentration0, Number):
            concentration1_concentration0 = torch.Tensor([concentration1, concentration0])
        else:
            concentration1, concentration0 = broadcast_all(concentration1, concentration0)
            concentration1_concentration0 = torch.stack([concentration1, concentration0], -1)
        self._dirichlet = Dirichlet(concentration1_concentration0)
        super(Beta, self).__init__(self._dirichlet._batch_shape)

    def rsample(self, sample_shape=()):
        value = self._dirichlet.rsample(sample_shape).select(-1, 0)
        if isinstance(value, Number):
            value = self._dirichlet.concentration.new([value])
        return value

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def concentration1(self):
        result = self._dirichlet.concentration[..., 0]
        if isinstance(result, Number):
            return torch.Tensor([result])
        else:
            return result

    @property
    def concentration0(self):
        result = self._dirichlet.concentration[..., 1]
        if isinstance(result, Number):
            return torch.Tensor([result])
        else:
            return result
