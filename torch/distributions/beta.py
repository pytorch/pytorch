from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by `concentration1` and `concentration0`.

    Example::

        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        if isinstance(concentration1, Number) and isinstance(concentration0, Number):
            concentration1_concentration0 = torch.tensor([float(concentration1), float(concentration0)])
        else:
            concentration1, concentration0 = broadcast_all(concentration1, concentration0)
            concentration1_concentration0 = torch.stack([concentration1, concentration0], -1)
        self._dirichlet = Dirichlet(concentration1_concentration0)
        super(Beta, self).__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return (self.concentration1 * self.concentration0 /
                (total.pow(2) * (total + 1)))

    def rsample(self, sample_shape=()):
        value = self._dirichlet.rsample(sample_shape).select(-1, 0)
        if isinstance(value, Number):
            value = self._dirichlet.concentration.new_tensor(value)
        return value

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
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

    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
