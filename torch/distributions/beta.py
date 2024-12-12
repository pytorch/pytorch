# mypy: allow-untyped-defs
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _size


__all__ = ["Beta"]


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0` on t.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
        low (float or Tensor): lower range.
        high (float or Tensor): upper range.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive
    }
    has_rsample = True

    def __init__(
        self, concentration1, concentration0, low=0.0, high=1.0, validate_args=None
    ):
        self._scale = high - low
        self._location = low

        if isinstance(concentration1, Real) and isinstance(concentration0, Real):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        new._scale = self.scale.expand(batch_shape)
        new._location = self.location.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        low = self._location
        high = self._location + self._scale
        return(constraints.interval(low, high))

    @property
    def mean(self):
        return self.location + self.scale * (
            self.concentration1 / (self.concentration1 + self.concentration0)
        )

    @property
    def mode(self):
        return self.location + self.scale * self._dirichlet.mode[..., 0]

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return (
            self.scale.pow(2)
            * self.concentration1
            * self.concentration0
            / (total.pow(2) * (total + 1))
        )

    def rsample(self, sample_shape=()):
        sample = self._dirichlet.rsample(sample_shape).select(-1, 0)
        scaled_sample = self.location + self.scale * sample
        return scaled_sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        scaled_value = (value - self.location) / self.scale
        heads_tails = torch.stack([scaled_value, 1.0 - scaled_value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy() + torch.log(self.scale)

    @property
    def concentration1(self):
        result = self._dirichlet.concentration[..., 0]
        if isinstance(result, Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def concentration0(self):
        result = self._dirichlet.concentration[..., 1]
        if isinstance(result, Number):
            return torch.tensor([result])
        else:
            return result
        
    @property
    def location(self):
        if isinstance(self._location, Number):
            return torch.ones_like(self.concentration0) * self._location
        else:
            return self._location
        
    @property
    def scale(self):
        if isinstance(self._scale, Number):
            return torch.ones_like(self.concentration0) * self._scale
        else:
            return self._scale

    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, x, y):
        return (
            torch.lgamma(x)
            + torch.lgamma(y)
            - torch.lgamma(x + y)
            + torch.log(self.scale)
        )
