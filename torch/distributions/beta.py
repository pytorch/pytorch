# mypy: allow-untyped-defs
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _size
from typing import Union
from torch.distributions.utils import is_identically_zero, is_identically_one


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
        "concentration0": constraints.positive,
    }
    has_rsample = True

    def __init__(
        self,
        concentration1,
        concentration0,
        low: Union[Number, torch.Tensor] = 0.0,
        high: Union[Number, torch.Tensor] = 1.0,
        validate_args=None,
    ):
        self._low = low
        self._high = high

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

        if validate_args and not torch.gt(self.scale, 0).all():
            raise ValueError("Beta is not defined when low>= high")

        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        if isinstance(self._low, torch.Tensor):
            new._low = self.low.expand(batch_shape)
        else:
            new._low = self._low
        if isinstance(self._high, torch.Tensor):
            new._high = self.high.expand(batch_shape)
        else:
            new._high = self._high
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self._low, self._high)

    @property
    def mean(self):
        unscaled_mean = self.concentration1 / (
            self.concentration1 + self.concentration0
        )
        if self._has_default_scale_parameters():
            return unscaled_mean
        return self.location + self.scale * unscaled_mean

    @property
    def mode(self):
        unscaled_mode = self._dirichlet.mode[..., 0]
        if self._has_default_scale_parameters():
            return unscaled_mode
        return self.location + self.scale * unscaled_mode

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        unscaled_variance = (
            self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))
        )
        if self._has_default_scale_parameters():
            return unscaled_variance
        return (
            self.scale.pow(2) * unscaled_variance
        )

    def _has_default_scale_parameters(self):
        return is_identically_zero(self._low) and is_identically_one(self.high)

    def rsample(self, sample_shape=()):
        sample = self._dirichlet.rsample(sample_shape).select(-1, 0)
        if self._has_default_scale_parameters():
            return sample
        scaled_sample = self.location + self.scale * sample
        return scaled_sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if self._has_default_scale_parameters():
            heads_tails = torch.stack([value, 1.0 - value], -1)
            return self._dirichlet.log_prob(heads_tails)
        scaled_value = (value - self.location) / (self.scale)
        heads_tails = torch.stack([scaled_value, 1.0 - scaled_value], -1)
        return self._dirichlet.log_prob(heads_tails) - torch.log(self.scale)

    def entropy(self):
        if self._has_default_scale_parameters():
            return self._dirichlet.entropy()
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
        return self.low

    @property
    def scale(self):
        return self.high - self.low

    @property
    def low(self):
        if isinstance(self._low, Number):
            return torch.full_like(self.concentration0, self._low)
        else:
            return self._low

    @property
    def high(self):
        if isinstance(self._high, Number):
            return torch.full_like(self.concentration0, self._high)
        else:
            return self._high

    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, x, y):
        unscaled_log_normalizer = (torch.lgamma(x)
            + torch.lgamma(y)
            - torch.lgamma(x + y))
        if self._has_default_scale_parameters():
            return unscaled_log_normalizer
        return unscaled_log_normalizer + torch.log(self.scale)
