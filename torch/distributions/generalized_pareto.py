# mypy: allow-untyped-defs
from numbers import Number, Real
import math

import torch
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.
    
    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when loc = 0, concentration = 0), Pareto (when concentration > 0,
    loc = scale / concentration), and Uniform (when concentration = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=1, scale=1, and concentration=1
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(concentration, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        y = 1 / safe_conc + torch.ones_like(z)
        where_nonzero = torch.where(y == 0, y, y * torch.log1p(safe_conc * z))
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return -log_scale - torch.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(eq_zero, torch.ones_like(self.concentration), self.concentration)
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        return torch.log1p(-torch.exp(self.log_survival_function(value)))

    def cdf(self, value):
        return torch.exp(self.log_cdf(value))

    def icdf(self, value):
        k = self.concentration
        m = self.loc
        s = self.scale
        is_k_zero = torch.isclose(k, torch.tensor(0.0))
        safe_k = torch.where(is_k_zero, torch.ones_like(k), k)
        neglog1mp = -torch.log1p(-value)
        return m + s * torch.where(
            is_k_zero, neglog1mp, torch.expm1(k * neglog1mp) / safe_k
        )

    def _z(self, x):
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        concentration = self.concentration
        dtype = self.concentration.dtype
        valid = concentration < 1
        safe_conc = torch.where(
            valid, concentration, torch.tensor(0.5, dtype=dtype)
        )
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, torch.full_like(result, float("nan")))

    @property
    def variance(self):
        concentration = self.concentration
        dtype = self.concentration.dtype
        lim = torch.tensor(0.5, dtype=dtype)
        valid = concentration < lim
        safe_conc = torch.where(
            valid, concentration, torch.tensor(0.25, dtype=dtype)
        )
        result = self.scale**2 / (
            (1 - safe_conc) ** 2 * (1 - 2 * safe_conc)
        ) + torch.zeros_like(self.loc)
        return torch.where(valid, result, torch.full_like(result, float("nan")))

    def entropy(self):
        ans = torch.log(self.scale) + self.concentration + 1
        return torch.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        if (self.concentration < 0).any():
            upper = self.loc + self.scale / self.concentration.abs()
            return constraints.interval(self.loc, upper)
        else:
            return constraints.greater_than_eq(self.loc)
