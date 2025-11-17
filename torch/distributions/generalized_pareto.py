# mypy: allow-untyped-defs
import math
from numbers import Number, Real

import torch
from torch import inf, nan
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the
    implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """

    # pyrefly: ignore  # bad-override
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if (
            isinstance(loc, Number)
            and isinstance(scale, Number)
            and isinstance(concentration, Number)
        ):
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

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
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
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        return torch.log1p(-torch.exp(self.log_survival_function(value)))

    def cdf(self, value):
        return torch.exp(self.log_cdf(value))

    def icdf(self, value):
        loc = self.loc
        scale = self.scale
        concentration = self.concentration
        eq_zero = torch.isclose(concentration, torch.zeros_like(concentration))
        safe_conc = torch.where(eq_zero, torch.ones_like(concentration), concentration)
        logu = torch.log1p(-value)
        where_nonzero = loc + scale / safe_conc * torch.expm1(-safe_conc * logu)
        where_zero = loc - scale * logu
        return torch.where(eq_zero, where_zero, where_nonzero)

    def _z(self, x):
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = torch.where(valid, concentration, 0.5)
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, nan)

    @property
    def variance(self):
        concentration = self.concentration
        valid = concentration < 0.5
        safe_conc = torch.where(valid, concentration, 0.25)
        # pyrefly: ignore  # unsupported-operation
        result = self.scale**2 / ((1 - safe_conc) ** 2 * (1 - 2 * safe_conc))
        return torch.where(valid, result, nan)

    def entropy(self):
        ans = torch.log(self.scale) + self.concentration + 1
        return torch.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore  # bad-override
    def support(self):
        lower = self.loc
        upper = torch.where(
            self.concentration < 0, lower - self.scale / self.concentration, inf
        )
        return constraints.interval(lower, upper)
