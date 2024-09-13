# mypy: allow-untyped-defs
import torch
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.greater_than_eq(-1)
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        batch_shape = self.loc.size()
        super(GeneralizedPareto, self).__init__(
            batch_shape, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
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
        return -torch.log(self.scale) - torch.where(eq_zero, z, where_nonzero)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return 1 - torch.exp(torch.where(eq_zero, -z, where_nonzero))

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
        valid = concentration < 1
        safe_conc = torch.where(
            valid, concentration, torch.tensor(0.5, dtype=self.dtype)
        )
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, torch.full_like(result, float("nan")))

    @property
    def variance(self):
        concentration = self.concentration
        lim = torch.tensor(0.5, dtype=self.dtype)
        valid = concentration < lim
        safe_conc = torch.where(
            valid, concentration, torch.tensor(0.25, dtype=self.dtype)
        )
        result = self.scale**2 / (
            (1 - safe_conc) ** 2 * (1 - 2 * safe_conc)
        ) + torch.zeros_like(self.loc)
        return torch.where(valid, result, torch.full_like(result, float("nan")))

    def entropy(self):
        return torch.log(self.scale) + self.concentration + 1

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.greater_than_eq(self.scale)
