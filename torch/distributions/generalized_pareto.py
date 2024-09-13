# mypy: allow-untyped-defs
import torch
from torch import nn
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    r"""
    Generalized Pareto distribution.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when loc = 0, concentration = 0), Pareto (when concentration > 0,
    loc = scale / concentration), and Uniform (when concentration = -1).

    This distribution is often used to model the tails of other distributions.

    Args:
        loc (float or Tensor): The location parameter of the distribution.
        scale (float or Tensor): The scale parameter of the distribution. Must be positive.
        concentration (float or Tensor): The shape parameter of the distribution.

    Example::

        >>> m = GeneralizedPareto(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.3]))
        >>> m.sample()  # sample from the distribution
        tensor([1.5464])

    Note:
        The support of the distribution is always lower bounded by `loc`. When `concentration < 0`,
        the support is also upper bounded by `loc + scale / abs(concentration)`.
    """
    arg_constraints = {'loc': constraints.real,
                       'scale': constraints.positive,
                       'concentration': constraints.real}
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(loc, scale, concentration)
        if isinstance(loc, nn.Parameter):
            base_dist = "loc"
        elif isinstance(scale, nn.Parameter):
            base_dist = "scale"
        else:
            base_dist = "concentration"
        super(GeneralizedPareto, self).__init__(self.loc.size(), validate_args=validate_args)
        self._base_dist = base_dist

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = torch.where(valid, concentration, torch.tensor(0.5))
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, torch.tensor(float('nan')))

    @property
    def variance(self):
        concentration = self.concentration
        lim = torch.tensor(0.5)
        valid = concentration < lim
        safe_conc = torch.where(valid, concentration, torch.tensor(0.25))
        result = (self.scale**2 / ((1 - safe_conc)**2 * (1 - 2 * safe_conc))) + torch.zeros_like(self.loc)
        return torch.where(valid, result, torch.tensor(float('nan')))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        loc, scale, concentration = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        eq_zero = torch.isclose(concentration, torch.tensor(0., device=concentration.device))
        nonzero_conc = torch.where(eq_zero, torch.tensor(1., device=concentration.device), concentration)
        y = 1 / nonzero_conc + torch.ones_like(z)
        where_nonzero = torch.where(y == 0, y, y * torch.log1p(nonzero_conc * z))
        return -torch.log(scale) - torch.where(eq_zero, z, where_nonzero)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        loc, scale, concentration = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        eq_zero = torch.isclose(concentration, torch.tensor(0., device=concentration.device))
        nonzero_conc = torch.where(eq_zero, torch.tensor(1., device=concentration.device), concentration)
        where_nonzero = 1 - torch.pow(1 + nonzero_conc * z, -1 / nonzero_conc)
        return torch.where(eq_zero, 1 - torch.exp(-z), where_nonzero)

    def icdf(self, value):
        loc, scale, concentration = self.loc, self.scale, self.concentration
        eq_zero = torch.isclose(concentration, torch.tensor(0., device=concentration.device))
        safe_conc = torch.where(eq_zero, torch.tensor(1., device=concentration.device), concentration)
        where_nonzero = loc + scale / safe_conc * (torch.pow(1 - value, -safe_conc) - 1)
        where_zero = loc - scale * torch.log1p(-value)
        return torch.where(eq_zero, where_zero, where_nonzero)

    def log_survival_function(self, value):
        if self._validate_args:
            self._validate_sample(value)
        loc, scale, concentration = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        eq_zero = torch.isclose(concentration, torch.tensor(0., device=concentration.device))
        nonzero_conc = torch.where(eq_zero, torch.tensor(1., device=concentration.device), concentration)
        where_nonzero = -torch.log1p(nonzero_conc * z) / nonzero_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def entropy(self):
        return torch.log(self.scale) + self.concentration + 1

    @property
    def _natural_params(self):
        return (self.loc, self.scale, self.concentration)

    def _validate_args(self):
        if not self.validate_args:
            return
        with torch.no_grad():
            torch.distributions.utils.broadcast_all(self.loc, self.scale, self.concentration)
        if self.scale.min() <= 0:
            raise ValueError("scale parameter must be positive")

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        if (self.concentration < 0).any():
            upper = self.loc + self.scale / self.concentration.abs()
            return constraints.interval(self.loc, upper)
        else:
            return constraints.greater_than_eq(self.loc)
