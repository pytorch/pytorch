# mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all
from torch.distributions.weibull import Weibull


__all__ = ["Frechet"]


class Frechet(TransformedDistribution):
    r"""
    Samples from a Fréchet distribution.

    .. math::
        f(x; \alpha, s, m) = \frac{\alpha}{s} \left(\frac{x-m}{s}\right)^{-1-\alpha}
        \exp\left(-\left(\frac{x-m}{s}\right)^{-\alpha}\right)

    Args:
        loc (float or Tensor): location parameter (m).
        scale (float or Tensor): scale parameter (s).
        concentration (float or Tensor): shape parameter (alpha).
        validate_args (bool, optional): Whether to validate arguments.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # Required as a class attribute for TestDistributions.test_support_attributes
    support = constraints.real

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Weibull(
            torch.ones_like(self.scale), self.concentration, validate_args=validate_args
        )
        # PowerTransform expects a Tensor for the exponent to satisfy typing
        exponent = torch.tensor([-1.0], device=self.loc.device, dtype=self.loc.dtype)
        transforms: list = [
            PowerTransform(exponent=exponent),
            AffineTransform(loc=self.loc, scale=self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)
        # For distributions with parameter-dependent support, we set _support
        self._support = constraints.greater_than(self.loc)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Frechet, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        exponent = torch.tensor([-1.0], device=new.loc.device, dtype=new.loc.dtype)
        transforms: list = [
            PowerTransform(exponent=exponent),
            AffineTransform(loc=new.loc, scale=new.scale),
        ]
        super(Frechet, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        new._support = constraints.greater_than(new.loc)
        return new

    @property
    def mean(self) -> Tensor:
        res = self.loc + self.scale * torch.exp(
            torch.lgamma(1 - self.concentration_reciprocal)
        )
        return torch.where(self.concentration > 1, res, float("nan"))

    @property
    def mode(self) -> Tensor:
        return self.loc + self.scale * (
            self.concentration / (self.concentration + 1)
        ).pow_(self.concentration_reciprocal)

    @property
    def variance(self) -> Tensor:
        res = self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 - 2 * self.concentration_reciprocal)).sub_(
                torch.exp(2 * torch.lgamma(1 - self.concentration_reciprocal))
            )
        )
        return torch.where(self.concentration > 2, res, float("nan"))

    def entropy(self) -> Tensor:
        return (
            1
            + euler_constant * (1 + self.concentration_reciprocal)
            + torch.log(self.scale * self.concentration_reciprocal)
        )