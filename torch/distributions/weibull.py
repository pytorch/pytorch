import torch
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all

__all__ = ["Weibull"]


class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
    """
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive

    def __init__(self, scale, concentration, validate_args=None):
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Weibull, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    @property
    def mode(self):
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self):
        return self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal))
            - torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal))
        )

    def entropy(self):
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            + torch.log(self.scale * self.concentration_reciprocal)
            + 1
        )
