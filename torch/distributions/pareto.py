from typing import Optional

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all
from torch.types import _size


__all__ = ["Pareto"]


class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """

    arg_constraints = {"alpha": constraints.positive, "scale": constraints.positive}

    def __init__(
        self,
        scale: Tensor | float,
        alpha: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale, self.alpha = broadcast_all(scale, alpha)
        base_dist = Exponential(self.alpha, validate_args=validate_args)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(
        self, batch_shape: _size, _instance: Optional["Pareto"] = None
    ) -> "Pareto":
        new = self._get_checked_instance(Pareto, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        # mean is inf for alpha <= 1
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1)

    @property
    def mode(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        # var is inf for alpha <= 2
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return constraints.greater_than_eq(self.scale)

    def entropy(self) -> Tensor:
        return (self.scale / self.alpha).log() + (1 + self.alpha.reciprocal())
