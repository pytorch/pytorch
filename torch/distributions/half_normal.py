import math
from typing import Optional, Union

from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform
from torch.types import _size


__all__ = ["HalfNormal"]


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    base_dist: Normal

    def __init__(
        self,
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Normal(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(
        self, batch_shape: _size, _instance: Optional["HalfNormal"] = None
    ) -> "HalfNormal":
        new = self._get_checked_instance(HalfNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self) -> Tensor:
        return self.scale * 0.0

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        # For half-normal: log_prob(x) = log(2) + base_dist.log_prob(x) for x >= 0
        # But we need to handle negative values properly
        mask = value >= 0
        log_prob = self.base_dist.log_prob(value.abs()) + math.log(2)
        return log_prob * mask + (-inf) * (~mask)

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob: Tensor) -> Tensor:
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self) -> Tensor:
        return self.base_dist.entropy() - math.log(2)
