# mypy: allow-untyped-defs
from typing import Generic, Optional, TypeVar

import torch
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["Independent"]


D = TypeVar("D", bound=Distribution)


class Independent(Distribution, Generic[D]):
    r"""
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    base_dist: D

    def __init__(
        self,
        base_distribution: D,
        reinterpreted_batch_ndims: int,
        validate_args: Optional[bool] = None,
    ) -> None:
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        shape: Size = base_distribution.batch_shape + base_distribution.event_shape
        event_dim: int = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(
            batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        )
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self) -> bool:  # type: ignore[override]
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    # pyrefly: ignore  # bad-override
    def support(self):
        result = self.base_dist.support
        if self.reinterpreted_batch_ndims:
            result = constraints.independent(result, self.reinterpreted_batch_ndims)
        return result

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented"
            )
        return self.base_dist.enumerate_support(expand=expand)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.base_dist}, {self.reinterpreted_batch_ndims})"
        )
