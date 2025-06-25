# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import ComposeTransform, Transform
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["TransformedDistribution"]


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """

    arg_constraints: dict[str, constraints.Constraint] = {}

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Union[Transform, list[Transform]],
        validate_args: Optional[bool] = None,
    ) -> None:
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms"
                )
            self.transforms = transforms
        else:
            raise ValueError(
                f"transforms must be a Transform or list, but was {transforms}"
            )

        # Reshape base_distribution according to transforms.
        base_shape = base_distribution.batch_shape + base_distribution.event_shape
        base_event_dim = len(base_distribution.event_shape)
        transform = ComposeTransform(self.transforms)
        if len(base_shape) < transform.domain.event_dim:
            raise ValueError(
                f"base_distribution needs to have shape with size at least {transform.domain.event_dim}, but got {base_shape}."
            )
        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[
                : len(expanded_base_shape) - base_event_dim
            ]
            base_distribution = base_distribution.expand(base_batch_shape)
        reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(
                base_distribution, reinterpreted_batch_ndims
            )
        self.base_dist = base_distribution

        # Compute shapes.
        transform_change_in_event_dim = (
            transform.codomain.event_dim - transform.domain.event_dim
        )
        event_dim = max(
            transform.codomain.event_dim,  # the transform is coupled
            base_event_dim + transform_change_in_event_dim,  # the base dist is coupled
        )
        assert len(forward_shape) >= event_dim
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        shape = batch_shape + self.event_shape
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        base_batch_shape = shape[: len(shape) - len(self.base_dist.event_shape)]
        new.base_dist = self.base_dist.expand(base_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        if not self.transforms:
            return self.base_dist.support
        support = self.transforms[-1].codomain
        if len(self.event_shape) > support.event_dim:
            support = constraints.independent(
                support, len(self.event_shape) - support.event_dim
            )
        return support

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob: Union[Tensor, float] = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )
        return log_prob

    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value
