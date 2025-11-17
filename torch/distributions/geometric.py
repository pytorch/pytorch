# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, Number


__all__ = ["Geometric"]


class Geometric(Distribution):
    r"""
    Creates a Geometric distribution parameterized by :attr:`probs`,
    where :attr:`probs` is the probability of success of Bernoulli trials.

    .. math::

        P(X=k) = (1-p)^{k} p, k = 0, 1, ...

    .. note::
        :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
        hence draws samples in :math:`\{0, 1, \ldots\}`, whereas
        :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Geometric(torch.tensor([0.3]))
        >>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
        tensor([ 2.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`. Must be in range (0, 1]
        logits (Number, Tensor): the log-odds of sampling `1`.
    """

    # pyrefly: ignore  # bad-override
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.nonnegative_integer

    def __init__(
        self,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            # pyrefly: ignore  # read-only
            (self.probs,) = broadcast_all(probs)
        else:
            assert logits is not None  # helps mypy
            # pyrefly: ignore  # read-only
            (self.logits,) = broadcast_all(logits)
        probs_or_logits = probs if probs is not None else logits
        if isinstance(probs_or_logits, _Number):
            batch_shape = torch.Size()
        else:
            assert probs_or_logits is not None  # helps mypy
            batch_shape = probs_or_logits.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self._validate_args and probs is not None:
            # Add an extra check beyond unit_interval
            value = self.probs
            valid = value > 0
            if not valid.all():
                invalid_value = value.data[~valid]
                raise ValueError(
                    "Expected parameter probs "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to be positive but found invalid values:\n{invalid_value}"
                )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Geometric, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        super(Geometric, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return 1.0 / self.probs - 1.0

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.probs)

    @property
    def variance(self) -> Tensor:
        return (1.0 / self.probs - 1.0) / self.probs

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        tiny = torch.finfo(self.probs.dtype).tiny
        with torch.no_grad():
            if torch._C._get_tracing_state():
                # [JIT WORKAROUND] lack of support for .uniform_()
                u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
                u = u.clamp(min=tiny)
            else:
                u = self.probs.new(shape).uniform_(tiny, 1)
            return (u.log() / (-self.probs).log1p()).floor()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value, probs = broadcast_all(value, self.probs)
        probs = probs.clone(memory_format=torch.contiguous_format)
        probs[(probs == 1) & (value == 0)] = 0
        return value * (-probs).log1p() + self.probs.log()

    def entropy(self):
        return (
            binary_cross_entropy_with_logits(self.logits, self.probs, reduction="none")
            / self.probs
        )
