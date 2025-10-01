from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size, Number


__all__ = ["Poisson"]


class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """

    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self) -> Tensor:
        return self.rate

    @property
    def mode(self) -> Tensor:
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        return self.rate

    def __init__(
        self,
        rate: Union[Tensor, Number],
        validate_args: Optional[bool] = None,
    ) -> None:
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, _Number):
            batch_shape = torch.Size()  # type: ignore[attr-defined]
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(
        self,
        batch_shape: _size,
        _instance: Optional[Any] = None,
    ) -> "Poisson":
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)  # type: ignore[attr-defined]
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(
        self,
        sample_shape: _size = torch.Size(),  # type: ignore[attr-defined]
    ) -> Tensor:
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))  # type: ignore[attr-defined]

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        # Ensure value is treated as Tensor for operations
        assert isinstance(value, Tensor), "broadcast_all should return Tensor"
        return value.xlogy(rate) - rate - (value + 1).lgamma()  # type: ignore[attr-defined]

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.log(self.rate),)  # type: ignore[attr-defined]

    def _log_normalizer(self, x: Tensor) -> Tensor:
        return torch.exp(x)  # type: ignore[attr-defined]
