# mypy: allow-untyped-defs
import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Exponential"]


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.rate)

    @property
    def stddev(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        return self.rate.pow(-2)

    def __init__(self, rate, validate_args=None):
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, _Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -torch.log1p(-value) / self.rate

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (-self.rate,)

    def _log_normalizer(self, x):
        return -torch.log(-x)
