# mypy: allow-untyped-defs
import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _size


__all__ = ["ContinuousBernoulli"]


class ContinuousBernoulli(ExponentialFamily):
    r"""
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by 'probs' (in
    (0,1)) or 'logits' (real-valued). Note that, unlike the Bernoulli, 'probs'
    does not correspond to a probability and 'logits' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    _mean_carrier_measure = 0
    has_rsample = True

    def __init__(
        self, probs=None, logits=None, lims=(0.499, 0.501), validate_args=None
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            (self.probs,) = broadcast_all(probs)
            # validate 'probs' here if necessary as it is later clamped for numerical stability
            # close to 0 and 1, later on; otherwise the clamped 'probs' would always pass
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            self.probs = clamp_probs(self.probs)
        else:
            is_scalar = isinstance(logits, Number)
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        self._lims = lims
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        new._lims = self._lims
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def _outside_unstable_region(self):
        return torch.max(
            torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1])
        )

    def _cut_probs(self):
        return torch.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * torch.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(
            torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs)
        )
        cut_probs_above_half = torch.where(
            torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs)
        )
        log_norm = torch.log(
            torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))
        ) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    @property
    def mean(self):
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            torch.log1p(-cut_probs) - torch.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * torch.pow(x, 2)) * x
        return torch.where(self._outside_unstable_region(), mus, taylor)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / torch.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / torch.pow(torch.log1p(-cut_probs) - torch.log(cut_probs), 2)
        x = torch.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return torch.where(self._outside_unstable_region(), vars, taylor)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with torch.no_grad():
            return self.icdf(u)

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cut_probs = self._cut_probs()
        cdfs = (
            torch.pow(cut_probs, value) * torch.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = torch.where(self._outside_unstable_region(), cdfs, value)
        return torch.where(
            torch.le(value, 0.0),
            torch.zeros_like(value),
            torch.where(torch.ge(value, 1.0), torch.ones_like(value), unbounded_cdfs),
        )

    def icdf(self, value):
        cut_probs = self._cut_probs()
        return torch.where(
            self._outside_unstable_region(),
            (
                torch.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - torch.log1p(-cut_probs)
            )
            / (torch.log(cut_probs) - torch.log1p(-cut_probs)),
            value,
        )

    def entropy(self):
        log_probs0 = torch.log1p(-self.probs)
        log_probs1 = torch.log(self.probs)
        return (
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    @property
    def _natural_params(self):
        return (self.logits,)

    def _log_normalizer(self, x):
        """computes the log normalizing constant as a function of the natural parameter"""
        out_unst_reg = torch.max(
            torch.le(x, self._lims[0] - 0.5), torch.gt(x, self._lims[1] - 0.5)
        )
        cut_nat_params = torch.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * torch.ones_like(x)
        )
        log_norm = torch.log(torch.abs(torch.exp(cut_nat_params) - 1.0)) - torch.log(
            torch.abs(cut_nat_params)
        )
        taylor = 0.5 * x + torch.pow(x, 2) / 24.0 - torch.pow(x, 4) / 2880.0
        return torch.where(out_unst_reg, log_norm, taylor)
