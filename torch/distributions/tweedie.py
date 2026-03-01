# mypy: allow-untyped-defs

import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.poisson import Poisson
from torch.distributions.utils import broadcast_all, lazy_property
from torch.types import _Number


__all__ = ["Tweedie"]


class Tweedie(Distribution):
    r"""
    Creates a Tweedie distribution parameterized by :attr:`mu`, :attr:`dispersion` and :attr:`power`.
    The Tweedie distribution is a special case of the exponential dispersion model with variance function
    ..math:
      \mathrm{Var}(Y) = \mathrm{dispersion} * \mu^{\mathrm{power}}
    where :math:`1 < \mathrm{power} < 2`. It can be represented as a compound Poisson-Gamma distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Tweedie(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([1.5]))
        >>> m.sample()  # Tweedie distributed with mu=2, dispersion=1, power=1.5
        tensor([ 1.4784])

    Args:
        mu (float or Tensor): mean parameter of the distribution
        dispersion (float or Tensor): dispersion parameter of the distribution
        power (float or Tensor): power parameter of the distribution, must be in (1,2).

    Reference:
        Dunn, P. K., & Smyth, G. K. (2005). Series evaluation of Tweedie exponential
        dispersion model densities. Statistics and Computing, 15(4), 267-280.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "mu": constraints.positive,
        "dispersion": constraints.positive,
        "power": constraints.interval(1, 2),
    }
    support = constraints.nonnegative
    _mean_carrier_measure = 0

    def __init__(self, mu, dispersion, power, validate_args=None):
        self.mu, self.dispersion, self.power = broadcast_all(mu, dispersion, power)
        if (
            isinstance(mu, _Number)
            and isinstance(dispersion, _Number)
            and isinstance(power, _Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Tweedie, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.dispersion = self.dispersion.expand(batch_shape)
        new.power = self.power.expand(batch_shape)
        super(Tweedie, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.dispersion * torch.pow(self.mu, self.power)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.mu.dtype, device=self.mu.device)
        if self._validate_args:
            self._validate_sample(value)

        def log_prob_nonzero(y, mu, dispersion, power):
            if y.ndim > 1:
                y = torch.flatten(y)
            if mu.ndim > 1:
                mu = torch.flatten(mu)
            if dispersion.ndim > 1:
                dispersion = torch.flatten(dispersion)
            if power.ndim > 1:
                power = torch.flatten(power)

            def get_alpha(power):
                return (2 - power) / (1 - power)

            def get_jmax(y, dispersion, power):
                return torch.pow(y, 2 - power) / (dispersion * (2 - power))

            def get_log_z(y, dispersion, power):
                alpha = get_alpha(power)
                return (
                    -alpha * torch.log(y)
                    + alpha * torch.log(power - 1)
                    - torch.log(2 - power)
                    - (1 - alpha) * torch.log(dispersion)
                )

            def get_log_W(alpha, j, constant_log_W, pi):
                return (
                    j * (constant_log_W - (1 - alpha) * torch.log(j))
                    - torch.log(2 * pi)
                    - 0.5 * torch.log(-alpha)
                    - torch.log(j)
                )

            def get_log_W_max(alpha, j, pi):
                return (
                    j * (1 - alpha)
                    - torch.log(2 * pi)
                    - 0.5 * torch.log(-alpha)
                    - torch.log(j)
                )

            pi = torch.tensor(math.pi)
            alpha = get_alpha(power)
            log_z = get_log_z(y, dispersion, power)

            if torch.any(torch.isinf(log_z)):
                raise OverflowError("log(z) growing towards infinity")

            j_max = get_jmax(y, dispersion, power)
            constant_log_W = log_z + (1 - alpha) + alpha * torch.log(-alpha)
            log_W_max = get_log_W_max(alpha, j_max.round(), pi)

            j = max(torch.tensor(1), j_max.max().round())
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any((log_W_max - log_W) < 37):
                j += 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_U = j.item()

            j = max(torch.tensor(1), j_max.min().round())
            log_W = get_log_W(alpha, j, constant_log_W, pi)
            while torch.any(log_W_max - log_W < 37) and j > 1:
                j -= 1
                log_W = get_log_W(alpha, j, constant_log_W, pi)
                if torch.any(torch.isinf(log_W)):
                    break
            j_L = j.item()

            j = torch.arange(j_L, j_U + 1)
            j_2dim = torch.tile(j, (log_z.shape[0], 1)).to(torch.float32)
            log_W = (
                j_2dim * log_z[:, None]
                - torch.special.gammaln(j + 1)
                - torch.special.gammaln(-alpha[:, None] * j)
            )

            max_log_W = torch.max(log_W, dim=1).values
            sum_W = torch.exp(log_W - max_log_W[:, None]).sum(dim=1)

            return (
                max_log_W
                + torch.log(sum_W)
                - torch.log(y)
                + (
                    (
                        (y * torch.pow(mu, 1 - power) / (1 - power))
                        - torch.pow(mu, 2 - power) / (2 - power)
                    )
                    / dispersion
                )
            )

        value, mu, dispersion, power = broadcast_all(
            value, self.mu, self.dispersion, self.power
        )

        log_p = torch.full(value.shape, torch.nan)

        zeros = value == 0
        non_zeros = ~zeros

        if torch.any(zeros):
            log_p[zeros] = -(
                torch.pow(mu[zeros], 2 - power[zeros])
                / (dispersion[zeros] * (2 - power[zeros]))
            )

        if torch.any(non_zeros):
            log_p[non_zeros] = log_prob_nonzero(
                value[non_zeros], mu[non_zeros], dispersion[non_zeros], power[non_zeros]
            )

        return log_p

    @lazy_property
    def poisson_rate(self):
        return torch.pow(self.mu, 2 - self.power) / (self.dispersion * (2 - self.power))

    @lazy_property
    def gamma_concentration(self):
        return (2 - self.power) / (self.power - 1)

    @lazy_property
    def gamma_rate(self):
        return 1 / (
            self.dispersion * (self.power - 1) * torch.pow(self.mu, self.power - 1)
        )

    def sample(self, sample_shape=torch.Size()):
        rate, alpha, beta = self.poisson_rate, self.gamma_concentration, self.gamma_rate
        rate, alpha, beta = broadcast_all(rate, alpha, beta)

        with torch.no_grad():
            samples = Poisson(rate).sample(sample_shape)
            non_zeros = samples > 0

            if torch.any(non_zeros):
                alpha, beta = alpha.expand_as(samples), beta.expand_as(samples)
                samples[non_zeros] = Gamma(
                    samples[non_zeros] * alpha[non_zeros], beta[non_zeros]
                ).sample()

            return samples
