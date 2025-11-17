# mypy: allow-untyped-defs
import math
import warnings
from typing import Optional, Union

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.distributions.utils import lazy_property
from torch.types import _size, Number


__all__ = ["Wishart"]

_log_2 = math.log(2)


def _mvdigamma(x: Tensor, p: int) -> Tensor:
    assert x.gt((p - 1) / 2).all(), "Wrong domain for multivariate digamma function."
    return torch.digamma(
        x.unsqueeze(-1)
        - torch.arange(p, dtype=x.dtype, device=x.device).div(2).expand(x.shape + (-1,))
    ).sum(-1)


def _clamp_above_eps(x: Tensor) -> Tensor:
    # We assume positive input for this function
    return x.clamp(min=torch.finfo(x.dtype).eps)


class Wishart(ExponentialFamily):
    r"""
    Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\Sigma`,
    or its Cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`

    Example:
        >>> # xdoctest: +SKIP("FIXME: scale_tril must be at least two-dimensional")
        >>> m = Wishart(torch.Tensor([2]), covariance_matrix=torch.eye(2))
        >>> m.sample()  # Wishart distributed with mean=`df * I` and
        >>> # variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j

    Args:
        df (float or Tensor): real-valued parameter larger than the (dimension of Square matrix) - 1
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
        'torch.distributions.LKJCholesky' is a restricted Wishart distribution.[1]

    **References**

    [1] Wang, Z., Wu, Y. and Chu, H., 2018. `On equivalence of the LKJ distribution and the restricted Wishart distribution`.
    [2] Sawyer, S., 2007. `Wishart Distributions and Inverse-Wishart Sampling`.
    [3] Anderson, T. W., 2003. `An Introduction to Multivariate Statistical Analysis (3rd ed.)`.
    [4] Odell, P. L. & Feiveson, A. H., 1966. `A Numerical Procedure to Generate a SampleCovariance Matrix`. JASA, 61(313):199-203.
    [5] Ku, Y.-C. & Bloomfield, P., 2010. `Generating Random Wishart Matrices with Fractional Degrees of Freedom in OX`.
    """

    support = constraints.positive_definite
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def arg_constraints(self):
        return {
            "covariance_matrix": constraints.positive_definite,
            "precision_matrix": constraints.positive_definite,
            "scale_tril": constraints.lower_cholesky,
            "df": constraints.greater_than(self.event_shape[-1] - 1),
        }

    def __init__(
        self,
        df: Union[Tensor, Number],
        covariance_matrix: Optional[Tensor] = None,
        precision_matrix: Optional[Tensor] = None,
        scale_tril: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        assert (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) == 1, (
            "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
        )

        param = next(
            p
            for p in (covariance_matrix, precision_matrix, scale_tril)
            if p is not None
        )

        if param.dim() < 2:
            raise ValueError(
                "scale_tril must be at least two-dimensional, with optional leading batch dimensions"
            )

        if isinstance(df, Number):
            batch_shape = torch.Size(param.shape[:-2])
            self.df = torch.tensor(df, dtype=param.dtype, device=param.device)
        else:
            batch_shape = torch.broadcast_shapes(param.shape[:-2], df.shape)
            self.df = df.expand(batch_shape)
        event_shape = param.shape[-2:]

        if self.df.le(event_shape[-1] - 1).any():
            raise ValueError(
                f"Value of df={df} expected to be greater than ndim - 1 = {event_shape[-1] - 1}."
            )

        if scale_tril is not None:
            # pyrefly: ignore [read-only]
            self.scale_tril = param.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            # pyrefly: ignore [read-only]
            self.covariance_matrix = param.expand(batch_shape + (-1, -1))
        elif precision_matrix is not None:
            # pyrefly: ignore [read-only]
            self.precision_matrix = param.expand(batch_shape + (-1, -1))

        if self.df.lt(event_shape[-1]).any():
            warnings.warn(
                "Low df values detected. Singular samples are highly likely to occur for ndim - 1 < df < ndim.",
                stacklevel=2,
            )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self._batch_dims = [-(x + 1) for x in range(len(self._batch_shape))]

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

        # Chi2 distribution is needed for Bartlett decomposition sampling
        self._dist_chi2 = torch.distributions.chi2.Chi2(
            df=(
                self.df.unsqueeze(-1)
                - torch.arange(
                    self._event_shape[-1],
                    dtype=self._unbroadcasted_scale_tril.dtype,
                    device=self._unbroadcasted_scale_tril.device,
                ).expand(batch_shape + (-1,))
            )
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril.expand(cov_shape)
        new.df = self.df.expand(batch_shape)

        new._batch_dims = [-(x + 1) for x in range(len(batch_shape))]

        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)

        # Chi2 distribution is needed for Bartlett decomposition sampling
        new._dist_chi2 = torch.distributions.chi2.Chi2(
            df=(
                new.df.unsqueeze(-1)
                - torch.arange(
                    self.event_shape[-1],
                    dtype=new._unbroadcasted_scale_tril.dtype,
                    device=new._unbroadcasted_scale_tril.device,
                ).expand(batch_shape + (-1,))
            )
        )

        super(Wishart, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self) -> Tensor:
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        return (
            self._unbroadcasted_scale_tril
            @ self._unbroadcasted_scale_tril.transpose(-2, -1)
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self) -> Tensor:
        identity = torch.eye(
            self._event_shape[-1],
            device=self._unbroadcasted_scale_tril.device,
            dtype=self._unbroadcasted_scale_tril.dtype,
        )
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape
        )

    @property
    def mean(self) -> Tensor:
        return self.df.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def mode(self) -> Tensor:
        factor = self.df - self.covariance_matrix.shape[-1] - 1
        factor[factor <= 0] = nan
        return factor.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def variance(self) -> Tensor:
        V = self.covariance_matrix  # has shape (batch_shape x event_shape)
        diag_V = V.diagonal(dim1=-2, dim2=-1)
        return self.df.view(self._batch_shape + (1, 1)) * (
            V.pow(2) + torch.einsum("...i,...j->...ij", diag_V, diag_V)
        )

    def _bartlett_sampling(self, sample_shape=torch.Size()):
        p = self._event_shape[-1]  # has singleton shape

        # Implemented Sampling using Bartlett decomposition
        noise = _clamp_above_eps(
            self._dist_chi2.rsample(sample_shape).sqrt()
        ).diag_embed(dim1=-2, dim2=-1)

        i, j = torch.tril_indices(p, p, offset=-1)
        noise[..., i, j] = torch.randn(
            torch.Size(sample_shape) + self._batch_shape + (int(p * (p - 1) / 2),),
            dtype=noise.dtype,
            device=noise.device,
        )
        chol = self._unbroadcasted_scale_tril @ noise
        return chol @ chol.transpose(-2, -1)

    def rsample(
        self, sample_shape: _size = torch.Size(), max_try_correction=None
    ) -> Tensor:
        r"""
        .. warning::
            In some cases, sampling algorithm based on Bartlett decomposition may return singular matrix samples.
            Several tries to correct singular samples are performed by default, but it may end up returning
            singular matrix samples. Singular samples may return `-inf` values in `.log_prob()`.
            In those cases, the user should validate the samples and either fix the value of `df`
            or adjust `max_try_correction` value for argument in `.rsample` accordingly.
        """

        if max_try_correction is None:
            max_try_correction = 3 if torch._C._get_tracing_state() else 10

        sample_shape = torch.Size(sample_shape)
        sample = self._bartlett_sampling(sample_shape)

        # Below part is to improve numerical stability temporally and should be removed in the future
        is_singular = self.support.check(sample)
        if self._batch_shape:
            is_singular = is_singular.amax(self._batch_dims)

        if torch._C._get_tracing_state():
            # Less optimized version for JIT
            for _ in range(max_try_correction):
                sample_new = self._bartlett_sampling(sample_shape)
                sample = torch.where(is_singular, sample_new, sample)

                is_singular = ~self.support.check(sample)
                if self._batch_shape:
                    is_singular = is_singular.amax(self._batch_dims)

        else:
            # More optimized version with data-dependent control flow.
            if is_singular.any():
                warnings.warn("Singular sample detected.", stacklevel=2)

                for _ in range(max_try_correction):
                    sample_new = self._bartlett_sampling(is_singular[is_singular].shape)
                    sample[is_singular] = sample_new

                    is_singular_new = ~self.support.check(sample_new)
                    if self._batch_shape:
                        is_singular_new = is_singular_new.amax(self._batch_dims)
                    is_singular[is_singular.clone()] = is_singular_new

                    if not is_singular.any():
                        break

        return sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return (
            -nu
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            - torch.mvlgamma(nu / 2, p=p)
            + (nu - p - 1) / 2 * torch.linalg.slogdet(value).logabsdet
            - torch.cholesky_solve(value, self._unbroadcasted_scale_tril)
            .diagonal(dim1=-2, dim2=-1)
            .sum(dim=-1)
            / 2
        )

    def entropy(self):
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return (
            (p + 1)
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            + torch.mvlgamma(nu / 2, p=p)
            - (nu - p - 1) / 2 * _mvdigamma(nu / 2, p=p)
            + nu * p / 2
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return -self.precision_matrix / 2, (nu - p - 1) / 2

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        p = self._event_shape[-1]
        return (y + (p + 1) / 2) * (
            -torch.linalg.slogdet(-2 * x).logabsdet + _log_2 * p
        ) + torch.mvlgamma(y + (p + 1) / 2, p=p)
