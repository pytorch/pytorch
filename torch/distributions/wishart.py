import math
import warnings

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property
from torch.distributions.multivariate_normal import _precision_to_scale_tril


_log_2 = math.log(2)


def _mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    assert x.gt((p - 1) / 2).all(), "Wrong domain for multivariate digamma function."
    return torch.digamma(x.unsqueeze(-1) - torch.arange(p).div(2).expand(x.shape + (-1,))).sum(-1)

class Wishart(ExponentialFamily):
    r"""
    Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\Sigma`,
    or its Cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`
    The Wishart distribution can be parameterized either in terms of
    an outer product of general square root matrix
    :math:`\mathbf{\Sigma} = \mathbf{P}\mathbf{D}\mathbf{P}^\top = \mathbf{P'}\mathbf{P'}^\top`
    can be obtained via Cholesky decomposition of the covariance.
    Example:
        >>> m = Wishart(torch.eye(2), torch.Tensor([2]))
        >>> m.sample()  #Wishart distributed with mean=`df * I` and
                        #variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j
    Args:
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
        df (float or Tensor): real-valued parameter larger than the (dimension of Square matrix) - 1
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
        'torch.distributions.LKJCholesky' is a restricted Wishart distribution.[1]

    **References**

    [1] `On equivalence of the LKJ distribution and the restricted Wishart distribution`,
    Zhenxun Wang, Yunan Wu, Haitao Chu.
    """
    arg_constraints = {
        'covariance_matrix': constraints.positive_definite,
        'precision_matrix': constraints.positive_definite,
        'scale_tril': constraints.lower_cholesky,
        'df': constraints.greater_than(0),
    }
    support = constraints.positive_definite
    has_rsample = True

    def __init__(self,
                 df: torch.Tensor,
                 covariance_matrix: torch.Tensor = None,
                 precision_matrix: torch.Tensor = None,
                 scale_tril: torch.Tensor = None,
                 validate_args=None):
        assert (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) == 1, \
            "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."

        if scale_tril is not None:
            assert scale_tril.dim() > 1, \
                "scale_tril must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], df.shape)
            event_shape = scale_tril.shape[-2:]
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            assert covariance_matrix.dim() > 1, \
                "covariance_matrix must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], df.shape)
            event_shape = covariance_matrix.shape[-2:]
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        elif precision_matrix is not None:
            assert precision_matrix.dim() > 1, \
                "precision_matrix must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = torch.broadcast_shapes(precision_matrix.shape[:-2], df.shape)
            event_shape = precision_matrix.shape[-2:]
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))

        self.df = df.expand(batch_shape)
        self.arg_constraints['df'] = constraints.greater_than(event_shape[-1] - 1)
        if self.df.lt(event_shape[-1]).any():
            warnings.warn("Low df values detected. Singular samples are highly likely occured for ndim - 1 < df < ndim.")

        super(Wishart, self).__init__(batch_shape, event_shape, validate_args=validate_args)

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
        df_shape = batch_shape
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril.expand(cov_shape)
        new.df = self.df.expand(df_shape)

        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
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
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return (
            self._unbroadcasted_scale_tril @ self._unbroadcasted_scale_tril.transpose(-2, -1)
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self._event_shape[-1],
            device=self._unbroadcasted_scale_tril.device,
            dtype=self._unbroadcasted_scale_tril.dtype,
        )
        return torch.cholesky_solve(
            identity, self._unbroadcasted_scale_tril
        ).expand(self._batch_shape + self._event_shape)

    @property
    def mean(self):
        return self.df.view(self._batch_shape + (1, 1,)) * self.covariance_matrix

    @property
    def variance(self):
        V = self.covariance_matrix  # has shape (batch_shape x event_shape)
        diag_V = V.diagonal(dim1=-2, dim2=-1)
        return self.df.view(self._batch_shape + (1, 1,)) * (V.pow(2) + torch.einsum("...i,...j->...ij", diag_V, diag_V))

    def _barttlet_sampling(self, sample_shape=torch.Size()):
        p = self._event_shape[-1]  # has singleton shape

        # Implemented Sampling using Bartlett decomposition
        noise = self._dist_chi2.rsample(sample_shape).sqrt().diag_embed(dim1=-2, dim2=-1)
        i, j = torch.tril_indices(p, p, offset=-1)
        noise[..., i, j] = torch.randn(
            torch.Size(sample_shape) + self._batch_shape + (p * (p - 1) // 2,),
            dtype=noise.dtype,
            device=noise.device,
        )
        chol = self._unbroadcasted_scale_tril @ noise
        return chol @ chol.transpose(-2, -1)

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        sample = self._barttlet_sampling(sample_shape)

        # Below part is to improve numerical stability temporally and should be removed in the future
        is_singular = self.support.check(sample).logical_not()
        if len(self._batch_shape):
            reduced_batch_dims = [-(x + 1) for x in range(len(self._batch_shape))]
            is_singular = is_singular.amax(reduced_batch_dims)

        if is_singular.any():
            warnings.warn("Singular sample detected.")
            sample = sample[is_singular.logical_not()]

            while is_singular.any():
                new_sample = self._barttlet_sampling(is_singular[is_singular].shape)

                is_singular = self.support.check(new_sample).logical_not()
                if len(self._batch_shape):
                    is_singular = is_singular.amax(reduced_batch_dims)

                sample = torch.cat(
                    (sample, new_sample[is_singular.logical_not()]),
                    dim=-(len(self._batch_shape) + 3)
                )

        return sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return (
            - nu * p * _log_2 / 2
            - nu * self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - torch.mvlgamma(nu / 2, p=p)
            + (nu - p - 1) / 2 * torch.linalg.slogdet(value).logabsdet
            - torch.cholesky_solve(value, self._unbroadcasted_scale_tril).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 2
        )

    def entropy(self):
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        V = self.covariance_matrix  # has shape (batch_shape x event_shape)
        return (
            (p + 1) * self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + p * (p + 1) * _log_2 / 2
            + torch.mvlgamma(nu / 2, p=p)
            - _mvdigamma(nu / 2, p=p) * (nu - p - 1) / 2
            + nu * p / 2
        )

    @property
    def _natural_params(self):
        return (
            0.5 * self.df,
            - 0.5 * torch.cholesky_inverse(self._unbroadcasted_scale_tril)
        )

    def _log_normalizer(self, x, y):
        p = y.shape[-1]
        return x * (- torch.linalg.slogdet(-2 * y) + _log_2 * p) + _mvdigamma(x, p=p)
