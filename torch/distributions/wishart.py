import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def _mvdigamma(x, p) -> torch.Tensor:
    assert x > (p - 1) / 2, "Wrong domain for digamma function."
    return torch.digamma(x - torch.arange(p) / 2).sum()

class Wishart(Distribution):
    r"""
    Creates a Wishart distribution parameterized by a square root matrix of symmetric matrix.
    The Wishart distribution can be parameterized either in terms of
    an outer product of general square root matrix e,g.,
    :math:`\mathbf{\Sigma} = \mathbf{P}\mathbf{D}\mathbf{P}^\top = \mathbf{P'}\mathbf{P'}^\top` or
    an outer product of cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`
    can be obtained via Cholesky decomposition of the covariance.  
    Example:
        >>> m = Wishart(torch.eye(2), torch.Tensor([2]))
        >>> m.sample()  #Wishart distributed with mean=`df * I` and
                        #variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j
    Args:
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
        df (Tensor): real-valued parameter > larger than dimension of Square matrix
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
        'scale_tril': constraints.positive_definite,
        'df': None,
    }
    support = constraints.positive_definite
    has_rsample = True

    def __init__(self, covariance_matrix=None, precision_matrix=None, scale_tril=None, df=None,
                 bartlett_decomposition=True, validate_args=None):

        assert (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) == 1, \
            "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."

        if scale_tril is not None:
            assert scale_tril.dim() > 1, \
                "scale_tril matrix must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = scale_tril.shape[:-2]
            event_shape = scale_tril.shape[-2:]
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
            self.df = self.scale_tril.shape[-1] if df is None else df
        elif covariance_matrix is not None:
            assert covariance_matrix.dim() > 1, \
                "covariance_matrix must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = covariance_matrix.shape[:-2]
            event_shape = covariance_matrix.shape[-2:]
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
            self.df = self.covariance_matrix.shape[-1] if df is None else df
        else:
            assert precision_matrix.dim() > 1, \
                "precision_matrix must be at least two-dimensional, with optional leading batch dimensions"
            batch_shape = precision_matrix.shape[:-2]
            event_shape = precision_matrix.shape[-2:]
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
            self.df = self.precision_matrix.shape[-1] if df is None else df

        assert self.df > event_shape[-1] - 1, \
            "Degree of Freedom paramter should be larger than the dimension - 1"

        self.arg_constraints['df'] = constraints.greater_than(event_shape[-1] - 1)
        self.bartlett_decomposition = bartlett_decomposition

        super(Wishart, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

        self.dist_gamma = torch.distributions.Gamma(
            self.df - torch.arange(self._event_shape[0], device=self._unbroadcasted_scale_tril.device) / 2,
            1 / 2,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(Wishart, new).__init__(batch_shape,
                                     self.event_shape,
                                     validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return torch.einsum(
            "ik,jk->ij",
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self._event_shape[0],
            device=self._unbroadcasted_scale_tril.device,
            dtype=self._unbroadcasted_scale_tril.dtype,
        )
        return torch.cholesky_solve(
            identity, self._unbroadcasted_scale_tril
        ).expand(self._batch_shape + self._event_shape)

    @property
    def mean(self):
        return self.df * self.covariance_matrix

    @property
    def variance(self):
        V = self.covariance_matrix
        diag_V = V.diag()
        return self.df * (V.pow(2) + torch.einsum("i,j->ij", diag_V, diag_V))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)   

        # Implemented Bartlett decomposition
        if self.bartlett_decomposition:
            noise = self.dist_gamma.sample(sample_shape).diag_embed(dim1=-2, dim2=-1).sqrt()
            noise = noise + torch.randn(shape, device=noise.device).tril(diagonal=-1)
        else:
            noise = _standard_normal(
                shape,
                dtype=self._unbroadcasted_scale_tril.dtype,
                device=self._unbroadcasted_scale_tril.device,
            )
        chol = torch.einsum("ik,...kj->...ij", self._unbroadcasted_scale_tril, noise)
        return torch.einsum("...ik,...jk->...ij", chol, chol)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        nu = self.df
        p = self._event_shape[0]
        V = self.covariance_matrix
        return (
            - torch.mvlgamma(nu / 2, p=p)
            - nu * self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - (
                - (nu - p - 1) * value.logdet()
                + nu * p * math.log(2)
                + torch.einsum("ik,...kj->...ij", V.inverse(), value).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            ) / 2
        )

    def entropy(self):
        p = self._event_shape[0]
        V = self.covariance_matrix
        H = (
            (p + 1) * self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + p * (p + 1) * math.log(2) / 2
            + torch.mvlgamma(nu / 2, p=p)
            - (nu - p - 1) * _mvdigamma(nu / 2) / 2
            + nu * p / 2
        )
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
