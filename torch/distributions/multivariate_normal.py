import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


def _get_batch_shape(bmat, bvec):
    r"""
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    return torch.Size(vec_shape[:-1])


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    n = bvec.size(-1)
    batch_shape = _get_batch_shape(bmat, bvec)

    # to conform with `torch.bmm` interface, both bmat and bvec should have `.dim() == 3`
    bmat = bmat.expand(batch_shape + (n, n)).reshape((-1, n, n))
    bvec = bvec.unsqueeze(-1).expand(batch_shape + (n, 1)).reshape((-1, n, 1))
    return torch.bmm(bmat, bvec).view(batch_shape + (n,))


def _batch_potrf_lower(bmat):
    r"""
    Applies a Cholesky decomposition to all matrices in a batch of arbitrary shape.
    """
    n = bmat.size(-1)
    cholesky = torch.stack([C.potrf(upper=False) for C in bmat.reshape((-1, n, n))])
    return cholesky.view(bmat.shape)


def _batch_diag(bmat):
    r"""
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.reshape(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


def _batch_inverse(bmat):
    r"""
    Returns the inverses of a batch of square matrices.
    """
    n = bmat.size(-1)
    flat_bmat = bmat.reshape(-1, n, n)
    flat_inv_bmat = torch.stack([m.inverse() for m in flat_bmat], 0)
    return flat_inv_bmat.view(bmat.shape)


def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).reshape((-1,) + L.shape[-2:])
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precition matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor): mean of the distribution
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
    """
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        event_shape = torch.Size(loc.shape[-1:])
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril = scale_tril
            batch_shape = _get_batch_shape(scale_tril, loc)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix = covariance_matrix
            batch_shape = _get_batch_shape(covariance_matrix, loc)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix = precision_matrix
            self.covariance_matrix = _batch_inverse(precision_matrix)
            batch_shape = _get_batch_shape(precision_matrix, loc)
        self.loc = loc
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def scale_tril(self):
        return _batch_potrf_lower(self.covariance_matrix)

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(self.scale_tril, self.scale_tril.transpose(-1, -2))

    @lazy_property
    def precision_matrix(self):
        # TODO: use `torch.potri` on `scale_tril` once a backwards pass is implemented.
        scale_tril_inv = _batch_inverse(self.scale_tril)
        return torch.matmul(scale_tril_inv.transpose(-1, -2), scale_tril_inv)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        n = self.covariance_matrix.size(-1)
        var = torch.stack([cov.diag() for cov in self.covariance_matrix.view(-1, n, n)])
        return var.view(self.covariance_matrix.size()[:-1])

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self.scale_tril, diff)
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        return -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det

    def entropy(self):
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        H = 0.5 * (1.0 + math.log(2 * math.pi)) * self._event_shape[0] + log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
