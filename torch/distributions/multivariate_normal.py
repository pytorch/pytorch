import math

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
    return vec_shape[:-1]


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_potrf_lower(bmat):
    r"""
    Applies a Cholesky decomposition to all matrices in a batch of arbitrary shape.
    """
    n = bmat.size(-1)
    cholesky = torch.stack([m.potrf(upper=False) for m in bmat.reshape(-1, n, n)])
    return cholesky.reshape(bmat.shape)


def _batch_diag(bmat):
    r"""
    Returns the diagonals of a batch of square matrices.
    """
    return torch.diagonal(bmat, dim1=-2, dim2=-1)


def _batch_inverse(bmat):
    r"""
    Returns the inverses of a batch of square matrices.
    """
    n = bmat.size(-1)
    flat_bmat_inv = torch.stack([m.inverse() for m in bmat.reshape(-1, n, n)])
    return flat_bmat_inv.reshape(bmat.shape)


def _batch_trtrs_lower(bb, bA):
    """
    Applies `torch.trtrs` for batches of matrices. `bb` and `bA` should have
    the same batch shape.
    """
    flat_b = bb.reshape((-1,) + bb.shape[-2:])
    flat_A = bA.reshape((-1,) + bA.shape[-2:])
    flat_X = torch.stack([torch.trtrs(b, A, upper=False)[0] for b, A in zip(flat_b, flat_A)])
    return flat_X.reshape(bb.shape)


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bL = bL.expand(bx.shape[bx.dim() - bL.dim() + 1:] + (n,))
    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = _batch_trtrs_lower(flat_x_swap, flat_L).pow(2).sum(-2)  # shape = b x c
    return M_swap.t().reshape(bx.shape[:-1])


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

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
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self._unbroadcasted_scale_tril = scale_tril
            batch_shape = _get_batch_shape(scale_tril, loc)
            self.scale_tril = scale_tril.expand(batch_shape + event_shape + event_shape)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self._unbroadcasted_scale_tril = _batch_potrf_lower(covariance_matrix)
            batch_shape = _get_batch_shape(covariance_matrix, loc)
            self.covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            covariance_matrix = _batch_inverse(precision_matrix)
            self._unbroadcasted_scale_tril = _batch_potrf_lower(covariance_matrix)
            batch_shape = _get_batch_shape(precision_matrix, loc)
            self.precision_matrix = precision_matrix.expand(batch_shape + event_shape + event_shape)
            self.covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)

        self.loc = loc.expand(batch_shape + event_shape)
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return (torch.matmul(self._unbroadcasted_scale_tril,
                             self._unbroadcasted_scale_tril.transpose(-1, -2))
                .expand(self._batch_shape + self._event_shape + self._event_shape))

    @lazy_property
    def precision_matrix(self):
        # TODO: use `torch.potri` on `scale_tril` once a backwards pass is implemented.
        scale_tril_inv = _batch_inverse(self._unbroadcasted_scale_tril)
        return torch.matmul(scale_tril_inv.transpose(-1, -2), scale_tril_inv).expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(
            self._batch_shape + self._event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new_empty(shape).normal_()
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = _batch_diag(self._unbroadcasted_scale_tril).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = _batch_diag(self._unbroadcasted_scale_tril).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
