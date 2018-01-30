import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property


def _get_batch_shape(bmat, bvec):
    """
    Given a batch of matrices and a batch of vectors, compute the combined `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    return torch.Size(vec_shape[:-1])


def _batch_mv(bmat, bvec):
    """
    Performs a batched matrix-vector product, with an arbitrary batch shape.
    """
    batch_shape = bvec.shape[:-1]
    event_dim = bvec.shape[-1]
    bmat = bmat.expand(batch_shape + (event_dim, event_dim))
    if batch_shape != bmat.shape[:-2]:
        raise ValueError("Batch shapes do not match: matrix {}, vector {}".format(bmat.shape, bvec.shape))
    bvec = bvec.unsqueeze(-1)
    
    # conform with `torch.bmm` interface, for matrices with `.dim() == 3`
    if bvec.dim() == 2:
        bvec.unsqueeze(0) #_
     # flatten batch dimensions
    bvec = bvec.contiguous().view((-1, event_dim, 1))
    bmat = bmat.contiguous().view((-1, event_dim, event_dim)).expand((bvec.shape[0], -1, -1))
    return torch.bmm(bmat, bvec).squeeze(-1).view(batch_shape+(event_dim,)) 


def _batch_potrf_lower(bmat):
    """
    Applies a Cholesky decomposition to all matrices in a batch of arbitrary shape.
    """
    n = bmat.size(-1)
    cholesky = torch.stack([C.potrf(upper=False) for C in bmat.unsqueeze(0).contiguous().view((-1,n,n))])
    return cholesky.view(bmat.shape)


def _batch_diag(bmat):
    """
    Returns the diagonals of a batch of square matrices.
    """
    n = bmat.size(-1)
    dims = torch.arange(n, out=bmat.new(n)).long()
    if isinstance(dims, Variable):
        dims = dims.data # TODO: why can't I index with a Variable?
    return bmat[...,dims,dims]


def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}` 
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`. 
    
    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).contiguous().view((-1,)+L.shape[-2:])
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.
    
    The multivariate normal distribution can be parameterized either 
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}` 
    or a lower-triangular matrix :math:`\mathbf{L}` such that 
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top` as obtained via e.g. 
    Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): positive-definite covariance matrix
        scale_tril (Tensor or Variable): lower-triangular factor of covariance
        
    Note:
        Only one of `covariance_matrix` or `scale_tril` can be specified.
        
    """
    params = {'loc': constraints.real_vector,
              'covariance_matrix': constraints.positive_definite,
              'scale_tril': constraints.lower_cholesky }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        event_shape = torch.Size(loc.shape[-1:])
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified (but not both).")
        if scale_tril is None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be two-dimensional")
            self.covariance_matrix = covariance_matrix
            batch_shape = _get_batch_shape(covariance_matrix, loc)
        else:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be two-dimensional")
            self.scale_tril = scale_tril
            batch_shape = _get_batch_shape(scale_tril, loc)
        self.loc = loc
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @lazy_property
    def scale_tril(self):
        return _batch_potrf_lower(self.covariance_matrix)

    @lazy_property
    def covariance_matrix(self):
        # To use torch.bmm, we first squash the batch_shape into a single dimension
        flat_scale_tril = self.scale_tril.unsqueeze(0).contiguous().view((-1,)+self._event_shape*2)
        return torch.bmm(flat_scale_tril, flat_scale_tril.transpose(-1,-2)).view(self.scale_tril.shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        delta = value - self.loc
        M = _batch_mahalanobis(self.scale_tril, delta)
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        return -0.5*(M + self.loc.size(-1)*math.log(2*math.pi)) - log_det

    def entropy(self):
        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        H = 0.5*(1.0 + math.log(2*math.pi))*self._event_shape[0] + log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
