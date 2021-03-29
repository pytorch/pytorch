"""
This closely follows the implementation in NumPyro (https://github.com/pyro-ppl/numpyro).

Original copyright notice:

# Copyright: Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""

import math

import torch
from torch.distributions import constraints, Beta
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor propotional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices. Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Example::

        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    """
    arg_constraints = {'concentration': constraints.positive}
    support = constraints.corr_cholesky

    def __init__(self, dim, concentration=1., validate_args=None):
        if dim < 2:
            raise ValueError(f'Expected dim to be an integer greater than or equal to 2. Found dim={dim}.')
        self.dim = dim
        self.concentration, = broadcast_all(concentration)
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = torch.arange(self.dim - 1, dtype=self.concentration.dtype, device=self.concentration.device)
        offset = torch.cat([offset.new_zeros((1,)), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        super(LKJCholesky, self).__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim
        new.concentration = self.concentration.expand(batch_shape)
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        super(LKJCholesky, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # This uses the Onion method, but there are a few differences from [1] Sec. 3.2:
        # - This vectorizes the for loop and also works for heterogeneous eta.
        # - Same algorithm generalizes to n=1.
        # - The procedure is simplified since we are sampling the cholesky factor of
        #   the correlation matrix instead of the correlation matrix itself. As such,
        #   we only need to generate `w`.
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = torch.randn(self._extended_shape(sample_shape),
                               dtype=y.dtype,
                               device=y.device).tril(-1)
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.)
        w = torch.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        eps = torch.finfo(w.dtype).tiny
        diag_elems = torch.clamp(1 - torch.sum(w**2, dim=-1), min=eps).sqrt()
        w += torch.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        # See: https://mc-stan.org/docs/2_25/functions-reference/cholesky-lkj-correlation-distribution.html
        # The probability of a correlation matrix is proportional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # Additionally, the Jacobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) = prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i
        if self._validate_args:
            self._validate_sample(value)
        diag_elems = value.diagonal(dim1=-1, dim2=-2)[..., 1:]
        order = torch.arange(2, self.dim + 1)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = torch.sum(order * diag_elems.log(), dim=-1)
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = torch.lgamma(alpha) * dm1
        numerator = torch.mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term
