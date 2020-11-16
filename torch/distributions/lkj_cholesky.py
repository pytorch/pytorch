from torch.distributions.utils import broadcast_all


class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor propotional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices.

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    """
    arg_constraints = {'concentraion': constraints.positive}
    support = constraints.corr_cholesky

    def __init__(self, dim, concentration=1., validate_args=None):
        if dim < 2:
            raise ValueError(f'Expected dim to be an integer greater than or equal to 2. Found dim={dim}.')
        self.dim = dim
        self.concentration = broadcast_all(concentration)
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = torch.arange(self.dim-1, dtype=self.concentration.dtype, device=self.concentration.device)
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = dist.Beta(beta_conc1, beta_conc0)
        super(LKJCholesky, self).__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # There are a few differences from [1] Sec. 3.2:
        # - This vectorizes the for loop and also works for heterogeneous eta.
        # - Same algorithm generalizes to n=1.
        # - The procedure is simplified since we are sampling the cholesky factor of
        #   the correlation matrix instead of the correlation matrix itself. As such,
        #   we only need to generate `w`.
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = torch.randn(self._extended_shape(shape) + (self.dim, self.dim)).tril(diag=-1)
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.)
        w = torch.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        diag_embed = (1 - (w**2).sum(dim=-1).clamp(min=0.).sqrt()
        w.diagonal(dim1=-1).copy_(diag_embed)
        return w

    def log_prob(self, value):


    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc
