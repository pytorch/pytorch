from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Beta(Distribution):
    r"""
    Creates a Beta distribution parameterized by concentration `alpha` and `beta`.

    Example::

        >>> m = Beta(torch.Tensor([0.5]), torch.Tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration alpha and beta
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        alpha (float or Tensor or Variable): 1st concentration parameter of the distribution
        beta (float or Tensor or Variable): 2nd concentration parameter of the distribution
    """
    params = {'alpha': constraints.positive, 'beta': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, alpha, beta):
        if isinstance(alpha, Number) and isinstance(beta, Number):
            alpha_beta = torch.Tensor([alpha, beta])
        else:
            alpha, beta = broadcast_all(alpha, beta)
            alpha_beta = torch.stack([alpha, beta], -1)
        self._dirichlet = Dirichlet(alpha_beta)
        super(Beta, self).__init__(self._dirichlet._batch_shape)

    def rsample(self, sample_shape=()):
        value = self._dirichlet.rsample(sample_shape).select(-1, 0)
        if isinstance(value, Number):
            value = self._dirichlet.alpha.new([value])
        return value

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def alpha(self):
        result = self._dirichlet.alpha[..., 0]
        if isinstance(result, Number):
            return torch.Tensor([result])
        else:
            return result

    @property
    def beta(self):
        result = self._dirichlet.alpha[..., 1]
        if isinstance(result, Number):
            return torch.Tensor([result])
        else:
            return result
