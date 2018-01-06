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
        >>> m.sample()  # Beta distributed with concentrarion alpha
         0.1046
        [torch.FloatTensor of size 2]

    Args:
        alpha (Tensor or Variable): concentration parameter of the distribution
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
