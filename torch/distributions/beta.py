from numbers import Number

import torch
from torch.autograd import Variable
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
    has_rsample = True

    def __init__(self, alpha, beta):
        alpha, beta = broadcast_all(alpha, beta)
        alpha_beta = torch.stack([alpha, beta], -1)
        self.dirichlet = Dirichlet(alpha_beta)

    def rsample(self, sample_shape=()):
        return self.dirichlet.rsample(sample_shape).select(-1, 0)

    def log_prob(self, value):
        if isinstance(value, Number):
            heads_tails = torch.Tensor([value, 1.0 - value])
        else:
            heads_tails = torch.stack([value, 1.0 - value], -1)
        return self.dirichlet.log_prob(heads_tails)
