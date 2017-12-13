from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.distribution import Distribution


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
        alpha_num = isinstance(alpha, Number)
        beta_num = isinstance(beta, Number)
        if alpha_num and beta_num:
            alpha_beta = torch.Tensor([alpha, beta])
        else:
            if alpha_num and not beta_num:
                alpha = beta.new(beta.size()).fill_(alpha)
            elif not alpha_num and beta_num:
                beta = alpha.new(alpha.size()).fill_(beta)
            elif alpha.size() != beta.size():
                raise ValueError('Expected alpha.size() == beta.size(), actual {} vs {}'.format(
                    alpha.size(), beta.size()))
            alpha_beta = torch.stack([alpha, beta], -1)
        self.dirichlet = Dirichlet(alpha_beta)

    def sample(self):
        return self.dirichlet.sample().select(-1, 0)

    def sample_n(self, n):
        return self.dirichlet.sample_n(n).select(-1, 0)

    def log_prob(self, value):
        if isinstance(value, Number):
            heads_tails = torch.Tensor([value, 1.0 - value])
        else:
            heads_tails = torch.stack([value, 1.0 - value], -1)
        return self.dirichlet.log_prob(heads_tails)
