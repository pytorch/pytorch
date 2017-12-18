from numbers import Number

import torch
from torch.autograd import Variable, Function
from torch.autograd.function import once_differentiable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


def _standard_gamma(alpha):
    if not isinstance(alpha, Variable):
        return torch._C._standard_gamma(alpha)
    return alpha._standard_gamma()


class Gamma(Distribution):
    r"""
    Creates a Gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> m = Gamma(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # Gamma distributed with shape alpha=1 and rate beta=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        alpha (float or Tensor or Variable): shape parameter of the distribution
        beta (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    has_rsample = True

    def __init__(self, alpha, beta):
        self.alpha, self.beta = broadcast_all(alpha, beta)
        if isinstance(alpha, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.alpha.size()
        super(Gamma, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return _standard_gamma(self.alpha.expand(shape)) / self.beta.expand(shape)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return (self.alpha * torch.log(self.beta) +
                (self.alpha - 1) * torch.log(value) -
                self.beta * value - torch.lgamma(self.alpha))
