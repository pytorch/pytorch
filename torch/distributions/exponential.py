from numbers import Number
import math
import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Exponential(Distribution):
    r"""
    Creates a Exponential distribution parameterized by rate `lambd`.

    Example::

        >>> m = Exponential(torch.Tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate lambd=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        lambd (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    has_rsample = True

    def __init__(self, lambd):
        self.lambd, = broadcast_all(lambd)
        batch_shape = torch.Size() if isinstance(lambd, Number) else self.lambd.size()
        super(Exponential, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.lambd.new(*shape).uniform_()
        return -u.log() / self.lambd.expand(shape)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        log = math.log if isinstance(self.lambd, Number) else torch.log
        return log(self.lambd) - self.lambd * value
