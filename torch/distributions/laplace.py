from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _finfo, broadcast_all


class Laplace(Distribution):
    r"""
    Creates a Laplace distribution parameterized by `loc` and 'scale'.

    Example::

        >>> m = Laplace(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of the distribution
        scale (float or Tensor or Variable): scale of the distribution
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return 2 * self.scale.pow(2)

    @property
    def stddev(self):
        return (2 ** 0.5) * self.scale

    def __init__(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Laplace, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new(shape).uniform_(_finfo(self.loc).eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

    def entropy(self):
        return 1 + torch.log(2 * self.scale)
