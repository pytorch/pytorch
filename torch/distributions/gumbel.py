from numbers import Number
import math
import torch
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import _finfo, broadcast_all

euler_constant = 0.57721566490153286060  # Euler Mascheroni Constant


class Gumbel(TransformedDistribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
         1.0124
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = _finfo(self.loc)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps)
        else:
            batch_shape = self.scale.size()
            base_dist = Uniform(self.loc.new(self.loc.size()).fill_(finfo.tiny), 1 - finfo.eps)
        transforms = [ExpTransform().inv, AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
                      ExpTransform().inv, AffineTransform(loc=loc, scale=-self.scale)]
        super(Gumbel, self).__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc + self.scale * euler_constant

    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
