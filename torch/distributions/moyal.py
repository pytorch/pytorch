from numbers import Number
import math
import torch
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import (AffineTransform, ErfTransform,
                                            ExpTransform, PowerTransform)
from torch.distributions.utils import broadcast_all
from torch.distributions.gumbel import euler_constant


class Moyal(TransformedDistribution):
    r"""
    Samples from a Moyal Distribution.

    Examples::

        >>> m = Moyal(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Moyal distribution with loc=1, scale=2
        tensor([1.2672])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, Number) and isinstance(scale, Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps)
        else:
            base_dist = Uniform(torch.full_like(self.loc, finfo.tiny),
                                torch.full_like(self.loc, 1 - finfo.eps))
        transforms = [AffineTransform(loc=1., scale=-torch.ones_like(self.scale)), ErfTransform().inv,
                      PowerTransform(exponent=2. * torch.ones_like(self.scale)),
                      AffineTransform(0, scale=2. * torch.ones_like(self.scale)),
                      ExpTransform().inv, AffineTransform(loc=self.loc, scale=-self.scale)]
        super(Moyal, self).__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Moyal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super(Moyal, self).expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return self.loc + self.scale * (euler_constant + math.log(2))

    @property
    def stddev(self):
        return (math.pi / math.sqrt(2)) * self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def entropy(self):
        return 0.5 * (1 + 2 * self.scale.log() + euler_constant +
                      math.log(4 * math.pi))
