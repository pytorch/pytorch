from numbers import Number
import math
import torch
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all
from torch.distributions.gumbel import euler_constant


class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
    """
    arg_constraints = {'scale': constraints.positive, 'concentration': constraints.positive}
    support = constraints.positive

    def __init__(self, scale, concentration, validate_args=None):
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(self.scale.new(self.scale.size()).fill_(1.0))
        transforms = [PowerTransform(exponent=self.concentration_reciprocal),
                      AffineTransform(loc=0, scale=self.scale)]
        super(Weibull, self).__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    @property
    def variance(self):
        return self.scale.pow(2) * (torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal)) -
                                    torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal)))

    def entropy(self):
        return euler_constant * (1 - self.concentration_reciprocal) + \
            torch.log(self.scale * self.concentration_reciprocal) + 1
