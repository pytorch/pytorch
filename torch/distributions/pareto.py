from numbers import Number

import math

import torch
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all


class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
         1.5623
        [torch.FloatTensor of size 1]

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}

    def __init__(self, scale, alpha, validate_args=None):
        self.scale, self.alpha = broadcast_all(scale, alpha)
        base_dist = Exponential(self.alpha)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        super(Pareto, self).__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        # mean is inf for alpha <= 1
        a = self.alpha.clone().clamp(min=1)
        return a * self.scale / (a - 1)

    @property
    def variance(self):
        # var is inf for alpha <= 2
        a = self.alpha.clone().clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    @constraints.dependent_property
    def support(self):
        return constraints.greater_than(self.scale)

    def entropy(self):
        return ((self.scale / self.alpha).log() + (1 + self.alpha.reciprocal()))
