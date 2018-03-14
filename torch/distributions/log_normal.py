from torch.distributions import constraints
from torch.distributions.transforms import ExpTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    `mean` and `std` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log ofthe distribution
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale):
        super(LogNormal, self).__init__(Normal(loc, scale), ExpTransform())

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc
