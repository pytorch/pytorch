import math
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale`.
        X~Normal(loc, scale)
        Y=exp(X)~LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of log of distribution
        scale (float or Tensor or Variable): standard deviation of log ofthe distribution
    """
    params = {'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale):
        super(HalfNormal, self).__init__(Normal(0, scale), AbsTransform())

    @property
    def scale(self):
        return self.base_dist.scale

    def log_prob(self, value):
        return math.log(2) + self.base_dist.log_prob(value)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)
