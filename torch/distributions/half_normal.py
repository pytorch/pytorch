import math
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale`.
        X~Normal(0, scale)
        Y=abs(X)~HalfNormal(scale)

    Example::

        >>> m = HalfNormal(torch.Tensor([1.0]))
        >>> m.sample()  # half-normal distributed with and stddev=sqrt(1-2/pi)*scale
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        scale (float or Tensor or Variable): standard deviation/sqrt(1-2/pi) of the distribution
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
