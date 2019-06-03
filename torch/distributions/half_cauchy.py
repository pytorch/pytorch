import math

from torch._six import inf
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution


class HalfCauchy(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """
    arg_constraints = {'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale, validate_args=None):
        base_dist = Cauchy(0, scale)
        super(HalfCauchy, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super(HalfCauchy, self).expand(batch_shape, _instance=new)

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob[value.expand(log_prob.shape) < 0] = -inf
        return log_prob

    def cdf(self, value):
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)
