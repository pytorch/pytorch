import math

from torch._six import inf
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    """
    arg_constraints = {'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, scale, validate_args=None):
        base_dist = Normal(0, scale, validate_args=False)
        super(HalfNormal, self).__init__(base_dist, AbsTransform(),
                                         validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfNormal, _instance)
        return super(HalfNormal, self).expand(batch_shape, _instance=new)

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob[value.expand(log_prob.shape) < 0] = -inf
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)
