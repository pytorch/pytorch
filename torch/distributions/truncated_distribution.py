import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class TruncatedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies truncation to base distribution.
    """
    def __init__(self, base_distribution, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super(TruncatedDistribution, self).__init__(*args, **kwargs)
        self.base_dist = base_distribution
        self.params = self.base_dist.params
        self.lower_bound, self.upper_bound, _ = broadcast_all(lower_bound, upper_bound,
                                                              getattr(self.base_dist,
                                                                      list(self.self.base_dist.params.keys)[0]))
        self.params['lower_bound': constraints.dependent, 'upper_bound': constraints.dependent]

    @constraints.dependent_property
    def support(self):
        # Note: The proper way to implement this is intersection([lower_bound, upper_bound], base_dist.support)
        # This requires intersection method to be implemented for constraints.
        return constraints.interval(self.lower_bound, self.upper_bound)

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched via inverse cdf sampling. Note that this
        is a generic sampler which is not the most efficient or accurate around tails of base distribution.
        """
        shape = shape = self._extended_shape(sample_shape)
        u = getattr(self.base_dist, list(self.base_dist.params.keys())[0]).new(shape).uniform()
        return (self.icdf(self.base_dist.cdf(self.lower_bound) +
                u * (self.base_dist.cdf(self.upper_bound) - self.base_dist.cdf(self.lower_bound))))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        Returns -inf in value is out of bounds
        """
        log_prob = self.base_dist.log_prob(value)
        log_prob[(value < self.lower_bound) | (value > self.upper_bound)] = -float('inf')
        log_prob = log_prob - (self.base_dist.cdf(self.upper_bound) - self.base_dist.cdf(self.lower_bound)).log()

    def cdf(self, value):
        """
        Cumulative distribution function for the truncated distribution
        """
        return ((self.base_dist.cdf(value) - self.base_dist.cdf(self.lower_bound)) /
                (self.base_dist.cdf(self.upper_bound) - self.base_dist.cdf(self.lower_bound)))
