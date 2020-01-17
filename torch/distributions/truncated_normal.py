import torch
from torch.distributions.normal import Normal
from torch.distributions.truncated_distribution import TruncatedDistribution
import math


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])

    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010

    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte()
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            assert(lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            x[accept] = proposed_x[accept]
            done |= accept
    return x


class TruncatedNormal(TruncatedDistribution):
    r"""
    Creates a normal distribution parameterized by
    `loc` and `scale`, bounded to [`lower_bound`, `upper_bound`]

    Example::

        >>> m = TruncatedNormal(torch.Tensor([0.0]), torch.Tensor([1.0]), torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normal distributed with mean=0 and stddev=1, bounded to [0,1]
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (Tensor): mean of the distribution
        scale (Tensor): standard deviation of the distribution
        lower_bound (Tensor): lower bound for the distribution. Best to keep it greater than loc-4*scale for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than loc+4*scale for
        stable results
    """

    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(Normal(loc, scale), lower_bound, upper_bound, *args, **kwargs)
        # TODO: default values for lower and upper bounds

    def rsample(self, sample_shape=torch.Size()):
        eps = self._standard_truncnorm_sample((self.lower_bound - self.base_dist.mean) / self.base_dist.stddev,
                                              (self.upper_bound - self.base_dist.mean) / self.base_dist.stddev,
                                              sample_shape=sample_shape)
        return eps * self.base_dist.stddev + self.base_dist.mean


    @property
    def mean(self):
        return (self.base_dist.mean +
                self.base_dist.stddev *
                (self.base_dist.log_prob(self.lower_bound).exp() -
                 self.base_dist.log_prob(self.upper_bound).exp()) / self.Z)

    @property
    def variance(self):
        return self.base_dist.variance * (
            1 +
            (self.lower_bound * self.base_dist.log_prob(self.lower_bound).exp() -
             self.upper_bound * self.base_dist.log_prob(self.upper_bound).exp()
             ) / self.Z -
            (self.base_dist.log_prob(self.lower_bound).exp() -
             self.base_dist.log_prob(self.upper_bound).exp()
             )**2 / self.Z**2
            )

    @property
    def stddev(self):
        return self.variance ** 0.5

    @property
    def entropy():
        return (0.5 * math.log(2 * math.pi) + self.stddev.log() + self.Z.log() +
                0.5 * (self.lower_bound * self.base_dist.log_prob(self.lower_bound).exp() -
                       self.upper_bound * self.base_dist.log_prob(self.upper_bound).exp()
                       ) / self.Z)
