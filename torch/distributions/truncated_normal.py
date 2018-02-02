from torch.distributions.normal import Normal
from torch.distributions.truncated_distribution import TruncatedDistribution


class TruncatedNormal(TruncatedDistribution):
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
        loc (float or Tensor or Variable): mean of log of distribution
        scale (float or Tensor or Variable): standard deviation of log ofthe distribution
    """
    def __init__(self, loc, scale, lower_bound, upper_bound):
        super(TruncatedNormal, self).__init__(Normal(loc, scale), lower_bound, upper_bound)
