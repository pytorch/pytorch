from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import InvertibleBoltzmannTransform


class LogitNormal(TransformedDistribution):
    r"""
    Creates a logit-normal distribution parameterized by
    `mean` and `std` where::

        X ~ LogitNormal(loc, scale)
        Y = log(X / (1 - X)) ~ Normal(loc, scale)

    Example::

        >>> m = LogitNormal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # logit-normal distributed with mean=0 and stddev=1
         0.4655
         0.5345
        [torch.FloatTensor of size (2,)]

    Args:
        loc (float or Tensor): mean
        scale (float or Tensor): standard deviation
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale):
        super(LogitNormal, self).__init__(
            Normal(loc, scale), InvertibleBoltzmannTransform())
        # TODO: temporary fix to avoid dimension mismatch in the checks
        self.base_dist._validate_log_prob_arg = (lambda x: None)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
