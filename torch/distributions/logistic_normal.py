from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ComposeTransform, ExpTransform, StickBreakingTransform
from torch.distributions.utils import _sum_rightmost


class LogisticNormal(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by
    `mean` and `std` where::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X)) ~ Normal(loc, scale)

    Example::

        >>> m = LogisticNormal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # logistic-normal distributed with mean=0 and stddev=1
         0.4655
         0.5345
        [torch.FloatTensor of size (2,)]

    Args:
        loc (float or Tensor): mean
        scale (float or Tensor): standard deviation
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale):
        super(LogisticNormal, self).__init__(
            Normal(loc, scale), StickBreakingTransform())
        # Adjust event shape since StickBreakingTransform adds 1 dimension
        self._event_shape = self._event_shape.__class__([
            s + 1 for s in self._event_shape
        ])

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        self.base_dist._validate_log_prob_arg(value[..., :-1])
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_prob -= _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                       event_dim - transform.event_dim)
            y = x
        log_prob += _sum_rightmost(self.base_dist.log_prob(y),
                                   event_dim - len(self.base_dist.event_shape))
        return log_prob

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
