import torch
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
        self._event_shape = torch.Size([s + 1 for s in self._event_shape])

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
