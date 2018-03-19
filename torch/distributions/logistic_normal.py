import torch
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ComposeTransform, ExpTransform, StickBreakingTransform


class LogisticNormal(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by `loc` and `scale`
    that define the base `Normal` distribution transformed with the
    `StickBreakingTransform` such that::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)

    Example::

        >>> m = LogisticNormal(torch.Tensor([0.0] * 3), torch.Tensor([1.0] * 3))
        >>> m.sample()  # logistic-normal distributed with mean=(0, 0, 0) and
                        # stddev=(1, 1, 1) of the base Normal distribution
         0.4163
         0.1386
         0.3539
         0.0912
        [torch.FloatTensor of size (4,)]

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution
    """
    params = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        super(LogisticNormal, self).__init__(
            Normal(loc, scale), StickBreakingTransform(),
            validate_args=validate_args)
        # Adjust event shape since StickBreakingTransform adds 1 dimension
        self._event_shape = torch.Size([s + 1 for s in self._event_shape])

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
