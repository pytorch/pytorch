from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform

__all__ = ['LogisticNormal']

class LogisticNormal(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `StickBreakingTransform` such that::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution

    Example::

        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)
        >>> # of the base Normal distribution
        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])

    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super(LogisticNormal, self).__init__(base_dist,
                                             StickBreakingTransform(),
                                             validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticNormal, _instance)
        return super(LogisticNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.base_dist.scale
