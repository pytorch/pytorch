import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, _batch_vector_diag


class MultivariateNormalDiag(Distribution):
    r"""
    A multivariate normal distribution with diagonal covariance.

    Example:

        >>> m = MultivariateNormalDiag(torch.zeros([2]), torch.ones([2]))
        >>> m.sample()  # normally distributed with mean=`[0, 0]`, scalae_diag=`[1, 1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        scale_diag (Tensor): diagonal of standard deviations with shape
            `batch_shape + event_shape`
    """
    arg_constraints = {"loc": constraints.real,
                       "scale_diag": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale_diag, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]
        if scale_diag.shape[-1:] != event_shape:
            raise ValueError("scale_diag must be a batch of vectors with shape {}".format(event_shape))

        try:
            self.loc, self.scale_diag = torch.broadcast_tensors(loc, scale_diag)
        except RuntimeError:
            raise ValueError("Incompatible batch shapes: loc {}, scale_diag {}"
                             .format(loc.shape, scale_diag.shape))
        batch_shape = self.loc.shape[:-1]
        super(LowRankMultivariateNormal, self).__init__(batch_shape, event_shape,
                                                        validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        return self.scale_diag.pow(2)

    @lazy_property
    def covariance_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(2))

    @lazy_property
    def precision_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(-2))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new_empty(shape).normal_()
        return self.loc + self.scale_diag * eps

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        return (
            -0.5 * self._event_shape[0] * math.log(2 * math.pi) -
            self.scale_diag.log().sum(-1) -
            0.5 * (diff / self.scale_diag).pow(2).sum(-1)
        )

    def entropy(self):
        return (
            0.5 * self._event_shape[0] * (math.log(2 * math.pi) + 1) +
            self.scale_diag.log().sum(-1)
        )
