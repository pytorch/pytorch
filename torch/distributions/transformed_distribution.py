import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.
    """
    def __init__(self, base_distribution, transforms):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError("transforms must be a Transform or list, but was {}".format(transforms))
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] + [t.event_dim for t in self.transforms])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super(TransformedDistribution, self).__init__(batch_shape, event_shape)

    @constraints.dependent_property
    def params(self):
        return self.base_dist.params  # TODO add params of transforms?

    @constraints.dependent_property
    def support(self):
        return self.transforms[-1].codomain if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from base distribution
        and applies `transform()` for every transform in the list.
        """
        x = self.base_dist.sample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies `transform()`
        for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score using the score
        of the base distribution and the log abs det jacobian
        """
        self.base_dist._validate_log_prob_arg(value)
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

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the transform(s) and computing
        the score of the base distribution
        """
        self.base_dist._validate_log_prob_arg(value)
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        return self.base_dist.cdf(value)

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using transform(s) and computing
        the score of the base distribution
        """
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value
