import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.distributions.distribution import Distribution


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms to a base distribution.
    Let f be the composition of transforms applied,
    X ~ BaseDistribution
    Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
    log p(Y) = log p(X) + log det (dX/dY)
    """
    def __init__(self, base_distribution, transforms=[], *args, **kwargs):
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms

    @constraints.dependent_property
    def params(self):
        return self.base_dist.params  # TODO add params of transforms?

    @constraints.dependent_property
    def support(self):
        return self.transforms[-1].codomain if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape

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
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_prob -= transform.log_abs_det_jacobian(x, y)
            y = x
        log_prob += self.base_dist.log_prob(y)
        return log_prob
