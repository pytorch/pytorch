import torch
from numbers import Number
from torch.autograd import Variable
from torch.distributions.bijectors import Bijector
from torch.distributions.distribution import Distribution


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Bijectors to a base distribution.
    Let f be the composition of bijectors applied,
    X ~ BaseDistribution
    Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
    log p(Y) = log p(X) + log det (dX/dY)
    """
    def __init__(self, base_distribution, bijectors=[], *args, **kwargs):
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.has_rsample = base_distribution.has_rsample
        self.base_dist = base_distribution
        if isinstance(bijectors, Bijector):
            self.bijectors = [bijectors, ]
        elif isinstance(bijectors, list):
            for bijector in bijectors:
                if not isinstance(bijector, Bijector):
                    raise ValueError("bijectors must be a Bijector or a list of Bijectors")
            self.bijectors = bijectors

    def sample(self, sample_shape=torch.Size()):
        """
        :returns: a sample y
        :rtype: torch.autograd.Variable

        Sample from base distribution and pass through bijector(s)
        """
        x = self.base_dist.sample(sample_shape)
        next_input = x
        for bijector in self.bijectors:
            y = bijector.forward(next_input)
            if bijector.add_inverse_to_cache:
                bijector._add_intermediate_to_cache(next_input, y, 'x')
            next_input = y
        return next_input

    def rsample(self, sample_shape=torch.Size()):
        """
        :returns: a sample y
        :rtype: torch.autograd.Variable

        Sample from base distribution and pass through bijector(s)
        """
        x = self.base_dist.rsample(sample_shape)
        next_input = x
        for bijector in self.bijectors:
            y = bijector.forward(next_input)
            if bijector.add_inverse_to_cache:
                bijector._add_intermediate_to_cache(next_input, y, 'x')
            next_input = y
        return next_input

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    def log_prob(self, value):
        """
        :param y: a value sampled from the transformed distribution
        :type y: torch.autograd.Variable

        :returns: the score (the log pdf) of y
        :rtype: torch.autograd.Variable

        Scores the sample by inverting the bijector(s) and computing the score using the score
        of the base distribution and the log det jacobian
        """
        log_pdf = 0.0
        for bijector in reversed(self.bijectors):
            log_pdf += bijector.inverse_log_det_jacobian(value)
            value = bijector.inverse(value)
        log_pdf += self.base_dist.log_prob(value)
        return log_pdf
