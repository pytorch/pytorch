import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.constraints import Constraint
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs


class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return self.lower_bound <= value


class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (self.lower_bound <= value) & (value < self.upper_bound)


class NegativeBinomial(Distribution):
    r"""
        Creates a Negative Binomial distribution, i.e. distribution
        of the number of independent identical Bernoulli trials
        needed before `total_count` failures are achieved. The probability
        of success of each Bernoulli trial is `probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the open interval (0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """
    arg_constraints = {'total_count': _GreaterThanEq(0),
                       'probs': _HalfOpenInterval(0., 1.)}
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            self.total_count, self.probs, = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            self.total_count, self.logits, = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super(NegativeBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        return self.total_count * torch.exp(self.logits)

    @property
    def variance(self):
        return self.mean / torch.sigmoid(-self.logits)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = torch.distributions.Gamma(concentration=self.total_count,
                                             rate=(1 - self.probs) / self.probs).sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = (self.total_count * F.logsigmoid(-self.logits) +
                                 value * F.logsigmoid(self.logits))

        log_normalization = (-torch.lgamma(self.total_count + value) + torch.lgamma(1. + value) +
                             torch.lgamma(self.total_count))

        return log_unnormalized_prob - log_normalization
