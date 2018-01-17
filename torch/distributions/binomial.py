from numbers import Number
import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs
from torch.distributions.utils import clamp_probs
from torch.autograd import Variable


class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by `total_count` and
    either `probs` or `logits` (but not both).

    -   Requires a single shared `total_count` for all
        parameters and samples.

    Example::

        >>> m = Binomial(100, torch.Tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
         0
         22
         71
         100
        [torch.FloatTensor of size 4]]

    Args:
        total_count (int): number of Bernoulli trials
        probs (Tensor or Variable): Event probabilities
        logits (Tensor or Variable): Event log-odds
    """
    params = {'probs': constraints.unit_interval}
    has_enumerate_support = True

    def __init__(self, total_count=1, probs=None, logits=None):
        if not isinstance(total_count, Number):
            raise NotImplementedError('inhomogeneous total_count is not supported')
        self.total_count = total_count
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            self.probs, = broadcast_all(probs)
        else:
            self.logits, = broadcast_all(logits)

        probs_or_logits = probs if probs is not None else logits
        if isinstance(probs_or_logits, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = probs_or_logits.size()
        super(Binomial, self).__init__(batch_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape) + (self.total_count,)
        return torch.bernoulli(self.probs.unsqueeze(-1).expand(shape)).sum(dim=-1)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        probs = clamp_probs(self.probs)
        log_factorial_n = math.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                value * self.logits + self.total_count * torch.log1p(-probs))

    def enumerate_support(self):
        values = torch.arange(self.total_count)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
