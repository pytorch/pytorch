from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property
from torch.nn.functional import binary_cross_entropy_with_logits


class Bernoulli(Distribution):
    r"""
    Creates a Bernoulli distribution parameterized by `probs`.

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> m = Bernoulli(torch.Tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
         0.0
        [torch.FloatTensor of size 1]

    Args:
        probs (Number, Tensor or Variable): the probabilty of sampling `1`
    """
    params = {'probs': constraints.unit_interval}
    support = constraints.boolean
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None):
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
        super(Bernoulli, self).__init__(batch_shape)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduce=False)

    def entropy(self):
        return binary_cross_entropy_with_logits(self.logits, self.probs, reduce=False)

    def enumerate_support(self):
        values = torch.arange(2).long()
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
