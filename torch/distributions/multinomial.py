import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from numbers import Number
from torch.distributions import constraints
from torch.distributions.utils import log_sum_exp, lazy_property


class Multinomial(Distribution):
    r"""
    Creates a Multinomial distribution parameterized by `probs` and `total_count`.

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    Example::

        >>> m = Multinomial(10, torch.Tensor([ 1, 1, 1, 1]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         2
         2
         3
         3
        [torch.FloatTensor of size 4]


    Args:
        total_count (Tensor or Variable): event probabilities
        probs (Tensor or Variable): event probabilities
    """

    params = {'total_count': constraints.nonnegative_integer, 'probs': constraints.simplex}

    def __init__(self, total_count, probs=None, logits=None):
        self.total_count = total_count if isinstance(total_count, Number) else total_count.data[0]
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            probs = probs / probs.sum(-1, keepdim=True)
        else:
            logits = logits - log_sum_exp(logits)
        self._categorical = Categorical(probs=probs) if probs is not None else Categorical(logits=logits)
        batch_shape = probs.size()[:-1] if probs is not None else logits.size()[:-1]
        event_shape = probs.size()[-1:] if probs is not None else logits.size()[-1:]
        super(Multinomial, self).__init__(batch_shape, event_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        num_events = self._event_shape[0]
        extended_shape = torch.Size((1,)) if not sample_shape else sample_shape
        samples = self._categorical.sample((self.total_count,)+extended_shape).float()
        countsList = [torch.histc(m, bins=num_events, min=0, max=num_events-1)
                      for m in torch.unbind(samples.view(self.total_count, -1), dim=-1)]
        counts = torch.stack(countsList, dim=0)
        return counts.contiguous().view(self._extended_shape(sample_shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        batch_log_pdf_shape = torch.Size((1,)) if not value.size()[:-1] else value.size()[:-1]
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        log_powers = (self.logits[value != 0] * value[value != 0]).sum(-1)
        batch_log_pdf = log_factorial_n - log_factorial_xs + log_powers
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def entropy(self):
        raise NotImplementedError
