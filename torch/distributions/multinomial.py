import torch
import math
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical


class Multinomial(Distribution):
    r"""
    Creates a Multinomial distribution parameterized by `probs` and `total_count`.

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    Example::

        >>> m = Multinomial(torch.Tensor(10, [ 1, 1, 1, 1]))
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
    def __init__(self, total_count, probs):
        self.total_count = total_count
        self._categorical = Categorical(probs)
        batch_shape = probs.size()[:-1]
        event_shape = probs.size()[-1:]
        super(Multinomial, self).__init__(batch_shape, event_shape)

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        print(sample_shape)
        num_events = self._event_shape[0]
        print(sample_shape)
        extended_shape = torch.Size((1,)) if not sample_shape else sample_shape
        print(extended_shape)
        samples = self._categorical.sample((self.total_count,)+extended_shape).float()
        print(samples)
        countsList = [torch.histc(m, bins=num_events, min=0, max=num_events-1)
                      for m in torch.unbind(samples.view(self.total_count, -1), dim=-1)]
        print(countsList)
        counts = torch.stack(countsList, dim=0)
        print(counts)
        return counts.contiguous().view(self._extended_shape(sample_shape))

    def log_prob(self, value):
        batch_log_pdf_shape = self.batch_shape(value) + (1,)
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        log_powers = (value * torch.log(self.ps)).sum(-1)
        batch_log_pdf = log_factorial_n - log_factorial_xs + log_powers
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def entropy(self):
        return self._categorical.entropy()
