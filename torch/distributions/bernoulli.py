import torch
from torch.distributions.distribution import Distribution


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
        probs (Tensor or Variable): the probabilty of sampling `1`
    """

    def __init__(self, probs):
        self.probs = probs

    def sample(self, sample_shape=()):
        if len(sample_shape) == 0:
            return torch.bernoulli(self.probs)
        elif len(sample_shape) == 1:
            return torch.bernoulli(self.probs.expand(sample_shape[0], *self.probs.size()))
        else:
        	raise NotImplementedError("sample is not implemented for len(sample_shape)>1")

    def log_prob(self, value):
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - self.probs, self.probs])).log()

        # evaluate using the values
        return log_pmf.gather(0, value.unsqueeze(0).long()).squeeze(0)
