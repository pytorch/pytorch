import torch
from torch.distributions.distribution import Distribution


class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by `probs`.

    .. note::
        It is equivalent to the distribution that ``multinomial()`` samples from.

    Samples are integers from `0 ... K-1` where `K` is probs.size(-1).

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         3
        [torch.LongTensor of size 1]

    Args:
        probs (Tensor or Variable): event probabilities
    """

    def __init__(self, probs):
        if probs.dim() != 1 and probs.dim() != 2:
            # TODO: treat higher dimensions as part of the batch
            raise ValueError("probs must be 1D or 2D")
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1, True).squeeze(-1)

    def sample_n(self, n):
        if n == 1:
            return self.sample().expand(1, 1)
        else:
            return torch.multinomial(self.probs, n, True).t()

    def log_prob(self, value):
        p = self.probs / self.probs.sum(-1, keepdim=True)
        if value.dim() == 1 and self.probs.dim() == 1:
            # special handling until we have 0-dim tensor support
            return p.gather(-1, value).log()

        return p.gather(-1, value.unsqueeze(-1)).squeeze(-1).log()
