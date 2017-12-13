import torch
from torch.autograd import Variable
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
    has_enumerate_support = True

    def __init__(self, probs):
        if probs.dim() != 1 and probs.dim() != 2:
            # TODO: treat higher dimensions as part of the batch
            raise ValueError("probs must be 1D or 2D")
        self.probs = probs

    def sample(self, sample_shape=()):
        if len(sample_shape) == 0:
            return torch.multinomial(self.probs, 1, True).squeeze(-1)
        elif len(sample_shape) == 1:
            if sample_shape[0] == 1:
                return self.sample().expand(1, 1)
            else:
                return torch.multinomial(self.probs, sample_shape[0], True).t()
        else:
            raise NotImplementedError("sample is not implemented for len(sample_shape)>1")

    def log_prob(self, value):
        p = self.probs / self.probs.sum(-1, keepdim=True)
        if value.dim() == 1 and self.probs.dim() == 1:
            # special handling until we have 0-dim tensor support
            return p.gather(-1, value).log()

        return p.gather(-1, value.unsqueeze(-1)).squeeze(-1).log()

    def enumerate_support(self):
        batch_shape, event_size = self.probs.shape[:-1], self.probs.shape[-1]
        values = torch.arange(event_size).long()
        values = values.view((-1,) + (1,) * len(batch_shape))
        values = values.expand((-1,) + batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
