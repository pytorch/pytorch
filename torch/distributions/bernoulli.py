import torch
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


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
    has_enumerate_support = True

    def __init__(self, probs):
        self.probs, = broadcast_all(probs)

    def sample(self):
        return torch.bernoulli(self.probs)

    def sample_n(self, n):
        return torch.bernoulli(self.probs.expand(n, *self.probs.size()))

    def log_prob(self, value):
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - self.probs, self.probs])).log()

        # evaluate using the values
        return log_pmf.gather(0, value.unsqueeze(0).long()).squeeze(0)

    def enumerate_support(self):
        batch_shape = self.probs.shape
        values = torch.arange(2).long()
        values = values.view((-1,) + (1,) * len(batch_shape))
        values = values.expand((-1,) + batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
