from numbers import Number

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
        if isinstance(probs, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.probs.size()
        super(Bernoulli, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        param_shape = value.size()
        probs = self.probs.expand(param_shape)
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - probs, probs], dim=-1)).log()
        # evaluate using the values
        return log_pmf.gather(-1, value.unsqueeze(-1).long()).squeeze(-1)

    def enumerate_support(self):
        values = torch.arange(2).long()
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
