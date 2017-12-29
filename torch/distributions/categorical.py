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
        self.probs = probs
        batch_shape = self.probs.size()[:-1]
        super(Categorical, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        num_events = self.probs.size()[-1]
        sample_shape = self._extended_shape(sample_shape)
        param_shape = sample_shape + self.probs.size()[-1:]
        probs = self.probs.expand(param_shape)
        probs_2d = probs.contiguous().view(-1, num_events)
        sample_2d = torch.multinomial(probs_2d, 1, True)
        return sample_2d.contiguous().view(sample_shape)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        param_shape = value.size() + self.probs.size()[-1:]
        log_pmf = (self.probs / self.probs.sum(-1, keepdim=True)).log()
        log_pmf = log_pmf.expand(param_shape)
        return log_pmf.gather(-1, value.unsqueeze(-1).long()).squeeze(-1)

    def entropy(self):
        p_log_p = torch.log(self.probs) * self.probs
        p_log_p[self.probs == 0] = 0
        return -p_log_p.sum(-1)

    def enumerate_support(self):
        num_events = self.probs.size()[-1]
        values = torch.arange(num_events).long()
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
