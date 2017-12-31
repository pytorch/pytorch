import torch
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical


class OneHotCategorical(Distribution):
    r"""
    Creates a one-hot categorical distribution parameterized by `probs`.

    Samples are one-hot coded vectors of size probs.size(-1).

    See also: :func:`torch.distributions.Categorical`

    Example::

        >>> m = OneHotCategorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         0
         0
         1
         0
        [torch.FloatTensor of size 4]

    Args:
        probs (Tensor or Variable): event probabilities
    """
    has_enumerate_support = True

    def __init__(self, probs):
        self._categorical = Categorical(probs)
        batch_shape = probs.size()[:-1]
        event_shape = probs.size()[-1:]
        super(OneHotCategorical, self).__init__(batch_shape, event_shape)

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        one_hot = probs.new(self._extended_shape(sample_shape)).zero_()
        indices = self._categorical.sample(sample_shape)
        if indices.dim() < one_hot.dim():
            indices = indices.unsqueeze(-1)
        return one_hot.scatter_(-1, indices, 1)

    def log_prob(self, value):
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)

    def entropy(self):
        return self._categorical.entropy()

    def enumerate_support(self):
        probs = self._categorical.probs
        n = self.event_shape[0]
        if isinstance(probs, Variable):
            values = Variable(torch.eye(n, out=probs.data.new(n, n)))
        else:
            values = torch.eye(n, out=probs.new(n, n))
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        return values.expand((n,) + self.batch_shape + (n,))
