import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution


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
    params = {'probs': constraints.simplex}
    support = constraints.simplex
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None):
        self._categorical = Categorical(probs, logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super(OneHotCategorical, self).__init__(batch_shape, event_shape)

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

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
        n = self.event_shape[0]
        values = self._new((n, n))
        torch.eye(n, out=values.data if isinstance(values, Variable) else values)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        return values.expand((n,) + self.batch_shape + (n,))
