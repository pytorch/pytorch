from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs


class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by `total_count` and
    either `probs` or `logits` (but not both). `total_count` must be
    broadcastable with `probs`/`logits`.

    Example::

        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
         0
         22
         71
         100
        [torch.FloatTensor of size 4]]

        >>> m = Binomial(torch.Tensor([[5.], [10.]]), torch.Tensor([0.5, 0.8]))
        >>> x = m.sample()
         4  5
         7  6
        [torch.FloatTensor of size (2,2)]

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.unit_interval}
    has_enumerate_support = True

    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            self.total_count, self.probs, = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.logits)
            is_scalar = isinstance(self.probs, Number)
        else:
            self.total_count, self.logits, = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)
            is_scalar = isinstance(self.logits, Number)

        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super(Binomial, self).__init__(batch_shape, validate_args=validate_args)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self):
        return self.total_count * self.probs

    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            max_count = max(int(self.total_count.max()), 1)
            shape = self._extended_shape(sample_shape) + (max_count,)
            bernoullis = torch.bernoulli(self.probs.unsqueeze(-1).expand(shape))
            if self.total_count.min() != max_count:
                arange = torch.arange(max_count, out=self.total_count.new_empty(max_count))
                mask = arange >= self.total_count.unsqueeze(-1)
                bernoullis.masked_fill_(mask, 0.)
            return bernoullis.sum(dim=-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        max_val = (-self.logits).clamp(min=0.0)
        # Note that: torch.log1p(-self.probs)) = max_val - torch.log1p((self.logits + 2 * max_val).exp()))
        return (log_factorial_n - log_factorial_k - log_factorial_nmk +
                value * self.logits + self.total_count * max_val -
                self.total_count * torch.log1p((self.logits + 2 * max_val).exp()))

    def enumerate_support(self):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = self._new(1 + total_count,)
        torch.arange(1 + total_count, out=values)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        return values
