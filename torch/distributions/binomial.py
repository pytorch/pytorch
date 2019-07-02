from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs


def _clamp_by_zero(x):
    # works like clamp(x, min=0) but has grad at 0 is 0.5
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2


class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """
    arg_constraints = {'total_count': constraints.nonnegative_integer,
                       'probs': constraints.unit_interval,
                       'logits': constraints.real}
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

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

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
                arange = torch.arange(max_count, dtype=self._param.dtype, device=self._param.device)
                mask = arange >= self.total_count.unsqueeze(-1)
                if torch._C._get_tracing_state():
                    # [JIT WORKAROUND] lack of support for .masked_fill_()
                    bernoullis[mask.expand(shape)] = 0.
                else:
                    bernoullis.masked_fill_(mask, 0.)
            return bernoullis.sum(dim=-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = (self.total_count * _clamp_by_zero(self.logits)
                          + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
                          - log_factorial_n)
        return value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = torch.arange(1 + total_count, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
