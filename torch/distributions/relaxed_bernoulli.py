import torch
from numbers import Number
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs


class LogitRelaxedBernoulli(Distribution):
    r"""
    Creates a LogitRelaxedBernoulli distribution parameterized by `probs` or `logits`,
    which is the logit of a RelaxedBernoulli distribution.

    Samples are logits of values in (0, 1). See [1] for more details.

    Args:
        temperature (Tensor):
        probs (Number, Tensor): the probabilty of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {'probs': constraints.unit_interval}
    support = constraints.real

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self.temperature = temperature
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            self.probs, = broadcast_all(probs)
        else:
            is_scalar = isinstance(logits, Number)
            self.logits, = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super(LogitRelaxedBernoulli, self).__init__(batch_shape, validate_args=validate_args)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.size()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        probs = clamp_probs(self.probs.expand(shape))
        uniforms = clamp_probs(self.probs.new(shape).uniform_())
        return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.temperature

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        return self.temperature.log() + diff - 2 * diff.exp().log1p()


class RelaxedBernoulli(TransformedDistribution):
    r"""
    Creates a RelaxedBernoulli distribution, parametrized by `temperature`, and either
    `probs` or `logits`. This is a relaxed version of the `Bernoulli` distribution, so
    the values are in (0, 1), and has reparametrizable samples.

    Example::

        >>> m = RelaxedBernoulli(torch.tensor([2.2]),
                                 torch.tensor([0.1, 0.2, 0.3, 0.99]))
        >>> m.sample()
         0.2951
         0.3442
         0.8918
         0.9021
        [torch.FloatTensor of size 4]

    Args:
        temperature (Tensor):
        probs (Number, Tensor): the probabilty of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    arg_constraints = {'probs': constraints.unit_interval}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(RelaxedBernoulli, self).__init__(LogitRelaxedBernoulli(temperature, probs, logits),
                                               SigmoidTransform(), validate_args=validate_args)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs
