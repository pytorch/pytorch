import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all, log_sum_exp
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform


class ExpRelaxedCategorical(Distribution):
    r"""
    Creates a ExpRelaxedCategorical parameterized by `probs` and `temperature`.
    Returns the log of a point in the simplex. Based on the interface to OneHotCategorical.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): the log probability of each event.

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {'probs': constraints.simplex}
    support = constraints.real
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super(ExpRelaxedCategorical, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        uniforms = clamp_probs(self.logits.new(self._extended_shape(sample_shape)).uniform_())
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - log_sum_exp(scores)

    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = (self.temperature.new(self.temperature.shape).fill_(K).lgamma() -
                     self.temperature.log().mul(-(K - 1)))
        score = logits - value.mul(self.temperature)
        score = (score - log_sum_exp(score)).sum(-1)
        return score + log_scale


class RelaxedOneHotCategorical(TransformedDistribution):
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by `temperature` and either `probs` or `logits`.
    This is a relaxed version of the `OneHotCategorical` distribution, so its
    values are on simplex, and has reparametrizable samples.

    Example::

        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
                                         torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()  # equal probability of 1, 1, 2, 3
         0.1294
         0.2324
         0.3859
         0.2523
        [torch.FloatTensor of size 4]

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): the log probability of each event.
    """
    arg_constraints = {'probs': constraints.simplex}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(RelaxedOneHotCategorical, self).__init__(ExpRelaxedCategorical(temperature, probs, logits),
                                                       ExpTransform(), validate_args=validate_args)

    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs
