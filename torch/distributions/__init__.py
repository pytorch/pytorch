r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions.

Policy gradient methods can be implemented using the
:meth:`~torch.distributions.Distribution.log_prob` method, when the probability
density function is differentiable with respect to its parameters. A basic
method is the REINFORCE rule:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # NOTE: this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()
"""

from .bernoulli import Bernoulli
from .beta import Beta
from .categorical import Categorical
from .dirichlet import Dirichlet
from .distribution import Distribution
from .gamma import Gamma
from .normal import Normal


__all__ = ['Distribution', 'Bernoulli', 'Beta', 'Categorical', 'Dirichlet', 'Gamma', 'Normal']
