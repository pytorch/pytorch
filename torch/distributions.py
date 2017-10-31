"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions.

The :meth:`log_prob` method is useful for policy gradient based methods. If the
parameters of the distribution are differentiable, then the result of ``log_prob``
is also differentiable.

Example::

    probs = network(input)
    m = Multinomial(probs)
    action = m.sample()
    loss = -m.log_prob(action) * get_reward(env, action)
    loss.backward()
"""
import math
from numbers import Number
import torch


__all__ = ['Distribution', 'Bernoulli', 'Multinomial', 'Normal']


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    def sample(self):
        """
        Generates a single sample or single batch of samples if the distribution
        parameters are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError


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
        probs (Tensor or Variable): the probabilty of sampling `1`
    """

    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.bernoulli(self.probs)

    def sample_n(self, n):
        return torch.bernoulli(self.probs.expand(n, *self.probs.size()))

    def log_prob(self, value):
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - self.probs, self.probs])).log()

        # evaluate using the values
        return log_pmf.gather(0, value.unsqueeze(0).long()).squeeze(0)


class Multinomial(Distribution):
    r"""
    Creates a multinomial distribution parameterized by `probs`.

    Samples are integers from `0 ... K-1` where `K` is probs.size(-1).

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Multinomial(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         3
        [torch.LongTensor of size 1]

    Args:
        probs (Tensor or Variable): event probabilities
    """

    def __init__(self, probs):
        if probs.dim() != 1 and probs.dim() != 2:
            # TODO: treat higher dimensions as part of the batch
            raise ValueError("probs must be 1D or 2D")
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1, True).squeeze(-1)

    def sample_n(self, n):
        if n == 1:
            return self.sample().expand(1, 1)
        else:
            return torch.multinomial(self.probs, n, True).t()

    def log_prob(self, value):
        p = self.probs / self.probs.sum(-1, keepdim=True)
        if value.dim() == 1 and self.probs.dim() == 1:
            # special handling until we have 0-dim tensor support
            return p.gather(-1, value).log()

        return p.gather(-1, value.unsqueeze(-1)).squeeze(-1).log()


class Normal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    `mean` and `std`.

    Example::

        >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normally distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        mean (float or Tensor or Variable): mean of the distribution
        std (float or Tensor or Variable): standard deviation of the distribution
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return torch.normal(self.mean, self.std)

    def sample_n(self, n):
        # cleanly expand float or Tensor or Variable parameters
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())
        return torch.normal(expand(self.mean), expand(self.std))

    def log_prob(self, value):
        # compute the variance
        var = (self.std ** 2)
        log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
        return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
