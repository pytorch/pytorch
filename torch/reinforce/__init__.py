"""
The ``reinforce`` package contains sampling functions that return both the
random samples and the log of the probability density function (pdf) or
probability mass function (pmf) evaluated at each sample. These
log-probabilities can be backpropagated through for policy gradient methods.

Example::

    probs = network(input)
    action, log_prob = torch.reinforce.multinomial(probs)
    loss = -log_prob * get_reward(env, action)
    loss.backward()
"""
import math
import torch


__all__ = ['multinomial', 'bernoulli', 'normal']


def multinomial(input, num_samples=1):
    r"""
    Samples from the multinomial distribution characterized by `input`. Returns
    the samples and the log-probabilities of each returned sample.

    See also: :func:`torch.multinomial`

    .. note::

        When `num_samples` is greater than one, this function always samples
        with replacement.

    Args:
        input (Variable): event probabilities
        num_samples (int): number of samples to draw

    Returns:
        a tuple of the samples and the log-probabilities
    """
    if input.dim() < 1 or input.dim() > 2:
        raise ValueError("multinomial expects a 1D or 2D tensor 'input' (got {})"
                         .format(input.dim()))

    s = torch.multinomial(input, num_samples, True)
    return s, _multinomial_logpmf(s, input)


def _multinomial_logpmf(s, p):
    # normalize probabilities: multinomial accepts probabilities that sum to
    # less than one
    p = p / p.sum(-1, keepdim=True)
    return p.gather(-1, s).log()


def bernoulli(input):
    r"""
    Draws binary random numbers from the bernoulli distribution characterized
    by `input`. Returns the samples and the log-probabilities of each returned
    sample.

    See also: :func:`torch.bernoulli`

    Args:
        input (Variable): event probabilities

    Returns:
        a tuple of the samples and the log-probabilities
    """
    s = torch.bernoulli(input)
    return s, _bernoulli_logpmf(s, input)


def _bernoulli_logpmf(s, p):
    # compute the log probabilities for 0 and 1
    log_pfms = torch.stack([1 - p, p]).log()

    # evaluate using the samples s.
    return log_pfms.gather(0, s.unsqueeze(0).long()).squeeze(0)


def normal(mean, std):
    r"""
    Draws random numbers from the normal distribution and returns the samples
    and the logarithm of the probability density function at each sample.

    See also: :func:`torch.normal`

    Args:
        mean (float or Variable): mean of the distribution
        std (float or Variable): standard deviation of the distribution

    Returns:
        a tuple of the samples and the log-probabilities
    """
    s = torch.normal(mean, std)
    return s, _normal_logpdf(s, mean, std)


def _normal_logpdf(s, mean, std):
    # compute the variance and clamp it to at least 1e-10 to avoid divide-by-zero
    var = (std ** 2).clamp(min=1e-10)

    # compute the log-pdf
    return (s - mean) ** 2 / (-2 * var) - std.log() - math.log(math.sqrt(2 * math.pi))
