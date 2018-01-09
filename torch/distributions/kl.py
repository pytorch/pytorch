import warnings
from functools import total_ordering

import torch

from .distribution import Distribution
from .gamma import Gamma

_KL_REGISTRY = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_KL_MEMOIZE = {}  # Memoized version mapping many specific (type, type) pairs to functions.


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError('Expected type_q to be a Distribution subclass but got {}'.format(type_q))

    def decorator(fun):
        _KL_REGISTRY[type_p, type_q] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [(super_p, super_q) for super_p, super_q in _KL_REGISTRY
               if issubclass(type_p, super_p) and issubclass(type_q, super_q)]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    left_p, left_q = min(_Match(*m) for m in matches).types
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn('Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.format(
            type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__),
            RuntimeWarning)
    return left_fun


def kl_divergence(p, q):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distrubution): A :class:`~torch.distributions.Distribution` object.
        q (Distrubution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Variable or Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError
    return fun(p, q)


################################################################################
# KL Divergence Implementations
################################################################################

@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    # Adapted from https://stats.stackexchange.com/questions/11646
    def f(a, b, c, d):
        return -d * a / c + b * a.log() - torch.lgamma(b) + (b - 1) * torch.digamma(d) + (1 - b) * c.log()

    return f(p.beta, p.alpha, p.beta, p.alpha) - f(q.beta, q.alpha, p.beta, p.alpha)
