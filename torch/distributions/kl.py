from collections import OrderedDict

import torch
from torch.distributions.distribution import Distribution

KL_REGISTRY = OrderedDict()


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Args:
        type_p (type): A distribution subclass.
        type_q (type): A distribution subclass.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError('Expected type_q to be a Distribution subclass but got {}'.format(type_q))
    p_registry = KL_REGISTRY.setdefault(type_p, OrderedDict())

    def decorator(fun):
        p_registry[type_q] = fun
        return fun

    return decorator


def _dispatch_kl(type_p, type_q):
    # Look for an exact match.
    try:
        return KL_REGISTRY[type_p][type_q]
    except KeyError:
        pass
    # Look for the first approximate match.
    for super_p, p_registry in KL_REGISTRY.items():
        if issubclass(type_p, super_p):
            try:
                return p_registry[type_q]
            except KeyError:
                for super_q, fun in p_registry.items():
                    if issubclass(type_q, super_q):
                        return fun
    raise NotImplementedError


def kl_divergence(p, q):
    """
    Compute Kullback-Leibler divergence `KL(p || q)` between two distributions.

    Args:
        p (Distribution): A distribution object.
        q (Distribution): A distribution object.

    Returns:
        (Variable or Tensor): A batch of KL distributions of shape `batch_shape`.
    """
    return _dispatch_kl(type(p), type(q))(p, q)
