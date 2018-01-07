from collections import OrderedDict

from torch.distributions.distribution import Distribution

_KL_REGISTRY = OrderedDict()
_KL_DISPATCH_TABLE = {}


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup order is:

    1.  First look for an exact match.
    2.  Then find the first pair of registered superclasses, in order that
        functions were registered.

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
        _KL_DISPATCH_TABLE.clear()  # reset since lookup order may change
        return fun

    return decorator


def _dispatch_kl(type_p, type_q):
    # Look for an exact match.
    try:
        return _KL_REGISTRY[type_p, type_q]
    except KeyError:
        pass
    # Look for the first approximate match.
    for super_p, super_q in _KL_REGISTRY:
        if issubclass(type_p, super_p) and issubclass(type_q, super_q):
            return _KL_REGISTRY[super_p, super_q]
    raise NotImplementedError


def kl_divergence(p, q):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distrubution): A :class:`~torch.distributions.Distribution` object.
        q (Distrubution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        (Variable or Tensor): A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_DISPATCH_TABLE[type(p), type(q)]
    except KeyError:
        try:
            fun = _dispatch_kl(type(p), type(q))
        except NotImplementedError:
            fun = NotImplemented
        _KL_DISPATCH_TABLE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError
    return fun(p, q)
