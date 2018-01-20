import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

__all__ = [
    'Bijector',
    'ExpBijector',
    'InverseBijector',
]


class Bijector(object):
    """
    Abstract class for bijective transformations with computable inverse log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Bijectors are intended to be short-lived objects. They memoize the forward
    and inverse computations to avoid work; therefore :meth:`inverse` is
    nearly free after calling :meth:`forward`. To clear the memoization cache,
    delete the object and create a new object.

    Derived classes should implement one or both of :meth:`_forward` or
    :meth:`_inverse` and should implement :meth:`log_abs_det_jacobian`.
    Derived classes may store intermediate results in the `._cache` dict.
    """
    is_injective = True

    def __init__(self):
        self._cache = {}

    def forward(self, x):
        """
        Invokes the bijection `x => y`.
        """
        try:
            return self._cache['forward', x]
        except KeyError:
            y = self._forward(x)
            self._cache['forward', x] = y
            self._cache['inverse', y] = x
            return y

    def inverse(self, y):
        """
        Inverts the bijection `y => x`.
        """
        try:
            return self._cache['inverse', y]
        except KeyError:
            x = self._inverse(y)
            self._cache['forward', x] = y
            self._cache['inverse', y] = x
            return x

    def _forward(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError


class InverseBijector(Bijector):
    """
    Inverts a single :class:`Bijector`.
    """
    def __init__(self, bijector):
        self.bijector = bijector

    @constraints.dependent_property
    def domain(self):
        return self.bijector.codomain

    @constraints.dependent_property
    def codomain(self):
        return self.bijector.domain

    def forward(self, x):
        return self.bijector.inverse(x)

    def inverse(self, y):
        return self.bijector.forward(y)

    def log_abs_det_jacobian(self, x, y):
        return -self.bijector.log_abs_det_jacobian(y, x)


class ExpBijector(Bijector):
    """
    Bijector for the mapping `y = exp(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive

    def _forward(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x
