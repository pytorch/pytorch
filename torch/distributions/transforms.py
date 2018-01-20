import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn.functional import sigmoid

__all__ = [
    'AffineTransform',
    'Transform',
    'ExpTransform',
    'InverseTransform',
    'AbsTransform',
]


class Transform(object):
    """
    Abstract class for transformations with computable inverse log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Transforms are intended to be short-lived objects. They memoize the forward
    and inverse computations to avoid work; therefore :meth:`inverse` is
    nearly free after calling :meth:`forward`. To clear the memoization cache,
    delete the object and create a new object.

    Derived classes should implement one or both of :meth:`_forward` or
    :meth:`_inverse` and should implement :meth:`log_abs_det_jacobian`.
    Derived classes may store intermediate results in the `._cache` dict.
    """

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


class InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
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


class ExpTransform(Transform):
    """
    Transform for the mapping `y = exp(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive

    def _forward(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x


class SigmoidTransform(Transform):
    """
    Transform for the mapping `y = sigmoid(x)` and `x = logit(y)`.
    """
    domain = constraints.real
    codomain = constraints.unit_interval

    def _forward(self, x):
        return sigmoid(x)

    def _inverse(self, y):
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -(y.reciprocal() + (1 - y).reciprocal()).log()


class AbsTransform(Transform):
    """
    Transform for the mapping `y = abs(x)`
    """
    domain = constraints.real
    codomain = constraints.positive

    def _forward(self, x):
        return x.abs()


class AffineTransform(Transform):
    """
    Transform for the pointwise affine mapping `y = loc + scale * x`.

    Args:
        loc (Tensor or Variable): Location parameter.
        scale (Tensor or Variable): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, loc, scale, event_dim=0):
        super(AffineTransform, self).__init__()
        self.loc = loc
        self.scale = scale
        self.event_dim = event_dim

    def _forward(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return y / self.scale - self.loc

    def log_abs_det_jacobian(self, x, y):
        result = torch.abs(self.scale).log()
        shape = x.shape
        for _ in range(self.event_dim):
            result = result.sum(-1)
            shape = shape[:-1]
        return result.expand(shape)
