import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn.functional import sigmoid

__all__ = [
    'AbsTransform',
    'AffineTransform',
    'ExpTransform',
    'InverseTransform',
    'LogprobTransform',
    'SigmoidTransform',
    'StickBreakingTransform',
    'Transform',
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

    def __eq__(self, other):
        return type(other) is type(self)

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def forward(self, x):
        """
        Invokes the memoized transform `x => y`.
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
        Inverts the memoized transform `y => x`.
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
    Transform via the mapping `y = exp(x)`.
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
    Transform via the mapping `y = sigmoid(x)` and `x = logit(y)`.
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
    Transform via the mapping `y = abs(x)`
    """
    domain = constraints.real
    codomain = constraints.positive

    def _forward(self, x):
        return x.abs()


class AffineTransform(Transform):
    """
    Transform via the pointwise affine mapping `y = loc + scale * x`.

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

    def __eq__(self, other):
        return (type(other) is AffineTransform) and self.loc.eq(other.loc).all() and self.scale.eq(other.scale).all()

    def _forward(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        result = torch.abs(self.scale).log()
        shape = x.shape
        if self.event_dim:
            # NOTE: no need for contiguous here
            result = result.view(*result.size()[:-self.event_dim], -1).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)


class LogprobTransform(Transform):
    """
    Transform from the simplex to unconstrained space via `y = log(x)`.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and this is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.simplex
    codomain = constraints.positive

    def _forward(self, x):
        probs = x
        return probs.log()

    def _inverse(self, y):
        logprobs = y
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        probs /= probs.sum(-1, True)
        return probs


class StickBreakingTransform(Transform):
    """
    Transform from the simplex to unconstrained of one fewer dimension via a
    stick-breaking process.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """
    domain = constraints.simplex
    codomain = constraints.positive

    def _forward(self, x):
        pmf = x
        cmf = pmf.cumsum(-1)
        sf = 1 - cmf
        units = x[..., :-1] / sf[..., :-1]
        return units.log()

    def _inverse(self, y):
        shape = y.shape[:-1] + (1 + y.shape[-1],)
        one = y.new([1]).expand(y.shape[:-1] + (1,))
        numer = sigmoid(y)
        denom = (1 - numer).cumprod(-1)
        probs = torch.cat([numer, one], -1) * torch.cat([one, denom], -1)
        return probs
