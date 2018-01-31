import weakref

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.utils import (_sum_rightmost, broadcast_all,
                                       lazy_property)
from torch.nn.functional import sigmoid

__all__ = [
    'AbsTransform',
    'AffineTransform',
    'BoltzmannTransform',
    'ComposeTransform',
    'ExpTransform',
    'LowerCholeskyTransform',
    'SigmoidTransform',
    'StickBreakingTransform',
    'Transform',
    'identity_transform',
]


class Transform(object):
    """
    Abstract class for invertable transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for tranforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        event_dim (int): Number of dimensions that are correlated together in
            the transform ``event_shape``. This should be 0 for pointwise
            transforms, 1 for transforms that act jointly on vectors, 2 for
            transforms that act jointly on matrices, etc.
    """
    bijective = False
    event_dim = 0

    def __init__(self, cache_size=0):
        self._cache_size = cache_size
        self._inv = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
        else:
            raise ValueError('cache_size must be 0 or 1')

    @property
    def inv(self):
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = weakref.ref(inv)
        return inv

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def __call__(self, x):
        """
        Computes the transform `x => y`.
        """
        if self._cache_size == 0:
            return self._call(x)
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._call(x)
        self._cached_x_y = x, y
        return y

    def _inv_call(self, y):
        """
        Inverts the transform `y => x`.
        """
        if self._cache_size == 0:
            return self._inverse(y)
        x_old, y_old = self._cached_x_y
        if y is y_old:
            return x_old
        x = self._inverse(y)
        self._cached_x_y = x, y
        return x

    def _call(self, x):
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


class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """
    def __init__(self, transform):
        super(_InverseTransform, self).__init__()
        self._inv = transform

    @constraints.dependent_property
    def domain(self):
        return self._inv.codomain

    @constraints.dependent_property
    def codomain(self):
        return self._inv.domain

    @property
    def bijective(self):
        return self._inv.bijective

    @property
    def event_dim(self):
        return self._inv.event_dim

    @property
    def inv(self):
        return self._inv

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        return self._inv == other._inv

    def __call__(self, x):
        return self._inv._inv_call(x)

    def log_abs_det_jacobian(self, x, y):
        return -self._inv.log_abs_det_jacobian(y, x)


class ComposeTransform(Transform):
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
    """
    def __init__(self, parts):
        super(ComposeTransform, self).__init__()
        self.parts = parts

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.parts == other.parts

    @constraints.dependent_property
    def domain(self):
        if not self.parts:
            return constraints.real
        return self.parts[0].domain

    @constraints.dependent_property
    def codomain(self):
        if not self.parts:
            return constraints.real
        return self.parts[-1].codomain

    @lazy_property
    def bijective(self):
        return all(p.bijective for p in self.parts)

    @lazy_property
    def event_dim(self):
        return max(p.event_dim for p in self.parts) if self.parts else 0

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = ComposeTransform([p.inv for p in reversed(self.parts)])
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        return inv

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        if not self.parts:
            return x.new([0]).expand_as(x)
        result = 0
        for part in self.parts:
            y = part(x)
            result += _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                     self.event_dim - part.event_dim)
            x = y
        return result


identity_transform = ComposeTransform([])


class ExpTransform(Transform):
    """
    Transform via the mapping `y = exp(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True

    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    def _call(self, x):
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
    bijective = True

    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        return sigmoid(x)

    def _inverse(self, y):
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -(y.reciprocal() + (1 - y).reciprocal()).log()


class AbsTransform(Transform):
    """
    Transform via the mapping `y = abs(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        return isinstance(other, AbsTransform)

    def _call(self, x):
        return x.abs()

    def _inverse(self, y):
        return y


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
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super(AffineTransform, self).__init__(cache_size=cache_size)
        self.loc, self.scale = broadcast_all(loc, scale)
        self.event_dim = event_dim

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False
        result = self.loc.eq(other.loc).all() and self.scale.eq(other.scale).all()
        if isinstance(result, Variable):
            result = result.data.view(-1)[0]
        return result

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        result = torch.abs(self.scale).log()
        shape = x.shape
        if self.event_dim:
            result_size = result.size()[:-self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)


class BoltzmannTransform(Transform):
    """
    Transform from unconstrained space to the simplex via `y = exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real
    codomain = constraints.simplex
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, BoltzmannTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        probs /= probs.sum(-1, True)
        return probs

    def _inverse(self, y):
        probs = y
        return probs.log()


class StickBreakingTransform(Transform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """
    domain = constraints.real
    codomain = constraints.simplex
    bijective = True
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, StickBreakingTransform)

    def _call(self, x):
        shape = x.shape[:-1] + (1 + x.shape[-1],)
        one = x.new([1]).expand(x.shape[:-1] + (1,))
        numer = sigmoid(x)
        denom = (1 - numer).cumprod(-1)
        probs = torch.cat([numer, one], -1) * torch.cat([one, denom], -1)
        return probs

    def _inverse(self, y):
        pmf = y
        cmf = pmf.cumsum(-1)
        sf = 1 - cmf
        units = y[..., :-1] / sf[..., :-1]
        return units.log()

    # TODO implement .log_abs_det_jacobian()


class LowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """
    domain = constraints.real
    codomain = constraints.lower_cholesky
    event_dim = 2

    def __eq__(self, other):
        return isinstance(other, LowerCholeskyTransform)

    def _call(self, x):
        if x.dim() != 2:
            raise NotImplementedError
        return x.tril(-1) + x.diag().exp().diag()

    def _inverse(self, y):
        if y.dim() != 2:
            raise NotImplementedError
        return y.tril(-1) + y.diag().log().diag()
