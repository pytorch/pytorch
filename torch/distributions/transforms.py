import math
import numbers
import weakref

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (_sum_rightmost, broadcast_all,
                                       lazy_property)
from torch.nn.functional import pad
from torch.nn.functional import softplus

__all__ = [
    'AbsTransform',
    'AffineTransform',
    'CatTransform',
    'ComposeTransform',
    'ExpTransform',
    'LowerCholeskyTransform',
    'PowerTransform',
    'SigmoidTransform',
    'TanhTransform',
    'SoftmaxTransform',
    'StackTransform',
    'StickBreakingTransform',
    'Transform',
    'identity_transform',
]


class Transform(object):
    """
    Abstract class for invertable transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for transforms whose inverses are either expensive or
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
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
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
        super(Transform, self).__init__()

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

    @property
    def sign(self):
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
        raise NotImplementedError

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        if type(self).__init__ is Transform.__init__:
            return type(self)(cache_size=cache_size)
        raise NotImplementedError("{}.with_cache is not implemented".format(type(self)))

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

    def __repr__(self):
        return self.__class__.__name__ + '()'


class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """
    def __init__(self, transform):
        super(_InverseTransform, self).__init__(cache_size=transform._cache_size)
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
    def sign(self):
        return self._inv.sign

    @property
    def event_dim(self):
        return self._inv.event_dim

    @property
    def inv(self):
        return self._inv

    def with_cache(self, cache_size=1):
        return self.inv.with_cache(cache_size).inv

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
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """
    def __init__(self, parts, cache_size=0):
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        super(ComposeTransform, self).__init__(cache_size=cache_size)
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
    def sign(self):
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

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

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ComposeTransform(self.parts, cache_size=cache_size)

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        if not self.parts:
            return torch.zeros_like(x)
        result = 0
        for part in self.parts[:-1]:
            y_tmp = part(x)
            result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y_tmp),
                                             self.event_dim - part.event_dim)
            x = y_tmp
        part = self.parts[-1]
        result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                         self.event_dim - part.event_dim)
        return result

    def __repr__(self):
        fmt_string = self.__class__.__name__ + '(\n    '
        fmt_string += ',\n    '.join([p.__repr__() for p in self.parts])
        fmt_string += '\n)'
        return fmt_string


identity_transform = ComposeTransform([])


class ExpTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \exp(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    def _call(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x


class PowerTransform(Transform):
    r"""
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    """
    domain = constraints.positive
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __init__(self, exponent, cache_size=0):
        super(PowerTransform, self).__init__(cache_size=cache_size)
        self.exponent, = broadcast_all(exponent)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    def __eq__(self, other):
        if not isinstance(other, PowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
        return x.pow(self.exponent)

    def _inverse(self, y):
        return y.pow(1 / self.exponent)

    def log_abs_det_jacobian(self, x, y):
        return (self.exponent * y / x).abs().log()


def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)


class SigmoidTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """
    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        return _clipped_sigmoid(x)

    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1. - finfo.eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x) - F.softplus(x)


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


class AbsTransform(Transform):
    r"""
    Transform via the mapping :math:`y = |x|`.
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
    r"""
    Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super(AffineTransform, self).__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self.event_dim = event_dim

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(self.loc, self.scale, self.event_dim, cache_size=cache_size)

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False

        if isinstance(self.loc, numbers.Number) and isinstance(other.loc, numbers.Number):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():
                return False

        if isinstance(self.scale, numbers.Number) and isinstance(other.scale, numbers.Number):
            if self.scale != other.scale:
                return False
        else:
            if not (self.scale == other.scale).all().item():
                return False

        return True

    @property
    def sign(self):
        if isinstance(self.scale, numbers.Number):
            return 1 if self.scale > 0 else -1 if self.scale < 0 else 0
        return self.scale.sign()

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        if isinstance(scale, numbers.Number):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            result = torch.abs(scale).log()
        if self.event_dim:
            result_size = result.size()[:-self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)


class SoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real
    codomain = constraints.simplex
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        return probs / probs.sum(-1, True)

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
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, (0, 1), value=1) * pad(z_cumprod, (1, 0), value=1)
        return y

    def _inverse(self, y):
        y_crop = y[..., :-1]
        offset = y.shape[-1] - y.new_ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1 - y_crop.cumsum(-1)
        # we clamp to make sure that sf is positive which sometimes does not
        # happen when y[-1] ~ 0 or y[:-1].sum() ~ 1
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        x = y_crop.log() - sf.log() + offset.log()
        return x

    def log_abs_det_jacobian(self, x, y):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        # use the identity 1 - sigmoid(x) = exp(-x) * sigmoid(x)
        detJ = (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)
        return detJ


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
        return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

    def _inverse(self, y):
        return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


class CatTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::
       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """
    def __init__(self, tseq, dim=0, lengths=None, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super(CatTransform, self).__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.transforms)
        self.dim = dim

    @lazy_property
    def length(self):
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.tseq, self.dim, self.lengths, cache_size)

    def _call(self, x):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == self.length
        yslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslices.append(trans(xslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(yslices, dim=self.dim)

    def _inverse(self, y):
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == self.length
        xslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            yslice = y.narrow(self.dim, start, length)
            xslices.append(trans.inv(yslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == self.length
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == self.length
        logdetjacs = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(logdetjacs, dim=self.dim)

    @property
    def bijective(self):
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    def domain(self):
        return constraints.cat([t.domain for t in self.transforms],
                               self.dim, self.lengths)

    @constraints.dependent_property
    def codomain(self):
        return constraints.cat([t.codomain for t in self.transforms],
                               self.dim, self.lengths)


class StackTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::
       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    """
    def __init__(self, tseq, dim=0, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super(StackTransform, self).__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        self.dim = dim

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StackTransform(self.transforms, self.dim, cache_size)

    def _slice(self, z):
        return [z.select(self.dim, i) for i in range(z.size(self.dim))]

    def _call(self, x):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == len(self.transforms)
        yslices = []
        for xslice, trans in zip(self._slice(x), self.transforms):
            yslices.append(trans(xslice))
        return torch.stack(yslices, dim=self.dim)

    def _inverse(self, y):
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == len(self.transforms)
        xslices = []
        for yslice, trans in zip(self._slice(y), self.transforms):
            xslices.append(trans.inv(yslice))
        return torch.stack(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == len(self.transforms)
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == len(self.transforms)
        logdetjacs = []
        yslices = self._slice(y)
        xslices = self._slice(x)
        for xslice, yslice, trans in zip(xslices, yslices, self.transforms):
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
        return torch.stack(logdetjacs, dim=self.dim)

    @property
    def bijective(self):
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    def domain(self):
        return constraints.stack([t.domain for t in self.transforms], self.dim)

    @constraints.dependent_property
    def codomain(self):
        return constraints.stack([t.codomain for t in self.transforms], self.dim)
