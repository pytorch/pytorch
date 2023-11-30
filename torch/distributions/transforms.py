import functools
import math
import numbers
import operator
import weakref
from typing import List

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
    _sum_rightmost,
    broadcast_all,
    lazy_property,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.nn.functional import pad, softplus

__all__ = [
    "AbsTransform",
    "AffineTransform",
    "CatTransform",
    "ComposeTransform",
    "CorrCholeskyTransform",
    "CumulativeDistributionTransform",
    "ExpTransform",
    "IndependentTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "PowerTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SoftplusTransform",
    "TanhTransform",
    "SoftmaxTransform",
    "StackTransform",
    "StickBreakingTransform",
    "Transform",
    "identity_transform",
]


class Transform:
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
    """

    bijective = False
    domain: constraints.Constraint
    codomain: constraints.Constraint

    def __init__(self, cache_size=0):
        self._cache_size = cache_size
        self._inv = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
        else:
            raise ValueError("cache_size must be 0 or 1")
        super().__init__()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inv"] = None
        return state

    @property
    def event_dim(self):
        if self.domain.event_dim == self.codomain.event_dim:
            return self.domain.event_dim
        raise ValueError("Please use either .domain.event_dim or .codomain.event_dim")

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
        raise NotImplementedError(f"{type(self)}.with_cache is not implemented")

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
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape):
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape):
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape


class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """

    def __init__(self, transform: Transform):
        super().__init__(cache_size=transform._cache_size)
        self._inv: Transform = transform

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        assert self._inv is not None
        return self._inv.codomain

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        assert self._inv is not None
        return self._inv.domain

    @property
    def bijective(self):
        assert self._inv is not None
        return self._inv.bijective

    @property
    def sign(self):
        assert self._inv is not None
        return self._inv.sign

    @property
    def inv(self):
        return self._inv

    def with_cache(self, cache_size=1):
        assert self._inv is not None
        return self.inv.with_cache(cache_size).inv

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        assert self._inv is not None
        return self._inv == other._inv

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._inv)})"

    def __call__(self, x):
        assert self._inv is not None
        return self._inv._inv_call(x)

    def log_abs_det_jacobian(self, x, y):
        assert self._inv is not None
        return -self._inv.log_abs_det_jacobian(y, x)

    def forward_shape(self, shape):
        return self._inv.inverse_shape(shape)

    def inverse_shape(self, shape):
        return self._inv.forward_shape(shape)


class ComposeTransform(Transform):
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """

    def __init__(self, parts: List[Transform], cache_size=0):
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        super().__init__(cache_size=cache_size)
        self.parts = parts

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.parts == other.parts

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        if not self.parts:
            return constraints.real
        domain = self.parts[0].domain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[-1].codomain.event_dim
        for part in reversed(self.parts):
            event_dim += part.domain.event_dim - part.codomain.event_dim
            event_dim = max(event_dim, part.domain.event_dim)
        assert event_dim >= domain.event_dim
        if event_dim > domain.event_dim:
            domain = constraints.independent(domain, event_dim - domain.event_dim)
        return domain

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if not self.parts:
            return constraints.real
        codomain = self.parts[-1].codomain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[0].domain.event_dim
        for part in self.parts:
            event_dim += part.codomain.event_dim - part.domain.event_dim
            event_dim = max(event_dim, part.codomain.event_dim)
        assert event_dim >= codomain.event_dim
        if event_dim > codomain.event_dim:
            codomain = constraints.independent(codomain, event_dim - codomain.event_dim)
        return codomain

    @lazy_property
    def bijective(self):
        return all(p.bijective for p in self.parts)

    @lazy_property
    def sign(self):
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

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

        # Compute intermediates. This will be free if parts[:-1] are all cached.
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)

        terms = []
        event_dim = self.domain.event_dim
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            terms.append(
                _sum_rightmost(
                    part.log_abs_det_jacobian(x, y), event_dim - part.domain.event_dim
                )
            )
            event_dim += part.codomain.event_dim - part.domain.event_dim
        return functools.reduce(operator.add, terms)

    def forward_shape(self, shape):
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape):
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def __repr__(self):
        fmt_string = self.__class__.__name__ + "(\n    "
        fmt_string += ",\n    ".join([p.__repr__() for p in self.parts])
        fmt_string += "\n)"
        return fmt_string


identity_transform = ComposeTransform([])


class IndependentTransform(Transform):
    """
    Wrapper around another transform to treat
    ``reinterpreted_batch_ndims``-many extra of the right most dimensions as
    dependent. This has no effect on the forward or backward transforms, but
    does sum out ``reinterpreted_batch_ndims``-many of the rightmost dimensions
    in :meth:`log_abs_det_jacobian`.

    Args:
        base_transform (:class:`Transform`): A base transform.
        reinterpreted_batch_ndims (int): The number of extra rightmost
            dimensions to treat as dependent.
    """

    def __init__(self, base_transform, reinterpreted_batch_ndims, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(
            self.base_transform, self.reinterpreted_batch_ndims, cache_size=cache_size
        )

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(
            self.base_transform.domain, self.reinterpreted_batch_ndims
        )

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(
            self.base_transform.codomain, self.reinterpreted_batch_ndims
        )

    @property
    def bijective(self):
        return self.base_transform.bijective

    @property
    def sign(self):
        return self.base_transform.sign

    def _call(self, x):
        if x.dim() < self.domain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform(x)

    def _inverse(self, y):
        if y.dim() < self.codomain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = _sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})"

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)


class ReshapeTransform(Transform):
    """
    Unit Jacobian transform to reshape the rightmost part of a tensor.

    Note that ``in_shape`` and ``out_shape`` must have the same number of
    elements, just as for :meth:`torch.Tensor.reshape`.

    Arguments:
        in_shape (torch.Size): The input event shape.
        out_shape (torch.Size): The output event shape.
    """

    bijective = True

    def __init__(self, in_shape, out_shape, cache_size=0):
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        if self.in_shape.numel() != self.out_shape.numel():
            raise ValueError("in_shape, out_shape have different numbers of elements")
        super().__init__(cache_size=cache_size)

    @constraints.dependent_property
    def domain(self):
        return constraints.independent(constraints.real, len(self.in_shape))

    @constraints.dependent_property
    def codomain(self):
        return constraints.independent(constraints.real, len(self.out_shape))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ReshapeTransform(self.in_shape, self.out_shape, cache_size=cache_size)

    def _call(self, x):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(batch_shape + self.out_shape)

    def _inverse(self, y):
        batch_shape = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(batch_shape + self.in_shape)

    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.new_zeros(batch_shape)

    def forward_shape(self, shape):
        if len(shape) < len(self.in_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.in_shape)
        if shape[cut:] != self.in_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.in_shape}"
            )
        return shape[:cut] + self.out_shape

    def inverse_shape(self, shape):
        if len(shape) < len(self.out_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.out_shape)
        if shape[cut:] != self.out_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.out_shape}"
            )
        return shape[:cut] + self.in_shape


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

    def __init__(self, exponent, cache_size=0):
        super().__init__(cache_size=cache_size)
        (self.exponent,) = broadcast_all(exponent)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    @lazy_property
    def sign(self):
        return self.exponent.sign()

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

    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))


def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


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
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x) - F.softplus(x)


class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return (-y).expm1().neg().log() + y

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)


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
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))


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
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def event_dim(self):
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(
            self.loc, self.scale, self.event_dim, cache_size=cache_size
        )

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False

        if isinstance(self.loc, numbers.Number) and isinstance(
            other.loc, numbers.Number
        ):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():
                return False

        if isinstance(self.scale, numbers.Number) and isinstance(
            other.scale, numbers.Number
        ):
            if self.scale != other.scale:
                return False
        else:
            if not (self.scale == other.scale).all().item():
                return False

        return True

    @property
    def sign(self):
        if isinstance(self.scale, numbers.Real):
            return 1 if float(self.scale) > 0 else -1 if float(self.scale) < 0 else 0
        return self.scale.sign()

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        if isinstance(scale, numbers.Real):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            result = torch.abs(scale).log()
        if self.event_dim:
            result_size = result.size()[: -self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[: -self.event_dim]
        return result.expand(shape)

    def forward_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )


class CorrCholeskyTransform(Transform):
    r"""
    Transforms an uncontrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert x into a lower triangular matrix in row order.
        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
           class :class:`StickBreakingTransform` to transform :math:`X_i` into a
           unit Euclidean length vector using the following steps:
           - Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \tanh(X_i)`.
           - Transforms into an unsigned domain: :math:`z_i = r_i^2`.
           - Applies :math:`s_i = StickBreakingTransform(z_i)`.
           - Transforms back into signed domain: :math:`y_i = sign(r_i) * \sqrt{s_i}`.
    """
    domain = constraints.real_vector
    codomain = constraints.corr_cholesky
    bijective = True

    def _call(self, x):
        x = torch.tanh(x)
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(min=-1 + eps, max=1 - eps)
        r = vec_to_tril_matrix(x, diag=-1)
        # apply stick-breaking on the squared values
        # Note that y = sign(r) * sqrt(z * z1m_cumprod)
        #             = (sign(r) * sqrt(z)) * sqrt(z1m_cumprod) = r * sqrt(z1m_cumprod)
        z = r**2
        z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
        # Diagonal elements must be 1.
        r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
        y = r * pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)
        return y

    def _inverse(self, y):
        # inverse stick-breaking
        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y_cumsum = 1 - torch.cumsum(y * y, dim=-1)
        y_cumsum_shifted = pad(y_cumsum[..., :-1], [1, 0], value=1)
        y_vec = tril_matrix_to_vec(y, diag=-1)
        y_cumsum_vec = tril_matrix_to_vec(y_cumsum_shifted, diag=-1)
        t = y_vec / (y_cumsum_vec).sqrt()
        # inverse of tanh
        x = (t.log1p() - t.neg().log1p()) / 2
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. We return `log_abs_det_jacobian` of `x` and the
        # flattened lower triangular part of `y`.

        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y1m_cumsum = 1 - (y * y).cumsum(dim=-1)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # also works for 2 x 2 matrix
        y1m_cumsum_tril = tril_matrix_to_vec(y1m_cumsum, diag=-2)
        stick_breaking_logdet = 0.5 * (y1m_cumsum_tril).log().sum(-1)
        tanh_logdet = -2 * (x + softplus(-2 * x) - math.log(2.0)).sum(dim=-1)
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        # Reshape from (..., N) to (..., D, D).
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        N = shape[-1]
        D = round((0.25 + 2 * N) ** 0.5 + 0.5)
        if D * (D - 1) // 2 != N:
            raise ValueError("Input is not a flattend lower-diagonal number")
        return shape[:-1] + (D, D)

    def inverse_shape(self, shape):
        # Reshape from (..., D, D) to (..., N).
        if len(shape) < 2:
            raise ValueError("Too few dimensions on input")
        if shape[-2] != shape[-1]:
            raise ValueError("Input is not square")
        D = shape[-1]
        N = D * (D - 1) // 2
        return shape[:-2] + (N,)


class SoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        return probs / probs.sum(-1, True)

    def _inverse(self, y):
        probs = y
        return probs.log()

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


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

    domain = constraints.real_vector
    codomain = constraints.simplex
    bijective = True

    def __eq__(self, other):
        return isinstance(other, StickBreakingTransform)

    def _call(self, x):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, [0, 1], value=1) * pad(z_cumprod, [1, 0], value=1)
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

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)


class LowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        return isinstance(other, LowerCholeskyTransform)

    def _call(self, x):
        return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

    def _inverse(self, y):
        return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


class PositiveDefiniteTransform(Transform):
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.positive_definite  # type: ignore[assignment]

    def __eq__(self, other):
        return isinstance(other, PositiveDefiniteTransform)

    def _call(self, x):
        x = LowerCholeskyTransform()(x)
        return x @ x.mT

    def _inverse(self, y):
        y = torch.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y)


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

    transforms: List[Transform]

    def __init__(self, tseq, dim=0, lengths=None, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.transforms)
        self.dim = dim

    @lazy_property
    def event_dim(self):
        return max(t.event_dim for t in self.transforms)

    @lazy_property
    def length(self):
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.transforms, self.dim, self.lengths, cache_size)

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
            logdetjac = trans.log_abs_det_jacobian(xslice, yslice)
            if trans.event_dim < self.event_dim:
                logdetjac = _sum_rightmost(logdetjac, self.event_dim - trans.event_dim)
            logdetjacs.append(logdetjac)
            start = start + length  # avoid += for jit compat
        # Decide whether to concatenate or sum.
        dim = self.dim
        if dim >= 0:
            dim = dim - x.dim()
        dim = dim + self.event_dim
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

    @property
    def bijective(self):
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    def domain(self):
        return constraints.cat(
            [t.domain for t in self.transforms], self.dim, self.lengths
        )

    @constraints.dependent_property
    def codomain(self):
        return constraints.cat(
            [t.codomain for t in self.transforms], self.dim, self.lengths
        )


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

    transforms: List[Transform]

    def __init__(self, tseq, dim=0, cache_size=0):
        assert all(isinstance(t, Transform) for t in tseq)
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
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


class CumulativeDistributionTransform(Transform):
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.

    Example::

        # Construct a Gaussian copula from a multivariate normal.
        base_dist = MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=LKJCholesky(2).sample(),
        )
        transform = CumulativeDistributionTransform(Normal(0, 1))
        copula = TransformedDistribution(base_dist, [transform])
    """

    bijective = True
    codomain = constraints.unit_interval
    sign = +1

    def __init__(self, distribution, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.distribution = distribution

    @property
    def domain(self):
        return self.distribution.support

    def _call(self, x):
        return self.distribution.cdf(x)

    def _inverse(self, y):
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)
