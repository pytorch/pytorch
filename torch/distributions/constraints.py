# mypy: allow-untyped-defs

from typing import Any, Callable, Optional


r"""
The following constraints are implemented:

- ``constraints.boolean``
- ``constraints.cat``
- ``constraints.corr_cholesky``
- ``constraints.dependent``
- ``constraints.greater_than(lower_bound)``
- ``constraints.greater_than_eq(lower_bound)``
- ``constraints.independent(constraint, reinterpreted_batch_ndims)``
- ``constraints.integer_interval(lower_bound, upper_bound)``
- ``constraints.interval(lower_bound, upper_bound)``
- ``constraints.less_than(upper_bound)``
- ``constraints.lower_cholesky``
- ``constraints.lower_triangular``
- ``constraints.MixtureSameFamilyConstraint(base_constraint)``
- ``constraints.multinomial``
- ``constraints.nonnegative``
- ``constraints.nonnegative_integer``
- ``constraints.one_hot``
- ``constraints.positive_integer``
- ``constraints.positive``
- ``constraints.positive_semidefinite``
- ``constraints.positive_definite``
- ``constraints.real_vector``
- ``constraints.real``
- ``constraints.simplex``
- ``constraints.symmetric``
- ``constraints.stack``
- ``constraints.square``
- ``constraints.symmetric``
- ``constraints.unit_interval``
"""

import torch


__all__ = [
    "Constraint",
    "boolean",
    "cat",
    "corr_cholesky",
    "dependent",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "independent",
    "integer_interval",
    "interval",
    "half_open_interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "lower_triangular",
    "MixtureSameFamilyConstraint",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "positive",
    "positive_semidefinite",
    "positive_definite",
    "positive_integer",
    "real",
    "real_vector",
    "simplex",
    "square",
    "stack",
    "symmetric",
    "unit_interval",
]


class Constraint:
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """

    is_discrete = False  # Default to continuous.
    event_dim = 0  # Default to univariate.

    def check(self, value):
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + "()"


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    Args:
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        if self._is_discrete is NotImplemented:
            raise NotImplementedError(".is_discrete cannot be determined statically")
        return self._is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        if self._event_dim is NotImplemented:
            raise NotImplementedError(".event_dim cannot be determined statically")
        return self._event_dim

    def __call__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return _Dependent(is_discrete=is_discrete, event_dim=event_dim)

    def check(self, x):
        raise ValueError("Cannot determine validity of dependent constraint")


def is_dependent(constraint):
    """
    Checks if ``constraint`` is a ``_Dependent`` object.

    Args:
        constraint : A ``Constraint`` object.

    Returns:
        ``bool``: True if ``constraint`` can be refined to the type ``_Dependent``, False otherwise.

    Examples:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch.distributions.constraints import is_dependent

        >>> dist = Bernoulli(probs=torch.tensor([0.6], requires_grad=True))
        >>> constraint1 = dist.arg_constraints["probs"]
        >>> constraint2 = dist.arg_constraints["logits"]

        >>> for constraint in [constraint1, constraint2]:
        >>>     if is_dependent(constraint):
        >>>         continue
    """
    return isinstance(constraint, _Dependent)


class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high

            @constraints.dependent_property(is_discrete=False, event_dim=0)
            def support(self):
                return constraints.interval(self.low, self.high)

    Args:
        fn (Callable): The function to be decorated.
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        is_discrete: Optional[bool] = NotImplemented,
        event_dim: Optional[int] = NotImplemented,
    ) -> None:
        super().__init__(fn)
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, fn: Callable[..., Any]) -> "_DependentProperty":  # type: ignore[override]
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self): ...
        """
        return _DependentProperty(
            fn, is_discrete=self._is_discrete, event_dim=self._event_dim
        )


class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        assert isinstance(base_constraint, Constraint)
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        if result.dim() < self.reinterpreted_batch_ndims:
            expected = self.base_constraint.event_dim + self.reinterpreted_batch_ndims
            raise ValueError(
                f"Expected value.dim() >= {expected} but got {value.dim()}"
            )
        result = result.reshape(
            result.shape[: result.dim() - self.reinterpreted_batch_ndims] + (-1,)
        )
        result = result.all(-1)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__[1:]}({repr(self.base_constraint)}, {self.reinterpreted_batch_ndims})"


class MixtureSameFamilyConstraint(Constraint):
    """
    Constraint for the :class:`~torch.distribution.MixtureSameFamily`
    distribution that adds back the rightmost batch dimension before
    performing the validity check with the component distribution
    constraint.

    Args:
        base_constraint: The ``Constraint`` object of
            the component distribution of
            the :class:`~torch.distribution.MixtureSameFamily` distribution.
    """

    def __init__(self, base_constraint):
        assert isinstance(base_constraint, Constraint)
        self.base_constraint = base_constraint
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return self.base_constraint.event_dim

    def check(self, value):
        """
        Check validity of ``value`` as a possible outcome of sampling
        the :class:`~torch.distribution.MixtureSameFamily` distribution.
        """
        unsqueezed_value = value.unsqueeze(-1 - self.event_dim)
        result = self.base_constraint.check(unsqueezed_value)
        if value.dim() < self.event_dim:
            raise ValueError(
                f"Expected value.dim() >= {self.event_dim} but got {value.dim()}"
            )
        num_dim_to_keep = value.dim() - self.event_dim
        result = result.reshape(result.shape[:num_dim_to_keep] + (-1,))
        result = result.all(-1)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_constraint)})"


class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """

    is_discrete = True

    def check(self, value):
        return (value == 0) | (value == 1)


class _OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """

    is_discrete = True
    event_dim = 1

    def check(self, value):
        is_boolean = (value == 0) | (value == 1)
        is_normalized = value.sum(-1).eq(1)
        return is_boolean.all(-1) & is_normalized


class _IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """

    is_discrete = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (
            (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)
        )

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """

    is_discrete = True

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(upper_bound={self.upper_bound})"
        return fmt_string


class _IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """

    is_discrete = True

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value >= self.lower_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """

    def check(self, value):
        return value == value  # False for NANs.


class _GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return self.lower_bound < value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return self.lower_bound <= value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return value < self.upper_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(upper_bound={self.upper_bound})"
        return fmt_string


class _Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (self.lower_bound <= value) & (value < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim = 1

    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)


class _Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """

    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, x):
        return (x >= 0).all(dim=-1) & (x.sum(dim=-1) <= self.upper_bound)


class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """

    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]


class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """

    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (
            (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        )

        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        return lower_triangular & positive_diagonal


class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """

    event_dim = 2

    def check(self, value):
        tol = (
            torch.finfo(value.dtype).eps * value.size(-1) * 10
        )  # 10 is an adjustable fudge factor
        row_norm = torch.linalg.norm(value.detach(), dim=-1)
        unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim=-1)
        return _LowerCholesky().check(value) & unit_row_norm


class _Square(Constraint):
    """
    Constrain to square matrices.
    """

    event_dim = 2

    def check(self, value):
        return torch.full(
            size=value.shape[:-2],
            fill_value=(value.shape[-2] == value.shape[-1]),
            dtype=torch.bool,
            device=value.device,
        )


class _Symmetric(_Square):
    """
    Constrain to Symmetric square matrices.
    """

    def check(self, value):
        square_check = super().check(value)
        if not square_check.all():
            return square_check
        return torch.isclose(value, value.mT, atol=1e-6).all(-2).all(-1)


class _PositiveSemidefinite(_Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.eigvalsh(value).ge(0).all(-1)


class _PositiveDefinite(_Symmetric):
    """
    Constrain to positive-definite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.cholesky_ex(value).info.eq(0)


class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """

    def __init__(self, cseq, dim=0, lengths=None):
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq = list(cseq)
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.cseq)
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return max(c.event_dim for c in self.cseq)

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length  # avoid += for jit compat
        return torch.cat(checks, self.dim)


class _Stack(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """

    def __init__(self, cseq, dim=0):
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq = list(cseq)
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        dim = max(c.event_dim for c in self.cseq)
        if self.dim + dim < 0:
            dim += 1
        return dim

    def check(self, value):
        assert -value.dim() <= self.dim < value.dim()
        vs = [value.select(self.dim, i) for i in range(value.size(self.dim))]
        return torch.stack(
            [constr.check(v) for v, constr in zip(vs, self.cseq)], self.dim
        )


# Public interface.
dependent = _Dependent()
dependent_property = _DependentProperty
independent = _IndependentConstraint
boolean = _Boolean()
one_hot = _OneHot()
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
integer_interval = _IntegerInterval
real = _Real()
real_vector = independent(real, 1)
positive = _GreaterThan(0.0)
nonnegative = _GreaterThanEq(0.0)
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
multinomial = _Multinomial
unit_interval = _Interval(0.0, 1.0)
interval = _Interval
half_open_interval = _HalfOpenInterval
simplex = _Simplex()
lower_triangular = _LowerTriangular()
lower_cholesky = _LowerCholesky()
corr_cholesky = _CorrCholesky()
square = _Square()
symmetric = _Symmetric()
positive_semidefinite = _PositiveSemidefinite()
positive_definite = _PositiveDefinite()
cat = _Cat
stack = _Stack
