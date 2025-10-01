r"""
The following constraints are implemented. For parameter-free constraints,
a canonical instance is provided, otherwise a lower cased alias.

- value constraints
    - :class:`Boolean`, with canonical instance :data:`boolean`
    - :class:`Real`, with canonical instance :data:`real`

- continuous interval constraints
    - :class:`Interval`, with alias :meth:`interval`
    - :class:`HalfOpenInterval`, with alias :meth:`half_open_interval`
    - :class:`GreaterThan`, with alias :meth:`greater_than`
    - :class:`GreaterThanEq`, with alias :meth:`greater_than_eq`
    - :class:`LessThan`, with alias :meth:`less_than`
    - :class:`UnitInterval`, with canonical instance :data:`unit_interval`
    - :class:`Positive`, with canonical instance :data:`positive`
    - :class:`Negative`, with canonical instance :data:`negative`
    - :class:`NonNegative`, with canonical instance :data:`nonnegative`

- discrete interval constraints
    - :class:`IntegerInterval`, with alias :meth:`integer_interval`
    - :class:`IntegerGreaterThan`, with alias :meth:`greater_than`
    - :class:`IntegerLessThan`, with alias :meth:`less_than`
    - :class:`PositiveInteger`, with canonical instance :data:`positive_integer`
    - :class:`NonNegativeInteger`, with canonical instance :data:`nonnegative_integer`

- vector constraints (``event_dim=1``)
    - :class:`Multinomial`, with alias :meth:`multinomial`
    - :class:`OneHot`, with canonical instance :data:`one_hot`
    - :class:`RealVector`, with canonical instance :data:`real_vector`
    - :class:`Simplex`, with canonical instance :data:`simplex`

- matrix constraints (``event_dim=2``)
    - :class:`Square`, with canonical instance :data:`square`
    - :class:`Symmetric`, with canonical instance :data:`symmetric`
    - :class:`PositiveDefinite`, with canonical instance :data:`positive_definite`
    - :class:`PositiveSemidefinite`, with canonical instance :data:`positive_semidefinite`
    - :class:`LowerCholesky`, with canonical instance :data:`lower_cholesky`
    - :class:`LowerTriangular`, with canonical instance :data:`lower_triangular`
    - :class:`CorrCholesky`,  canonical instance :data:`corr_cholesky`

- generic constraints
    - :class:`Independent`, with alias :meth:`independent`
    - :class:`Cat`, with alias :meth:`cat`
    - :class:`Stack`, with alias :meth:`stack`
    - :class:`MixtureSameFamilyConstraint`, with alias :meth:`mixture_same_family`

- dependent constraints
    - :class:`Dependent`, with canonical instance :data:`dependent`
    - :class:`DependentProperty`, with alias :meth:`dependent_property`

"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import (
    Callable,
    ClassVar,
    Final,
    Generic,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Self, TypeIs

import torch
from torch import Tensor


__all__ = [
    # Base Classes
    "Constraint",
    # Dependent Constraints
    "Dependent",
    "DependentProperty",
    "is_dependent",
    # Classes
    "Boolean",
    "Cat",
    "CorrCholesky",
    "GreaterThan",
    "GreaterThanEq",
    "HalfOpenInterval",
    "Independent",
    "IntegerGreaterThan",
    "IntegerInterval",
    "IntegerLessThan",
    "Interval",
    "LessThan",
    "LowerCholesky",
    "LowerTriangular",
    "MixtureSameFamilyConstraint",
    "Multinomial",
    "Negative",
    "NonNegative",
    "NonNegativeInteger",
    "OneHot",
    "Positive",
    "PositiveDefinite",
    "PositiveInteger",
    "PositiveSemidefinite",
    "Real",
    "RealVector",
    "Simplex",
    "Square",
    "Stack",
    "Symmetric",
    "UnitInterval",
    # Constraint Classes / Instances
    "boolean",
    "corr_cholesky",
    "dependent",
    "lower_cholesky",
    "lower_triangular",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "positive",
    "positive_definite",
    "positive_integer",
    "positive_semidefinite",
    "real",
    "real_vector",
    "simplex",
    "square",
    "symmetric",
    "unit_interval",
    # aliases
    "cat",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "half_open_interval",
    "independent",
    "integer_interval",
    "interval",
    "less_than",
    "mixture_same_family",
    "multinomial",
    "stack",
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

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def event_dim(self) -> int:
        return 0

    @abstractmethod
    def check(self, value: Tensor, /) -> Tensor:
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# dependent constraints ----------------------------------------------------------------


class Dependent(Constraint):
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

    def __init__(
        self, *, is_discrete: bool = NotImplemented, event_dim: int = NotImplemented
    ) -> None:
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        if self._is_discrete is NotImplemented:
            raise NotImplementedError(".is_discrete cannot be determined statically")
        return self._is_discrete

    @property
    def event_dim(self) -> int:
        if self._event_dim is NotImplemented:
            raise NotImplementedError(".event_dim cannot be determined statically")
        return self._event_dim

    def __call__(
        self, *, is_discrete: bool = NotImplemented, event_dim: int = NotImplemented
    ) -> "Dependent":
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return Dependent(is_discrete=is_discrete, event_dim=event_dim)

    def check(self, x: Tensor) -> Tensor:
        raise ValueError("Cannot determine validity of dependent constraint")


def is_dependent(constraint: Constraint) -> TypeIs[Dependent]:
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
    return isinstance(constraint, Dependent)


T = TypeVar("T", covariant=True)
R = TypeVar("R", covariant=True)


class DependentProperty(property, Dependent, Generic[T, R]):
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
        fn: Optional[Callable[[T], R]] = None,
        *,
        is_discrete: bool = NotImplemented,
        event_dim: int = NotImplemented,
    ) -> None:
        property.__init__(self, fn)
        Dependent.__init__(self, is_discrete=is_discrete, event_dim=event_dim)

    if TYPE_CHECKING:
        # Needed because subclassing property is not fully supported in mypy
        @overload  # type: ignore[no-overload-impl]
        def __get__(self, instance: None, owner: type, /) -> Self: ...
        @overload
        def __get__(self, instance: T, owner: Optional[type] = None, /) -> R: ...  # type: ignore[misc]

    _T = TypeVar("_T", contravariant=True)
    _R = TypeVar("_R", covariant=True)

    def __call__(
        self,
        fn: Optional[Callable[[_T], _R]] = None,
        /,
        *,
        is_discrete: bool = NotImplemented,
        event_dim: int = NotImplemented,
    ) -> "DependentProperty[_T, _R]":
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self): ...
        """
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return DependentProperty(fn, is_discrete=is_discrete, event_dim=event_dim)


# generic constraints ------------------------------------------------------------------
Con = TypeVar("Con", bound=Constraint, covariant=True)


class Independent(Constraint, Generic[Con]):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraint: Con, reinterpreted_batch_ndims: int) -> None:
        assert isinstance(base_constraint, Constraint)
        assert isinstance(reinterpreted_batch_ndims, int)
        assert reinterpreted_batch_ndims >= 0

        self.base_constraint: Final[Con] = base_constraint
        self.reinterpreted_batch_ndims: Final[int] = reinterpreted_batch_ndims
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value: Tensor) -> Tensor:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base_constraint!r}, {self.reinterpreted_batch_ndims})"


class MixtureSameFamilyConstraint(Constraint, Generic[Con]):
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

    def __init__(self, base_constraint: Con) -> None:
        assert isinstance(base_constraint, Constraint)
        self.base_constraint: Final[Con] = base_constraint
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:
        return self.base_constraint.event_dim

    def check(self, value: Tensor) -> Tensor:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base_constraint!r})"


class Cat(Constraint, Generic[Con]):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """

    def __init__(
        self,
        cseq: Sequence[Con],
        dim: int = 0,
        lengths: Optional[Sequence[int]] = None,
    ) -> None:
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq: Final[list[Con]] = list(cseq)
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths: Final[list[int]] = list(lengths)
        assert len(self.lengths) == len(self.cseq)
        self.dim: Final[int] = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:
        return max(c.event_dim for c in self.cseq)

    def check(self, value: Tensor) -> Tensor:
        assert -value.dim() <= self.dim < value.dim()
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length  # avoid += for jit compat
        return torch.cat(checks, self.dim)


class Stack(Constraint, Generic[Con]):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """

    def __init__(self, cseq: Sequence[Con], dim: int = 0) -> None:
        assert all(isinstance(c, Constraint) for c in cseq)
        self.cseq: Final[list[Con]] = list(cseq)
        self.dim: Final[int] = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:
        dim = max(c.event_dim for c in self.cseq)
        if self.dim + dim < 0:
            dim += 1
        return dim

    def check(self, value: Tensor) -> Tensor:
        assert -value.dim() <= self.dim < value.dim()
        vs = [value.select(self.dim, i) for i in range(value.size(self.dim))]
        return torch.stack(
            [constr.check(v) for v, constr in zip(vs, self.cseq)], self.dim
        )


# value constraints --------------------------------------------------------------------


class Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """

    is_discrete: ClassVar[bool] = True

    def check(self, value: Tensor) -> Tensor:
        return (value == 0) | (value == 1)


class Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """

    def check(self, value: Tensor) -> Tensor:
        return value == value  # False for NANs.


# interval constraints -------------------------------------------------------------------


class IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """

    is_discrete: ClassVar[bool] = True

    def __init__(
        self, lower_bound: Union[int, Tensor], upper_bound: Union[int, Tensor]
    ) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        self.upper_bound: Final[Union[float, Tensor]] = upper_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return (
            (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)
        )

    def __repr__(self) -> str:
        lower_bound, upper_bound = self.lower_bound, self.upper_bound
        return f"{self.__class__.__name__}({lower_bound=}, {upper_bound=})"


class IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """

    is_discrete: ClassVar[bool] = True

    def __init__(self, upper_bound: Union[int, Tensor]) -> None:
        self.upper_bound: Final[Union[float, Tensor]] = upper_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return (value % 1 == 0) & (value <= self.upper_bound)

    def __repr__(self) -> str:
        upper_bound = self.upper_bound
        return f"{self.__class__.__name__}({upper_bound=})"


class IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """

    is_discrete: ClassVar[bool] = True

    def __init__(self, lower_bound: Union[int, Tensor]) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return (value % 1 == 0) & (value >= self.lower_bound)

    def __repr__(self) -> str:
        lower_bound = self.lower_bound
        return f"{self.__class__.__name__}({lower_bound=})"


class NonNegativeInteger(IntegerGreaterThan):
    def __init__(self) -> None:
        super().__init__(lower_bound=0)


class PositiveInteger(IntegerGreaterThan):
    def __init__(self) -> None:
        super().__init__(lower_bound=1)


class Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """

    def __init__(
        self, lower_bound: Union[float, Tensor], upper_bound: Union[float, Tensor]
    ) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        self.upper_bound: Final[Union[float, Tensor]] = upper_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self) -> str:
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        return f"{self.__class__.__name__}({lower_bound=}, {upper_bound=})"


class HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """

    def __init__(
        self, lower_bound: Union[float, Tensor], upper_bound: Union[float, Tensor]
    ) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        self.upper_bound: Final[Union[float, Tensor]] = upper_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return (self.lower_bound <= value) & (value < self.upper_bound)

    def __repr__(self) -> str:
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        return f"{self.__class__.__name__}({lower_bound=}, {upper_bound=})"


class UnitInterval(Interval):
    """
    Constrain to the unit interval `[0, 1]`.
    """

    def __init__(self) -> None:
        super().__init__(lower_bound=0.0, upper_bound=1.0)


class GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """

    def __init__(self, lower_bound: Union[float, Tensor]) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return self.lower_bound < value

    def __repr__(self) -> str:
        lower_bound = self.lower_bound
        return f"{self.__class__.__name__}({lower_bound=})"


class GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf]`.
    """

    def __init__(self, lower_bound: Union[float, Tensor]) -> None:
        self.lower_bound: Final[Union[float, Tensor]] = lower_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return self.lower_bound <= value

    def __repr__(self) -> str:
        lower_bound = self.lower_bound
        return f"{self.__class__.__name__}({lower_bound=})"


class LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """

    def __init__(self, upper_bound: Union[float, Tensor]) -> None:
        self.upper_bound: Final[Union[float, Tensor]] = upper_bound
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return value < self.upper_bound

    def __repr__(self) -> str:
        upper_bound = self.upper_bound
        return f"{self.__class__.__name__}({upper_bound=})"


class Positive(GreaterThan):
    def __init__(self) -> None:
        super().__init__(lower_bound=0.0)


class NonNegative(GreaterThanEq):
    def __init__(self) -> None:
        super().__init__(lower_bound=0.0)


class Negative(LessThan):
    def __init__(self) -> None:
        super().__init__(upper_bound=0.0)


# vector constraints (event_dim=1) -----------------------------------------------------


class RealVector(Independent[Real]):
    """
    Constrain to a real vector.
    """

    def __init__(self) -> None:
        super().__init__(Real(), reinterpreted_batch_ndims=1)


class OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """

    is_discrete: ClassVar[bool] = True
    event_dim: ClassVar[int] = 1

    def check(self, value: Tensor) -> Tensor:
        is_boolean = (value == 0) | (value == 1)
        is_normalized = value.sum(-1).eq(1)
        return is_boolean.all(-1) & is_normalized


class Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim: ClassVar[int] = 1

    def check(self, value: Tensor) -> Tensor:
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)


class Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """

    is_discrete: ClassVar[bool] = True
    event_dim: ClassVar[int] = 1

    def __init__(self, upper_bound: Union[int, Tensor]) -> None:
        self.upper_bound = upper_bound

    def check(self, x: Tensor) -> Tensor:
        return (x >= 0).all(dim=-1) & (x.sum(dim=-1) <= self.upper_bound)


# matrix constraints (event_dim=2) -----------------------------------------------------


class LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """

    event_dim: ClassVar[int] = 2

    def check(self, value: Tensor) -> Tensor:
        value_tril = value.tril()
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]


class LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """

    event_dim: ClassVar[int] = 2

    def check(self, value: Tensor) -> Tensor:
        value_tril = value.tril()
        lower_triangular = (
            (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        )

        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        return lower_triangular & positive_diagonal


class CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """

    event_dim: ClassVar[int] = 2

    def check(self, value: Tensor) -> Tensor:
        tol = (
            torch.finfo(value.dtype).eps * value.size(-1) * 10
        )  # 10 is an adjustable fudge factor
        row_norm = torch.linalg.norm(value.detach(), dim=-1)
        unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim=-1)
        return LowerCholesky().check(value) & unit_row_norm


class Square(Constraint):
    """
    Constrain to square matrices.
    """

    event_dim: ClassVar[int] = 2

    def check(self, value: Tensor) -> Tensor:
        return torch.full(
            size=value.shape[:-2],
            fill_value=(value.shape[-2] == value.shape[-1]),
            dtype=torch.bool,
            device=value.device,
        )


class Symmetric(Square):
    """
    Constrain to Symmetric square matrices.
    """

    def check(self, value: Tensor) -> Tensor:
        square_check = super().check(value)
        if not square_check.all():
            return square_check
        return torch.isclose(value, value.mT, atol=1e-6).all(-2).all(-1)


class PositiveSemidefinite(Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """

    def check(self, value: Tensor) -> Tensor:
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.eigvalsh(value).ge(0).all(-1)


class PositiveDefinite(Symmetric):
    """
    Constrain to positive-definite matrices.
    """

    def check(self, value: Tensor) -> Tensor:
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.cholesky_ex(value).info.eq(0)


# canonical instances
# dependent constraints
dependent: Final[Dependent] = Dependent()
# value constraints
boolean: Final[Boolean] = Boolean()
real: Final[Real] = Real()
# interval constraints
nonnegative: Final[NonNegative] = NonNegative()
nonnegative_integer: Final[NonNegativeInteger] = NonNegativeInteger()
positive: Final[Positive] = Positive()
positive_integer: Final[PositiveInteger] = PositiveInteger()
unit_interval: Final[UnitInterval] = UnitInterval()
# vector constraints
one_hot: Final[OneHot] = OneHot()
real_vector: Final[RealVector] = RealVector()
simplex: Final[Simplex] = Simplex()
# matrix constraints
corr_cholesky: Final[CorrCholesky] = CorrCholesky()
lower_cholesky: Final[LowerCholesky] = LowerCholesky()
lower_triangular: Final[LowerTriangular] = LowerTriangular()
positive_definite: Final[PositiveDefinite] = PositiveDefinite()
positive_semidefinite: Final[PositiveSemidefinite] = PositiveSemidefinite()
square: Final[Square] = Square()
symmetric: Final[Symmetric] = Symmetric()


# aliases
dependent_property = DependentProperty
# generic constraints
cat = Cat
independent = Independent
mixture_same_family = MixtureSameFamilyConstraint
stack = Stack
# interval constraints
interval = Interval
integer_interval = IntegerInterval
half_open_interval = HalfOpenInterval
greater_than = GreaterThan
greater_than_eq = GreaterThanEq
less_than = LessThan
# vector constraints
multinomial = Multinomial
