import builtins
import dataclasses
import inspect
import math
import sys
import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._pytree import SUPPORTED_NODES

from .exported_program import ExportedProgram

if TYPE_CHECKING:
    from sympy import Symbol

    from torch._guards import Source

    from ..fx.experimental.symbolic_shapes import ShapeEnv, StrictMinMaxConstraint

__all__ = ["Constraint", "Dim", "dims", "dynamic_dim"]


class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        if min_ == 2:
            min_ = None
        if max_ == sys.maxsize - 1:
            max_ = None
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"

    def __add__(cls, other):
        # e.g., dim + 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to add {other} to {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x + other)

    def __radd__(cls, other):
        return cls + other

    def __sub__(cls, other):
        # e.g., dim - 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to subtract {other} from {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x - other)

    def __rsub__(cls, other):
        raise NotImplementedError(
            f"Attempted to negate {cls.__name__}. "
            "(Only increasing linear operations with integer coefficients are supported.)"
        )

    def __mul__(cls, other):
        # e.g., dim * 2
        if type(other) is not int or other <= 0:
            raise NotImplementedError(
                f"Attempted to multiply {other} with {cls.__name__}, where a positive integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x * other)

    def __rmul__(cls, other):
        return cls * other

    def _derived_name(cls, fn):
        from sympy import sympify

        return str(fn(sympify(cls.__name__)))

    def _derive(cls, fn):
        return _DerivedDim(cls._derived_name(fn), (int,), {"root": cls, "fn": fn})


class _DerivedDim(_Dim):
    """
    Metaclass for derived :func:`Dim` types.

    Currently we only support increasing linear expressions with integer coefficients.
    In other words, a derived Dim can always be written in the form Ax + B, where
    x is a regular Dim (i.e., non-derived Dim), A and B are integers, and A is positive.
    (In particular, the latter ensures that x < y => Ax + B < Ay + B.)
    These restrictions on the form of derived Dims makes the metatheory simpler: e.g.,
    it simplifies computing ranges for derived Dims, solving for underlying regular Dims,
    deciding equalities between derived Dims, and so on.

    The function lambda x: Ax + B is expressed by `fn`, where x is a normal Dim, `root`.
    The range of a derived Dim is computed by mapping `fn` over the range of its `root`.
    """

    @property
    def min(self):
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        _min_symint = self.fn(Integer(self.root.min))  # type: ignore[attr-defined]
        assert _min_symint >= 2, (
            f"Expected derived min value of {self.__name__} to be >= 2. "
            f"Please specify an appropriate min value for {self.root.__name__} "  # type: ignore[attr-defined]
            f"(currently {self.root.min})."  # type: ignore[attr-defined]
        )
        return int(_min_symint)

    @property
    def max(self):
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        _max_symint = self.fn(Integer(self.root.max))  # type: ignore[attr-defined]
        assert _max_symint <= sys.maxsize - 1, (
            f"Expected derived max value of {self.__name__} to be <= {sys.maxsize - 1}. "
            f"Please specify an appropriate max value for {self.root.__name__} "  # type: ignore[attr-defined]
            f"(currently {self.root.max})."  # type: ignore[attr-defined]
        )
        return int(_max_symint)

    def _derive(self, fn):
        # We support nesting, e.g., 2*dim + 1.
        # This is implemented by composing operations on the same root.
        # As a consequence, roots are always regular Dims (i.e., not derived Dims).
        return _DerivedDim(
            self._derived_name(fn),
            (int,),
            {"root": self.root, "fn": lambda x: fn(self.fn(x))},  # type: ignore[attr-defined]
        )


def Dim(name: str, *, min: Optional[int] = None, max: Optional[int] = None):
    """
    :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
    It can be used to describe multiple possible values of a dynamic tensor dimension.
    Note that different dynamic dimensions of the same tensor, or of different tensors,
    can be described by the same type.

    Args:
        name (str): Human-readable name for debugging.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        A type that can be used in dynamic shape specifications for tensors.
    """
    _min = 2 if min is None else builtins.max(min, 2)
    _max = sys.maxsize - 1 if max is None else builtins.min(max, sys.maxsize - 1)
    assert _max > _min, f"Cannot create Dim with inconsistent min={min}, max={max}"
    dim = _Dim(name, (int,), {"min": _min, "max": _max})
    dim.__module__ = getattr(
        inspect.getmodule(inspect.stack()[1][0]), "__name__", "__main__"
    )
    return dim


def dims(*names: str, min: Optional[int] = None, max: Optional[int] = None):
    """
    Util to create multiple :func:`Dim` types.
    """
    return tuple(Dim(name, min=min, max=max) for name in names)


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`dynamic_dim`.
    """

    w_tensor: Any  # weakref to torch.Tensor
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


class _ConstraintFactory(type):
    """
    Metaclass that ensures a private constructor for :class:`_Constraint`
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
            f"Please use torch.export.dynamic_dim() to create one"
        )

    def _create(
        cls, w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
    ):
        return super().__call__(
            w_tensor, t_id, dim, constraint_range, shared, debug_name
        )


def _create_constraint(
    w_tensor, t_id, dim, constraint_range, shared=None, debug_name=None
):
    return _Constraint._create(
        w_tensor, t_id, dim, constraint_range, shared, debug_name
    )


@dataclasses.dataclass
class _Constraint(_ConstraintTarget, metaclass=_ConstraintFactory):
    """

    .. warning::
        Do not construct :class:`_Constraint` directly, use :func:`dynamic_dim` instead.

    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.

    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: "StrictMinMaxConstraint"
    # Represent that `constraint_range` is shared with another _ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[_ConstraintTarget] = None
    debug_name: Optional[str] = None

    def _clone_with_range(self, lower=2, upper=math.inf):
        # Import sympy locally
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.value_ranges import ValueRanges

        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            self.shared,
            self.debug_name,
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of _Constraint. "
            "If you are trying to combine _Constraint's with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }

    def __eq__(self, other):
        if not isinstance(other, _Constraint):
            raise TypeError(
                "A dynamic dim can be specified equal only to another dynamic dim. "
                f"Equality with {type(other)} is not supported."
            )

        # import sympy locally
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        if self.debug_name is None:
            debug_name = other.debug_name
        else:
            assert other.debug_name is None or self.debug_name == other.debug_name
            debug_name = self.debug_name
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=_ConstraintTarget(other.w_tensor, other.t_id, other.dim),
            debug_name=debug_name,
        )


@dataclasses.dataclass
class _PhantomRoot:
    """
    This represents the root of a derived Dim where the root does not directly
    specify the shape of any input dimension, but the derived Dim does.

    e.g., the input shapes 2*dim and dim + 1 are related via a "phantom" dim.

    The fields `name`, `constraint_range`, and `val` carried by a phantom root
    help create a symbol for it. Any derived dims with this phantom root are
    backed by expressions over this symbol.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"
    val: int


@dataclasses.dataclass
class _DerivedConstraint(_ConstraintTarget):
    """
    This represents a derived Dim, whose root is either a regular constraint target
    (which directly specifies the shape of some input dimension) or a phantom root
    (which does so indirectly).
    """

    # NOTE: This is not currently a subclass of _Constraint because we do not support
    # `shared` for derived `Dim`s. Indeed, sharing is a necessary concept only for
    # legacy constraints based on `dynamic_dim`: equality can be expressed simply by
    # reusing the same (derived or normal) `Dim`.
    root: Union[_ConstraintTarget, _PhantomRoot]
    fn: Callable
    constraint_range: "StrictMinMaxConstraint"
    debug_name: Optional[str] = None

    @property
    def shared(self):
        # Some code paths expect a union of _Constraint and _DerivedConstraint.
        # Thus we expose a `shared` field that is always None.
        # TODO(avik): clean this up
        return None

    @property
    def serializable_spec(self):
        # same as _Constraint.serializable_spec
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }


Constraint = Union[_Constraint, _DerivedConstraint]


def dynamic_dim(t: torch.Tensor, index: int, debug_name: Optional[str] = None):
    """
    .. warning::
        (This feature is DEPRECATED. See :func:`Dim` instead.)

    :func:`dynamic_dim` constructs a :class:`_Constraint` object that describes the dynamism of
    a dimension ``index`` of tensor ``t``. :class:`_Constraint` objects should be passed to
    ``constraints`` argument of :func:`export`.

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A :class:`_Constraint` object that describes shape dynamism. It can be passed to :func:`export` so
        that :func:`export` does not assume static size of specified tensor, i.e. keeping it dynamic
        as a symbolic size rather than specializing according to size of example tracing input.

    Specifically :func:`dynamic_dim` can be used to express following types of dynamism.

    - Size of a dimension is dynamic and unbounded::

        t0 = torch.rand(2, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size rather than always being static size 2
        constraints = [dynamic_dim(t0, 0)]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with a lower bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a lower bound of 5 (inclusive)
        # Second dimension of t1 can be dynamic size with a lower bound of 2 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) >= 5,
            dynamic_dim(t1, 1) > 2,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with an upper bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a upper bound of 16 (inclusive)
        # Second dimension of t1 can be dynamic size with a upper bound of 8 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) <= 16,
            dynamic_dim(t1, 1) < 8,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic and it is always equal to size of another dynamic dimension::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # Sizes of second dimension of t0 and first dimension are always equal
        constraints = [
            dynamic_dim(t0, 1) == dynamic_dim(t1, 0),
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Mix and match all types above as long as they do not express conflicting requirements

    """
    from torch._dynamo.exc import UserError, UserErrorType

    if not isinstance(t, torch.Tensor):
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected tensor as input to dynamic_dim but got {type(t)}",
        )

    if t.dim() < 1:
        raise UserError(
            UserErrorType.DYNAMIC_DIM, "Cannot mark 0-dimension tensors to be dynamic"
        )

    if index >= t.dim():
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected the dimension passed to dynamic_dim to be in the range [0:{t.dim()-1}]"
            f" but got {index}, which is out of bounds for the given tensor.",
        )

    # Import sympy locally
    import sympy

    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
    from torch.utils._sympy.value_ranges import ValueRanges

    return _create_constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(
            vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False
        ),
        debug_name=debug_name,
    )


def _process_equalities(
    constraint: Constraint,
    get_sources: Callable[[int, int], List["Source"]],
    shape_env: "ShapeEnv",
    source_pairs: List[Tuple["Source", "Source"]],
    derived_equalities: List[Tuple["Source", Union["Source", "Symbol"], Callable]],
    phantom_symbols: Dict[str, "Symbol"],
):
    """
    Updates `source_pairs`, `derived_equalities`, and `phantom_symbols` (which become
    fields of `EqualityConstraint`) based on a given input `constraint`.
    """

    source, *other_sources = get_sources(constraint.t_id, constraint.dim)
    # When t.size()[dim] maps to src0, src1, ..., srcN, we add
    # constraints that make src0 "equal" to src1, ..., srcN.
    source_pairs.extend((source, other_source) for other_source in other_sources)
    if not isinstance(constraint, _DerivedConstraint):
        if constraint.shared is not None:
            # Moreover, when t.size()[dim] is specified equal to t'.size()[dim']
            # and t'.size()[dim'] maps to src1', ..., srcN', we add
            # constraints that also make src0 "equal" to src1', ..., srcN'.
            other_sources = get_sources(constraint.shared.t_id, constraint.shared.dim)
            source_pairs.extend(
                (source, other_source) for other_source in other_sources
            )
    else:
        # branch based on the root of the _DerivedConstraint
        if not isinstance(constraint.root, _PhantomRoot):
            # either root points to an input source
            root = get_sources(constraint.root.t_id, constraint.root.dim)[0]  # type: ignore[assignment]
        else:
            # or root points to a phantom symbol
            if constraint.root.name in phantom_symbols:
                root = phantom_symbols[constraint.root.name]  # type: ignore[assignment]
            else:
                # create a phantom symbol in the shape env based on the _PhantomRoot
                root = shape_env.create_symbol(
                    val=constraint.root.val,
                    source=torch._dynamo.source.ConstantSource(constraint.root.name),
                    dynamic_dim=torch.fx.experimental.symbolic_shapes.DimDynamic.DYNAMIC,
                    constraint_dim=constraint.root.constraint_range,
                )
                phantom_symbols[constraint.root.name] = root  # type: ignore[assignment]

        fn = constraint.fn
        # A derived equality (source, root, fn) informally corresponds to source = fn(root).
        # Here source describes an input and root might describe another input or a phantom symbol.
        derived_equalities.append((source, root, fn))


def _process_dynamic_shapes(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
) -> Optional[List[Constraint]]:
    from collections import defaultdict
    from collections.abc import Mapping, Sequence

    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return None

    kwargs = kwargs if kwargs is not None else {}

    def tree_zip(combined_args, dynamic_shapes):
        if isinstance(combined_args, (tuple, list)):
            if not isinstance(dynamic_shapes, Sequence):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic_shapes of a {type(combined_args)} to be a Sequence, "
                    f"got {dynamic_shapes} instead",
                )
            if len(combined_args) != len(dynamic_shapes):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected {dynamic_shapes} to have {len(combined_args)} items",
                )
            for i, shape in enumerate(dynamic_shapes):
                yield from tree_zip(combined_args[i], shape)
        elif isinstance(combined_args, dict):
            if not isinstance(dynamic_shapes, Mapping):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic_shapes of a {type(combined_args)} to be a Mapping, "
                    f"got {dynamic_shapes} instead",
                )
            if len(combined_args) != len(dynamic_shapes):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected {dynamic_shapes} to have {len(combined_args)} items",
                )
            for k, shape in dynamic_shapes.items():
                yield from tree_zip(combined_args[k], shape)
        elif type(combined_args) in SUPPORTED_NODES:
            if not isinstance(dynamic_shapes, Sequence):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic_shapes of a user-registered class (e.g., "
                    f"{type(combined_args)}) to be a Sequence that matches the "
                    f"flattened structure, but got {dynamic_shapes} instead",
                )
            yield from tree_zip(
                SUPPORTED_NODES[type(combined_args)].flatten_fn(combined_args)[0],
                dynamic_shapes,
            )
        elif isinstance(combined_args, torch.Tensor):
            yield (combined_args, dynamic_shapes)
        else:
            if dynamic_shapes is not None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic_shapes of a {type(combined_args)} to be None, "
                    f"got {dynamic_shapes} instead",
                )

    # map of Dim names representing input shape dimensions to constraints on them
    symbols: Dict[str, List[Constraint]] = defaultdict(list)
    # track roots that do not directly represent input shape dimensions
    phantom_roots: Dict[str, _PhantomRoot] = {}
    derived_constraints_with_phantom_root: List[_DerivedConstraint] = []

    def to_constraint(dim, tensor, i):
        import sympy

        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.solve import try_solve
        from torch.utils._sympy.value_ranges import ValueRanges

        def root_value():
            # given tensor.shape[i] is the value of dim = fn(root),
            # find the value of root
            symbol = sympy.Symbol(dim.root.__name__, integer=True)
            expr = dim.fn(symbol)
            solution = try_solve(sympy.Eq(expr, tensor.shape[i]), symbol)
            if solution is not None:
                return int(solution[1])  # type: ignore[call-overload]
            else:
                raise UserError(  # noqa: TRY200
                    UserErrorType.CONSTRAINT_VIOLATION,
                    f"Expected shape[{i}] = {tensor.shape[i]} of input Tensor to be "
                    f"of the form {expr}, where {symbol} is an integer",
                )

        if isinstance(dim, _DerivedDim):
            # generate a _DerivedConstraint where the root is:
            # - either a _ConstraintTarget (if dim.root directly describes an input shape)
            # - or a _PhantomRoot (otherwise)
            dim_root = dim.root  # type: ignore[attr-defined]
            if dim_root.__name__ in symbols:
                # root represents an input shape dimension
                root_constraint = symbols[dim_root.__name__][0]
                root = _ConstraintTarget(
                    root_constraint.w_tensor,
                    root_constraint.t_id,
                    root_constraint.dim,
                )
            elif dim_root.__name__ not in phantom_roots:
                # create a phantom root
                root = _PhantomRoot(  # type: ignore[assignment]
                    name=dim_root.__name__,
                    constraint_range=StrictMinMaxConstraint(
                        vr=ValueRanges(lower=dim_root.min, upper=dim_root.max),
                        warn_only=False,
                    ),
                    val=root_value(),
                )
                phantom_roots[dim_root.__name__] = root  # type: ignore[assignment]
            else:
                root = phantom_roots[dim_root.__name__]  # type: ignore[assignment]
            constraint = _DerivedConstraint(
                weakref.ref(tensor),
                id(tensor),
                i,
                root,
                dim.fn,  # type: ignore[attr-defined]
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max),
                    warn_only=False,
                ),
                debug_name=dim.__name__,
            )
            if isinstance(root, _PhantomRoot):
                # NOTE(avik): since we have not processed all inputs yet, we may replace this
                # with a root that does represent an input shape dimension later (see below)
                derived_constraints_with_phantom_root.append(constraint)
        else:
            constraint = dynamic_dim(tensor, i, debug_name=dim.__name__)
            if dim.min != 2:
                constraint = constraint >= dim.min
            if dim.max != sys.maxsize - 1:
                constraint = constraint <= dim.max
        return constraint

    bounds: Dict[str, Tuple[int, int]] = {}

    def check_same_bounds(dim):
        if dim.__name__ in symbols:
            min_, max_ = bounds[dim.__name__]
            if dim.min != min_ or dim.max != max_:
                this_ = _Dim.readable(dim.__name__, min_, max_)
                that_ = _Dim.readable(dim.__name__, dim.min, dim.max)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Found different definitions {this_} and {that_} "
                    f"for the same symbolic dimension {dim}!",
                )

        else:
            bounds[dim.__name__] = (dim.min, dim.max)

    def update_symbols(tensor, shape):
        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, _Dim):
                    check_same_bounds(dim)
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                else:
                    if dim is not None:
                        raise UserError(
                            UserErrorType.INVALID_INPUT,
                            f"Unexpected item #{i} ({dim}) in dynamic_shape {shape} of Tensor, "
                            "try None instead",
                        )
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                if isinstance(dim, _Dim):
                    check_same_bounds(dim)
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                else:
                    if dim is not None:
                        raise UserError(
                            UserErrorType.INVALID_INPUT,
                            f"Unexpected item #{i} ({dim}) in dynamic_shape {shape} of Tensor, "
                            "try None instead",
                        )
        else:
            if shape is not None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Unexpected dynamic_shape {shape} of Tensor, " "try None instead",
                )

    import inspect

    if isinstance(f, ExportedProgram):
        f = f.module()
    signature = (
        inspect.signature(f.forward)
        if isinstance(f, torch.nn.Module)
        else inspect.signature(f)
    )
    combined_args = signature.bind(*args, **kwargs).arguments

    # This means user didn't specify dynamic shapes with argument names.
    combined_args = combined_args if isinstance(dynamic_shapes, Mapping) else list(combined_args.values())  # type: ignore[assignment]
    for tensor, shape in tree_zip(combined_args, dynamic_shapes):
        update_symbols(tensor, shape)

    constraints = []
    for derived_constraint_with_phantom_root in derived_constraints_with_phantom_root:
        phantom_root_name = derived_constraint_with_phantom_root.root.name  # type: ignore[union-attr]
        if phantom_root_name in symbols:
            # We found an input shape dimension corresponding to this name, so we
            # do not need a phantom symbol for it after all.
            # NOTE(avik): Overall we want to maintain the invariant that roots that
            # are phantom symbols are really "phantom," i.e., they cannot be represented
            # by any input source. This is important when we are deciding derived equalities,
            # since we can focus our attention exclusively on input sources: deciding
            # derived equalities involving phantom symbols are, in comparison, trivial.
            derived_constraint_with_phantom_root.root = symbols[phantom_root_name][0]

    for dynamic_dims in symbols.values():
        if all(
            isinstance(dynamic_dim, _DerivedConstraint) for dynamic_dim in dynamic_dims
        ):
            constraints.extend(dynamic_dims)
        else:
            primary, *others = dynamic_dims
            if others:
                for other in others:
                    constraints.append(primary == other)  # type: ignore[arg-type]
            else:
                constraints.append(primary)

    return constraints  # type: ignore[return-value]


def _process_constraints(
    fake_mode,
    graph_module: torch.fx.GraphModule,
    num_lifted_params_buffers: int,
    example_inputs: List[torch.Tensor],
) -> Dict:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule): GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        range_constraints (Dict[sympy.Symbol, ValueRanges]): Mapping of
            symbols (from SymInts) appearing in the fake tensors in
            node.meta["val"] to their range constraints, which are a tuple
            containing (lower, upper) constraints.
    """
    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        InputDim,
    )

    # Import sympy locally
    from torch.fx.experimental.symbolic_shapes import SymInt
    from torch.utils._sympy.value_ranges import ValueRanges

    input_shape_constraints = graph_module.meta.get("input_shape_constraints", [])
    inline_constraints = graph_module.meta.get("inline_constraints", [])

    # Create dict mapping tensor_id to node names
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)
    # Create dict mapping placeholder node names to their nodes
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes):
        if node.op != "placeholder":
            # All placeholder nodes should be together in the beginning of the
            # graph
            break
        if i >= num_lifted_params_buffers:
            example_input = example_inputs[i - num_lifted_params_buffers]
            tensor_id_to_nodes[id(example_input)].append(node.name)
            placeholder_nodes[node.name] = node

    # Create dict mapping (node name, dim) a list of range (lower, upper)
    # constraints
    multi_range_constraints: Dict[InputDim, List[ValueRanges]] = defaultdict(list)
    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint["t_id"]]:
            node_dim = InputDim(node, constraint["dim"])

            # Accumulate range constraints
            multi_range_constraints[node_dim].append(
                ValueRanges(constraint["min"], constraint["max"])
            )

    # Create dict mapping symbol to a singular range (lower, upper)
    range_constraints: Dict[Any, ValueRanges] = {}

    # Add inline constraints to range_constraints
    range_constraints = {
        symbol: inline_constraints[symbol] for symbol in inline_constraints
    }

    free_symbols: Set["Symbol"] = set()
    # Add input range constraints to range_constraints
    for input_dim, multi_range_constraint in multi_range_constraints.items():  # type: ignore[assignment]
        # Simplify the range constraints into a single range constraint
        # Ex. ranges [2, 10] and [3, 11] would get merged to [3, 10]
        min_vals = [rc.lower for rc in multi_range_constraint]
        max_vals = [rc.upper for rc in multi_range_constraint]
        min_val = max(min_vals)  # type: ignore[type-var]
        max_val = min(max_vals)  # type: ignore[type-var]
        assert min_val <= max_val  # type: ignore[operator]

        # Add input node range constraints
        val = placeholder_nodes[input_dim.input_name].meta["val"]
        assert isinstance(val, FakeTensor)
        symint = val.shape[input_dim.dim]
        assert isinstance(
            symint, SymInt
        ), f"Expected SymInt but got {symint}: {type(symint)}"
        symbol = symint.node.expr
        range_constraints[symbol] = ValueRanges(min_val, max_val)
        free_symbols.update(symbol.free_symbols)

    for symbol in free_symbols:
        if symbol not in range_constraints:
            # Placeholders can have symbolic shapes that are derived expressions.
            # The above code will record direct range constraints for them
            # so that we can do runtime assertions. In addition, for serde checks
            # we want to record range constraints for their root symbols.
            range_constraints[symbol] = fake_mode.shape_env.var_to_range[symbol]

    return range_constraints
