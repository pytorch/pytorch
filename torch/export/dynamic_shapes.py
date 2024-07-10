# mypy: allow-untyped-defs
import dataclasses
import inspect
import sys
import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import torch
from torch.utils._pytree import (
    _get_node_type,
    BUILTIN_TYPES,
    SUPPORTED_NODES,
    tree_flatten,
    tree_map,
)

from .exported_program import ExportedProgram

if TYPE_CHECKING:
    from sympy import Symbol

    from torch._guards import Source

    from ..fx.experimental.symbolic_shapes import ShapeEnv, StrictMinMaxConstraint

__all__ = [
    "Constraint",
    "Dim",
    "dims",
    "dynamic_dim",
    "refine_dynamic_shapes_from_suggested_fixes",
]


class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        from torch.utils._sympy.numbers import int_oo

        if min_ == 2:
            min_ = None
        if max_ == int_oo:
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


class _StaticDim(_Dim):
    """
    Meta class for static :func:`Dim` types.

    This class is only for setting and checking static dim constraints,
    and the user should never interact with it.
    """

    @property
    def min(self):
        return self.value  # type: ignore[attr-defined]

    @property
    def max(self):
        return self.value  # type: ignore[attr-defined]


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

        from torch.utils._sympy.numbers import int_oo

        if self.root.min is -int_oo:  # type: ignore[attr-defined]
            return -int_oo  # fn not needed cuz increasing

        _min_symint = self.fn(Integer(self.root.min))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _min_symint >= 0, (
            f"Expected derived min value of {self.__name__} to be >= 0. "
            f"Please specify an appropriate min value for {root.__name__} "
            f"(currently {root.min})."
        )
        return int(_min_symint)

    @property
    def max(self):
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.max is int_oo:  # type: ignore[attr-defined]
            return int_oo  # fn not needed cuz increasing

        _max_symint = self.fn(Integer(self.root.max))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _max_symint <= sys.maxsize - 1, (
            f"Expected derived max value of {self.__name__} to be <= {sys.maxsize - 1}. "
            f"Please specify an appropriate max value for {root.__name__} "
            f"(currently {root.max})."
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
    from torch.utils._sympy.numbers import int_oo

    _min = 0 if min is None else min
    _max = int_oo if max is None else max
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

    def _clone_with_range(self, lower=0, upper=None):
        # Import sympy locally
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import ValueRanges

        if upper is None:
            upper = int_oo

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

    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.value_ranges import ValueRanges

    return _create_constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(vr=ValueRanges(lower=0, upper=int_oo), warn_only=False),
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


def _tree_map(
    func: Callable[..., Any],
    tree: Any,
    *dynamic_shapes: Any,
) -> Any:
    """
    Customized tree_map for mapping pytrees to dynamic_shapes.

    For built-in types (e.g., standard collections) this behaves exactly like tree_map.

    OTOH for a user-defined class C registered with pytree, we cannot assume that a C
    containing tensors can be mapped to a C containing dynamic shapes (i.e., C may not
    be a polymorphic container). In that case we use the flattened form of C instead.
    Thus a C(**tensors) that flattens to (**tensors) will map to (**dynamic_shapes).

    Args:
        func: function to apply to each (int, float, str, bool, None, torch.Tensor)
        tree: input pytree
        dynamic_shapes: zero or more (typically one) dynamic_shapes to match

    Returns:
        output pytree mapping func to each (int, float, str, bool, None, torch.Tensor)
    """

    def is_leaf(t):
        # BUILTIN_TYPES is a subset of SUPPORTED_NODES, the latter being all types
        # registered with pytree. Types *not* in BUILTIN_TYPES include primitive types
        # (int, float, str, bool, None, torch.Tensor), which are not in SUPPORTED_NODES,
        # as well as user-defined classes registered with pytree, which are.
        return _get_node_type(t) not in BUILTIN_TYPES

    def f(t, *dynamic_shapes):
        typ = _get_node_type(t)
        # typ is not in BUILTIN_TYPES
        if typ in SUPPORTED_NODES:
            # thus typ is a user-defined class registered with pytree,
            # in which case flatten and recurse
            return tree_map(
                f,
                SUPPORTED_NODES[typ].flatten_fn(t)[0],
                *dynamic_shapes,
                is_leaf=is_leaf,
            )
        else:
            return func(t, *dynamic_shapes)

    return tree_map(f, tree, *dynamic_shapes, is_leaf=is_leaf)


def _combine_args(f, args, kwargs, _is_torch_jit_trace=False):
    # combine args and kwargs following the signature of f, as it happens
    # in the body of f when called with *args, **kwargs
    if isinstance(f, ExportedProgram):
        f = f.module()
    if not _is_torch_jit_trace:
        signature = (
            inspect.signature(f.forward)
            if isinstance(f, torch.nn.Module)
            else inspect.signature(f)
        )
        kwargs = kwargs if kwargs is not None else {}
        return signature.bind(*args, **kwargs).arguments
    return args


class ShapesCollection:
    """
    Builder for dynamic_shapes.
    Used to assign dynamic shape specifications to tensors that appear in inputs.

    Example::
        args = ({"x": tensor_x, "others": [tensor_y, tensor_z]})

        dim = torch.export.Dim(...)
        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[tensor_y] = {0: dim * 2}
        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        self._shapes = {}

    def __setitem__(self, t, shape):
        assert isinstance(
            t, torch.Tensor
        ), f"Cannot assign shape to non-tensor type {type(t)}"
        # TODO(avik): check that shape is indeed a Shape
        t_id = id(t)
        if t_id in self._shapes:
            _shape = self._shapes[t_id]
            assert (
                shape == _shape
            ), f"Shapes assigned to tensor do not match: expected {_shape}, got {shape}"
        else:
            self._shapes[id(t)] = shape

    def __getitem__(self, t):
        t_id = id(t)
        if t_id in self._shapes:
            return self._shapes[t_id]
        else:
            return None

    def __len__(self):
        return len(self._shapes)

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Generate dynamic_shapes.
        """

        t_ids = set()

        def find_shape(t):
            t_id = id(t)
            if t_id in self._shapes:
                t_ids.add(t_id)
                return self._shapes[t_id]
            else:
                return None

        combined_args = _combine_args(m, args, kwargs)
        dynamic_shapes = _tree_map(find_shape, combined_args)
        if any(t_id not in t_ids for t_id in self._shapes):
            raise ValueError(
                "Some tensors that were assigned shapes were not found in args. "
                "Maybe such tensors were copied when passing them as args? "
                "Maybe such tensors are contained in classes that were not registered with pytree?"
            )
        return dynamic_shapes


def _process_dynamic_shapes(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    _is_torch_jit_trace=False,
) -> Optional[List[Constraint]]:
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return None

    # map of Dim names representing input shape dimensions to constraints on them
    symbols: Dict[str, List[Constraint]] = defaultdict(list)
    # track roots that do not directly represent input shape dimensions
    phantom_roots: Dict[str, _PhantomRoot] = {}
    derived_constraints_with_phantom_root: List[_DerivedConstraint] = []

    def to_constraint(dim, tensor, i):
        import sympy

        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.numbers import int_oo
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
                raise UserError(  # noqa: B904
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
        elif isinstance(dim, _StaticDim):
            constraint = _create_constraint(
                weakref.ref(tensor),
                id(tensor),
                i,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.value, upper=dim.value), warn_only=False  # type: ignore[attr-defined]
                ),
                debug_name=dim.__name__,
            )
        else:
            constraint = dynamic_dim(tensor, i, debug_name=dim.__name__)
            if dim.min != 0:
                constraint = constraint >= dim.min
            if dim.max != int_oo:
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
        def _create_static_dim(tensor, i, value):
            return _StaticDim(str(value), (int,), {"value": value})

        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, (int, _Dim)):
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
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
                if isinstance(dim, (int, _Dim)):
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
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

    def assoc_shapes(combined_args, dynamic_shapes):
        def assoc_shape(t, dynamic_shape):
            if isinstance(t, torch.Tensor):
                update_symbols(t, dynamic_shape)
            else:
                if dynamic_shape is not None:
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Cannot associate shape {dynamic_shape} to non-tensor type {type(t)}, "
                        f"expected None",
                    )

        _tree_map(assoc_shape, combined_args, dynamic_shapes)

    combined_args = _combine_args(
        f, args, kwargs, _is_torch_jit_trace=_is_torch_jit_trace
    )
    if not isinstance(dynamic_shapes, dict):
        assert isinstance(dynamic_shapes, (tuple, list))
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    assoc_shapes(combined_args, dynamic_shapes)

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


def _get_dim_name_mapping(
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None]
):
    name_to_dim = {}
    for dim in tree_flatten(
        dynamic_shapes,
        is_leaf=lambda x: isinstance(x, _Dim),
    )[0]:
        if dim is None or isinstance(dim, int):
            continue
        name_to_dim[dim.__name__] = dim
        if isinstance(dim, _DerivedDim):
            name_to_dim[dim.root.__name__] = dim.root  # type: ignore[attr-defined]
    return name_to_dim


def refine_dynamic_shapes_from_suggested_fixes(
    msg: str,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
) -> Union[Dict[str, Any], Tuple[Any], List[Any]]:
    """
    For working with export's dynamic shapes suggested fixes, and/or automatic dynamic shapes.
    Refines the given dynamic shapes spec, given a ConstraintViolation error message and the original dynamic shapes.

    For most cases behavior is straightforward - i.e. for suggested fixes that specialize or refine a Dim's range,
    or fixes that suggest a derived relation, the new dynamic shapes spec will be updated as such.

    e.g.
    Suggested fixes:

        dim = Dim('dim', min=3, max=6) -> this just refines the dim's range
        dim = 4 -> this specializes to a constant
        dy = dx + 1 -> dy was specified as an independent dim, but is actually tied to dx with this relation

    However, suggested fixes associated with derived dims can be more complicated.
    For example, if a suggested fix is provided for a root dim, the new derived dim value is evaluated based on the root.

    e.g.
    dx = Dim('dx')
    dy = dx + 2
    dynamic_shapes = {"x": (dx,), "y": (dy,)}

    Suggested fixes:

        dx = 4  # specialization will lead to dy also specializing = 6
        dx = Dim('dx', max=6)  # dy now has max = 8

    Derived dims suggested fixes can also be used to express divisibility constraints.
    This involves creating new root dims that aren't tied to a particular input shape.
    In this case the root dims won't appear directly in the new spec, but as a root of
    one of the dims.

    e.g.
    Suggested fixes:

        _dx = Dim('_dx', max=1024)  # this won't appear in the return result, but dx will
        dx = 4*_dx  # dx is now divisible by 4, with a max value of 4096
    """

    import re

    import sympy

    from torch._dynamo.exc import UserError, UserErrorType
    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    try:
        shape_fixes_msg = msg.split("Suggested fixes:")[1].strip()
    except Exception as exc:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Suggested fixes not found in error message given to refine_dynamic_shapes_from_suggested_fixes()",
        ) from exc

    # build shape_fixes dictionary
    shape_fixes = {}
    for fix in shape_fixes_msg.split("\n"):
        fix = fix.strip()
        if match := re.match(r"(.*) = Dim\('(.*)'.*\)", fix):
            name = match.group(1)
            _min, _max = None, None
            if match_min := re.match(r".* = Dim\('.*', min\=([0-9]+).*\)", fix):
                _min = int(match_min.group(1))
            if match_max := re.match(r".* = Dim\('.*'.*max\=([0-9]+)\)", fix):
                _max = int(match_max.group(1))
            shape_fixes[name] = Dim(name, min=_min, max=_max)
        else:
            name, expr = fix.split(" = ")
            expr = sympy.sympify(expr)
            if isinstance(expr, sympy.Number):
                shape_fixes[name] = int(expr)  # static, integer
            else:
                shape_fixes[name] = expr  # relation or derived dim

    name_to_dim = _get_dim_name_mapping(dynamic_shapes)

    # track derived dim roots
    roots: Set[str] = set()
    for k, c in shape_fixes.items():
        assert isinstance(c, (int, _Dim, _DerivedDim, sympy.Expr))
        if isinstance(c, sympy.Expr):  # check dim/derived dim expression
            assert _is_supported_equivalence(c)
            shape_fixes[k] = c
            roots.add(str(next(iter(c.free_symbols))))
        if isinstance(c, _DerivedDim):
            roots.add(c.root.__name__)  # type: ignore[attr-defined]

    # check keys are existing dims or new roots
    for k, c in shape_fixes.items():
        assert k in name_to_dim or k in roots

    # cache so we don't produce multiple derived dim objects
    derived_dim_cache: Dict[str, _DerivedDim] = {}

    def apply_fixes(dim, dummy):
        if dim is None or isinstance(dim, int):  # not dynamic
            return dim
        elif dim.__name__ in shape_fixes:  # directly fix
            fix = shape_fixes[dim.__name__]
            if isinstance(fix, sympy.Expr):  # now derived or related
                if str(fix) in derived_dim_cache:
                    return derived_dim_cache[str(fix)]
                else:
                    symbol = next(iter(fix.free_symbols))
                    # try to locate symbol
                    if symbol.name in shape_fixes:  # type: ignore[attr-defined]
                        root = shape_fixes[symbol.name]  # type: ignore[attr-defined]
                    else:
                        assert symbol.name in name_to_dim  # type: ignore[attr-defined]
                        root = name_to_dim[symbol.name]  # type: ignore[attr-defined]
                    # figure out value of fix
                    modulus, remainder = sympy.polys.polytools.div(fix, symbol)
                    dim = root
                    if modulus != 1:
                        dim = int(modulus) * dim
                    if remainder != 0:
                        dim = dim + int(remainder)
                    derived_dim_cache[str(fix)] = dim
                    return dim
            else:
                return fix
        elif isinstance(dim, _DerivedDim) and dim.root.__name__ in shape_fixes:  # type: ignore[attr-defined]
            if dim.__name__ in derived_dim_cache:
                return derived_dim_cache[dim.__name__]
            else:  # evaluate new derived value based on root
                _dim = dim.fn(shape_fixes[dim.root.__name__])  # type: ignore[attr-defined]
                derived_dim_cache[dim.__name__] = _dim
                return _dim
        return dim  # unchanged dim

    return _tree_map(apply_fixes, dynamic_shapes, dynamic_shapes)
