# mypy: allow-untyped-defs
import dataclasses
import inspect
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from enum import auto, Enum
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch.utils._pytree import (
    _get_node_type,
    BUILTIN_TYPES,
    keystr,
    LeafSpec,
    MappingKey,
    SequenceKey,
    SUPPORTED_NODES,
    tree_flatten,
    tree_map,
    tree_map_with_path,
)

from .exported_program import ExportedProgram


if TYPE_CHECKING:
    from sympy import Symbol

    from torch._guards import Source
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, StrictMinMaxConstraint

__all__ = [
    "Constraint",
    "Dim",
    "dims",
    "refine_dynamic_shapes_from_suggested_fixes",
    "AdditionalInputs",
]


log = logging.getLogger(__name__)


class _DimHintType(Enum):
    """
    Enum for dynamic shape hints.
    - AUTO means automatic inference of shape (static or dynamic).
    - STATIC means static shape (always specialized).
    - DYNAMIC means dynamic, will error out if specialized.
    """

    AUTO = auto()
    STATIC = auto()
    DYNAMIC = auto()


@dataclasses.dataclass
class _DimHint:
    """
    Internal class for dynamic shape hints.
    - min and max are optional.
    - _factory is for UX only, below example:
        auto_hint = _DimHint.AUTO()  # _factory=True
        bounded_hint = auto_hint(min=10, max=100)  # Returns new instance with _factory=False
        bounded_hint(min=5, max=50)  # Will fail, non-factory instance cannot be called
    """

    type: _DimHintType
    min: Optional[int] = None
    max: Optional[int] = None
    _factory: Optional[bool] = True

    @staticmethod
    def AUTO():
        return _DimHint(_DimHintType.AUTO)

    @staticmethod
    def DYNAMIC():
        return _DimHint(_DimHintType.DYNAMIC)

    @staticmethod
    def STATIC():
        return _DimHint(_DimHintType.STATIC)

    def __call__(self, min=None, max=None) -> "_DimHint":
        if not self._factory:
            raise TypeError(f"'{type(self)}' object is not callable")
        assert min is None or min >= 0, "min must be non-negative"
        assert max is None or max >= 0, "max must be non-negative"
        assert min is None or max is None or min <= max, "min must be <= max"
        return _DimHint(self.type, min=min, max=max, _factory=False)

    def __repr__(self):
        parts = [self.type.name]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        return f"DimHint({', '.join(parts)})"


class Dim:
    """
    The ``Dim`` class allows users to specify dynamism in their exported
    programs. By marking a dimension with a ``Dim``, the compiler associates the
    dimension with a symbolic integer containing a dynamic range.

    The API can be used in 2 ways: Dim hints (i.e. automatic dynamic shapes:
    ``Dim.AUTO``, ``Dim.DYNAMIC``, ``Dim.STATIC``), or named Dims (i.e.
    ``Dim("name", min=1, max=2)``).

    Dim hints provide the lowest barrier to exportability, with the user only
    needing to specify if a dimension if dynamic, static, or left for the
    compiler to decide (``Dim.AUTO``). The export process will automatically
    infer the remaining constraints on min/max ranges and relationships between
    dimensions.

    Example::

        class Foo(nn.Module):
            def forward(self, x, y):
                assert x.shape[0] == 4
                assert y.shape[0] >= 16
                return x @ y


        x = torch.randn(4, 8)
        y = torch.randn(8, 16)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
            "y": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Here, export would raise an exception if we replaced all uses of ``Dim.AUTO`` with ``Dim.DYNAMIC``,
    as ``x.shape[0]`` is constrained to be static by the model.

    More complex relations between dimensions may also be codegened as runtime assertion nodes by the compiler,
    e.g. ``(x.shape[0] + y.shape[1]) % 4 == 0``, to be raised if runtime inputs do not satisfy such constraints.

    You may also specify min-max bounds for Dim hints, e.g. ``Dim.AUTO(min=16, max=32)``, ``Dim.DYNAMIC(max=64)``,
    with the compiler inferring the remaining constraints within the ranges. An exception will be raised if
    the valid range is entirely outside the user-specified range.

    Named Dims provide a stricter way of specifying dynamism, where exceptions are raised if the compiler
    infers constraints that do not match the user specification. For example, exporting the previous
    model, the user would need the following ``dynamic_shapes`` argument::

        s0 = Dim("s0")
        s1 = Dim("s1", min=16)
        dynamic_shapes = {
            "x": {0: 4, 1: s0},
            "y": {0: s0, 1: s1},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Named Dims also allow specification of relationships between dimensions, up
    to univariate linear relations.  For example, the following indicates one
    dimension is a multiple of another plus 4::

        s0 = Dim("s0")
        s1 = 3 * s0 + 4

    """

    AUTO = _DimHint.AUTO()
    DYNAMIC = _DimHint.DYNAMIC()
    STATIC = _DimHint.STATIC()

    def __init__(
        self, name: str, *, min: Optional[int] = None, max: Optional[int] = None
    ):
        from torch.utils._sympy.numbers import int_oo

        _min = 0 if min is None else min
        _max = int_oo if max is None else max
        assert _max > _min, f"Cannot create Dim with inconsistent min={min}, max={max}"
        assert name.isidentifier(), f"Dim name must be a valid identifier, got {name}"
        self.__name__ = name
        self.min = _min
        self.max = _max

    def __add__(self, other) -> "Dim":
        # e.g., dim + 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to add {other} to {self.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x + other)

    def __radd__(self, other) -> "Dim":
        return self + other

    def __sub__(self, other) -> "Dim":
        # e.g., dim - 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to subtract {other} from {self.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x - other)

    def __rsub__(self, other) -> "Dim":
        raise NotImplementedError(
            f"Attempted to negate {self.__name__}. "
            "(Only increasing linear operations with integer coefficients are supported.)"
        )

    def __mul__(self, other) -> "Dim":
        # e.g., dim * 2
        if type(other) is not int or other <= 0:
            raise NotImplementedError(
                f"Attempted to multiply {other} with {self.__name__}, where a positive integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x * other)

    def __rmul__(self, other) -> "Dim":
        return self * other

    def _derived_name(self, fn) -> str:
        from sympy import sympify

        return str(fn(sympify(self.__name__)))

    def _derive(self, fn) -> "Dim":
        return _DerivedDim(self._derived_name(fn), self, fn)

    @staticmethod
    def _readable(name: str, min_: int, max_: int) -> str:
        from torch.utils._sympy.numbers import int_oo

        if min_ == 2:
            min_ = None  # type: ignore[assignment]
        if max_ == int_oo:
            max_ = None  # type: ignore[assignment]
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"

    def __repr__(self):
        return Dim._readable(self.__name__, self.min, self.max)


_Dim = Dim  # TODO(pianpwk): remove after it's no longer internally breaking


class _StaticDim(Dim):
    """
    Class for static :func:`Dim` types.

    This class is only for setting and checking static dim constraints,
    and the user should never interact with it.
    """

    def __init__(self, value: int):
        self.__name__ = str(value)
        self.value = value

    @property
    def min(self):  # type: ignore[override]
        return self.value  # type: ignore[attr-defined]

    @property
    def max(self):  # type: ignore[override]
        return self.value  # type: ignore[attr-defined]


class _DerivedDim(Dim):
    """
    Class for derived :func:`Dim` types.

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

    def __init__(self, name: str, root: Dim, fn: Callable):
        self.__name__ = name
        self.root = root
        self.fn = fn

    @property
    def min(self):  # type: ignore[override]
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
    def max(self):  # type: ignore[override]
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
            self.root,
            lambda x: fn(self.fn(x)),
        )

    def __repr__(self):
        return self.__name__


def dims(
    *names: str, min: Optional[int] = None, max: Optional[int] = None
) -> tuple[Dim, ...]:
    """
    Util to create multiple :func:`Dim` types.

    Returns:
        A tuple of :func:`Dim` types.
    """
    return tuple(Dim(name, min=min, max=max) for name in names)  # type: ignore[misc]


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.
    """

    t_id: int
    dim: int


@dataclasses.dataclass
class _Constraint(_ConstraintTarget):
    """
    This represents a Dim describing a constraint target.

    `name` is the name of the Dim.
    `constraint_range` contains the min/max bounds of the Dim.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"

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
        return _Constraint(
            self.t_id,
            self.dim,
            self.name,
            constraint_range,
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

    It can be thought of as a subclass of `_Constraint`, except that it does not
    support <, <=, >, >= operations.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"
    root: Union[_ConstraintTarget, _PhantomRoot]
    fn: Callable

    @property
    def serializable_spec(self):
        # same as _Constraint.serializable_spec
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }


@dataclasses.dataclass
class _RelaxedConstraint(_ConstraintTarget):
    """
    This represents a dim marked with Dim.AUTO/DYNAMIC (i.e. mark_dynamic() or maybe_mark_dynamic()),
    which leaves relations & min/max ranges for inference, instead of requiring explicit specification.
    The intention is for constraint violations to not be raised if produce_guards() finds equalities or
    relations between a _RelaxedConstraint and another type of _Constraint.
    """

    @property
    def serializable_spec(self):
        return {
            "t_id": self.t_id,
            "dim": self.dim,
        }


Constraint = Union[_Constraint, _DerivedConstraint, _RelaxedConstraint]


@dataclasses.dataclass
class _IntWrapper:
    """
    Dummy wrapper class to wrap around integer inputs so that when we parse the
    dynamic_shapes structure, we can mark if any of the integers were marked as
    dynamic.
    """

    val: int
    # Disallow specifying dynamism
    dynamism: Optional[Union[_DimHint, int]] = dataclasses.field(
        init=False, default=None
    )


def _process_equalities(
    constraint: Constraint,
    get_sources: Callable[[int, int], list["Source"]],
    shape_env: "ShapeEnv",
    names: dict[str, tuple[int, int]],
    source_pairs: list[tuple["Source", "Source"]],
    derived_equalities: list[tuple["Source", Union["Source", "Symbol"], Callable]],
    phantom_symbols: dict[str, "Symbol"],
    relaxed_sources: set["Source"],
):
    """
    Updates `source_pairs`, `derived_equalities`, and `phantom_symbols` (which become
    fields of `EqualityConstraint`) based on a given input `constraint`.
    """

    sources = get_sources(constraint.t_id, constraint.dim)
    if not sources:  # empty sources due to unused shapes
        return

    source, *other_sources = sources
    # When t.size()[dim] maps to src0, src1, ..., srcN, we add
    # constraints that make src0 "equal" to src1, ..., srcN.
    source_pairs.extend((source, other_source) for other_source in other_sources)
    if isinstance(constraint, _Constraint):
        if constraint.name in names:
            shared_t_id, shared_dim = names[constraint.name]
            other_sources = get_sources(shared_t_id, shared_dim)
            source_pairs.extend(
                (source, other_source) for other_source in other_sources
            )
        else:
            names[constraint.name] = (constraint.t_id, constraint.dim)
    elif isinstance(constraint, _DerivedConstraint):
        # branch based on the root of the _DerivedConstraint
        if not isinstance(constraint.root, _PhantomRoot):
            # either root points to an input source
            root = get_sources(constraint.root.t_id, constraint.root.dim)[0]
        else:
            # or root points to a phantom symbol
            if constraint.root.name in phantom_symbols:
                root = phantom_symbols[constraint.root.name]
            else:
                # create a phantom symbol in the shape env based on the _PhantomRoot
                root = shape_env.create_symbol(
                    val=constraint.root.val,
                    source=torch._dynamo.source.ConstantSource(constraint.root.name),
                    dynamic_dim=torch.fx.experimental.symbolic_shapes.DimDynamic.DYNAMIC,
                    constraint_dim=constraint.root.constraint_range,
                )
                phantom_symbols[constraint.root.name] = root

        fn = constraint.fn
        # A derived equality (source, root, fn) informally corresponds to source = fn(root).
        # Here source describes an input and root might describe another input or a phantom symbol.
        derived_equalities.append((source, root, fn))
    elif isinstance(constraint, _RelaxedConstraint):
        relaxed_sources.add(source)


def _tree_map_with_path(
    func: Callable[..., Any],
    tree: Any,
    *dynamic_shapes: Any,
    tree_name: Optional[str] = None,
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

    def f(path, t, *dynamic_shapes):
        typ = _get_node_type(t)
        # typ is not in BUILTIN_TYPES
        if typ in SUPPORTED_NODES:
            # thus typ is a user-defined class registered with pytree,
            # in which case flatten and recurse
            return tree_map_with_path(
                f,
                SUPPORTED_NODES[typ].flatten_fn(t)[0],
                *dynamic_shapes,
                is_leaf=is_leaf,
            )
        else:
            return func(path, t, *dynamic_shapes)

    try:
        return tree_map_with_path(f, tree, *dynamic_shapes, is_leaf=is_leaf)
    except ValueError as e:
        if "mismatch" in e.args[0]:
            # When PyTree finds a structural mismatch between tree and dynamic_shapes,
            # the error message is unfortunately quite horrible. Let's fix that.
            assert dynamic_shapes, "Cannot be a mismatch if there is no dynamic_shapes"
            assert tree_name, "Must provide a tree_name when there might be a mismatch"

            def _key(type_, context, i):
                # derive a PyTree key given the type, context, and child # of a TreeSpec
                if type_ is dict:
                    return MappingKey(context[i])
                if type_ in (list, tuple):
                    assert context is None
                    return SequenceKey(i)
                raise AssertionError(f"Did not expect type {type_}")

            def raise_mismatch_error(msg):
                from torch._dynamo.exc import UserError, UserErrorType

                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Detected mismatch between the structure of `{tree_name}` and `dynamic_shapes`: {msg}",
                    case_name="dynamic_shapes_validation",
                )

            def _compare(tree, dynamic_shapes, path):
                # raise an error at the point where tree and dynamic_shapes differ,
                # including the path to that point and the reason for the difference
                rendered_path = keystr(path)
                if isinstance(tree, LeafSpec):
                    return
                if isinstance(dynamic_shapes, LeafSpec):
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` is a {tree.type}, "
                        f"but `dynamic_shapes{rendered_path}` is not"
                    )
                if tree.type != dynamic_shapes.type:
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` is a {tree.type}, "
                        f"but `dynamic_shapes{rendered_path}` is a {dynamic_shapes.type}"
                    )
                if len(tree.children_specs) != len(dynamic_shapes.children_specs):
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` has {len(tree.children_specs)} elements, "
                        f"but `dynamic_shapes{rendered_path}` has {len(dynamic_shapes.children_specs)} elements"
                    )
                if tree.type is dict:
                    # context, children could be out of order
                    if sorted(tree.context) != sorted(dynamic_shapes.context):
                        raise_mismatch_error(
                            f"`{tree_name}{rendered_path}` has keys {tree.context}, "
                            f"but `dynamic_shapes{rendered_path}` has keys {dynamic_shapes.context}"
                        )
                    _remap = dict(
                        zip(dynamic_shapes.context, dynamic_shapes.children_specs)
                    )
                    dynamic_shapes_children_specs = [_remap[k] for k in tree.context]
                else:
                    dynamic_shapes_children_specs = dynamic_shapes.children_specs
                for i, (tree_, dynamic_shapes_) in enumerate(
                    zip(tree.children_specs, dynamic_shapes_children_specs)
                ):
                    _compare(
                        tree_,
                        dynamic_shapes_,
                        path + [_key(tree.type, tree.context, i)],
                    )

            _, tree_spec = tree_flatten(tree, is_leaf=is_leaf)
            for other_tree in dynamic_shapes:
                _, other_tree_spec = tree_flatten(other_tree, is_leaf)
                _compare(tree_spec, other_tree_spec, [])
        raise


def _combine_args(f, args, kwargs) -> dict[str, Any]:
    # combine args and kwargs following the signature of f, as it happens
    # in the body of f when called with *args, **kwargs
    if isinstance(f, ExportedProgram):
        f = f.module()

    signature = (
        inspect.signature(f.forward)
        if isinstance(f, torch.nn.Module)
        else inspect.signature(f)
    )
    kwargs = kwargs if kwargs is not None else {}
    return signature.bind(*args, **kwargs).arguments


class ShapesCollection:
    """
    Builder for dynamic_shapes.
    Used to assign dynamic shape specifications to tensors that appear in inputs.

    This is useful particularly when :func:`args` is a nested input structure, and it's
    easier to index the input tensors, than to replicate the structure of :func:`args` in
    the :func:`dynamic_shapes` specification.

    Example::

        args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

        dim = torch.export.Dim(...)
        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[tensor_y] = {0: dim * 2}
        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)

    To specify dynamism for integers, we need to first wrap the integers using
    _IntWrapper so that we have a "unique identification tag" for each integer.

    Example::

        args = {"x": tensor_x, "others": [int_x, int_y]}
        # Wrap all ints with _IntWrapper
        mapped_args = pytree.tree_map_only(int, lambda a: _IntWrapper(a), args)

        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[mapped_args["others"][0]] = Dim.DYNAMIC

        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [Dim.DYNAMIC, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        self._shapes = {}

    def __setitem__(self, t, shape):
        assert isinstance(t, (torch.Tensor, _IntWrapper)), (
            f"Cannot assign shape to non-tensor or non-_IntWrapper type {type(t)}"
        )

        # TODO(avik): check that shape is indeed a Shape

        t_id = id(t)
        if t_id in self._shapes:
            _shape = self._shapes[t_id]
            assert shape == _shape, (
                f"Shapes assigned to input do not match: expected {_shape}, got {shape}"
            )
        else:
            self._shapes[id(t)] = shape

    def __getitem__(self, t):
        t_id = id(t)
        if t_id not in self._shapes:
            self._shapes[t_id] = {}
        return self._shapes[t_id]

    def __len__(self):
        return len(self._shapes)

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Generates the :func:`dynamic_shapes` pytree structure according to :func:`args` and :func:`kwargs`.
        """

        t_ids = set()

        def find_shape(path, t):
            t_id = id(t)
            if t_id in self._shapes:
                t_ids.add(t_id)
                return self._shapes[t_id]
            else:
                return None

        combined_args = _combine_args(m, args, kwargs)
        dynamic_shapes = _tree_map_with_path(find_shape, combined_args)
        if any(t_id not in t_ids for t_id in self._shapes):
            raise ValueError(
                "Some tensors that were assigned shapes were not found in args. "
                "Maybe such tensors were copied when passing them as args? "
                "Maybe such tensors are contained in classes that were not registered with pytree?"
            )
        return dynamic_shapes


class AdditionalInputs:
    """
    Infers dynamic_shapes based on additional inputs.

    This is useful particularly for deployment engineers who, on the one hand, may
    have access to ample testing or profiling data that can provide a fair sense of
    representative inputs for a model, but on the other hand, may not know enough
    about the model to guess which input shapes should be dynamic.

    Input shapes that are different than the original are considered dynamic; conversely,
    those that are the same as the original are considered static. Moreover, we verify
    that the additional inputs are valid for the exported program. This guarantees that
    tracing with them instead of the original would have generated the same graph.

    Example::

        args0, kwargs0 = ...  # example inputs for export

        # other representative inputs that the exported program will run on
        dynamic_shapes = torch.export.AdditionalInputs()
        dynamic_shapes.add(args1, kwargs1)
        ...
        dynamic_shapes.add(argsN, kwargsN)

        torch.export(..., args0, kwargs0, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        self._examples = []

    def add(self, args, kwargs=None):
        """
        Additional input :func:`args` and :func:`kwargs`.
        """

        assert type(args) is tuple, f"Representative args {args} must be a tuple"
        assert kwargs is None or type(kwargs) is dict, (
            f"Representative kwargs {kwargs} must be None or a dict"
        )
        self._examples.append((args, kwargs))

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Infers a :func:`dynamic_shapes` pytree structure by merging shapes of the
        original input :func:`args` and :func:`kwargs` and of each additional input
        args and kwargs.
        """

        dynamic_shapes, *other_dynamic_shapes = [
            _tree_map_with_path(
                lambda path, t: tuple(t.shape) if isinstance(t, torch.Tensor) else t,
                _combine_args(m, args, kwargs),
            )
            for args, kwargs in [(args, kwargs), *self._examples]
        ]

        def _mark_dynamism(v, *other_vs):
            if not all(type(v) == type(other) for other in other_vs):
                raise ValueError(
                    "The following inputs were found to have differing types, "
                    f"so they cannot be marked as dynamic: {(v,) + other_vs}."
                )

            if isinstance(v, int) and not isinstance(v, bool):
                if all(other_v == v for other_v in other_vs):
                    return None
                else:
                    return Dim.DYNAMIC
            else:
                if not all(other_v == v for other_v in other_vs):
                    raise ValueError(
                        "The following inputs were found to have differing values, "
                        f"but they cannot be marked as dynamic: {(v,) + other_vs}."
                    )
                return None

        return tree_map(
            _mark_dynamism,
            dynamic_shapes,
            *other_dynamic_shapes,
            is_leaf=lambda i: type(i) is int,
        )

    def verify(self, ep):
        """
        Verifies that an exported program is valid for each additional input.
        """

        epm = ep.module()
        for args, kwargs in self._examples:
            torch.export._unlift._check_input_constraints_for_module(
                epm, args, kwargs or {}
            )


def _warn_on_None_dynamic_shape_dimension():
    msg = (
        "Using None as a dynamic shape dimension is deprecated. "
        "Please use Dim.STATIC instead"
    )
    # TODO(avik): raise an error in the future
    log.warning(msg)


def _check_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
):
    """
    Checks the dynamic_shapes specification for correctness,
    using combined args + kwargs as reference for inputs structure.
    """
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return
    if isinstance(dynamic_shapes, (tuple, list)):
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]

    bounds: dict[str, tuple[int, int]] = {}

    def check_same_bounds(dim):
        if dim.__name__ in bounds:
            min_, max_ = bounds[dim.__name__]
            if dim.min != min_ or dim.max != max_:
                this_ = Dim._readable(dim.__name__, min_, max_)
                that_ = Dim._readable(dim.__name__, dim.min, dim.max)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Found different definitions {this_} and {that_} "
                    f"for the same symbolic dimension {dim}!",
                )
        else:
            bounds[dim.__name__] = (dim.min, dim.max)

    def check_symbols(path, tensor, shape):
        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension mapped to index {i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f" but got {dim!r} instead)",
                        case_name="dynamic_shapes_validation",
                    )
        elif isinstance(shape, (tuple, list)):
            if len(shape) != len(tensor.shape):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic shape spec {shape} specified at `dynamic_shapes{keystr(path)}` "
                    f"to have the same length as the actual tensor shape {tensor.shape} "
                    f"(expected {len(tensor.shape)}, but got {len(shape)} instead)",
                    case_name="dynamic_shapes_validation",
                )
            for i, dim in enumerate(shape):
                if isinstance(dim, Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension #{i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f"but got {dim!r} instead)",
                        case_name="dynamic_shapes_validation",
                    )
        elif shape is not None:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Unexpected input tensor shape {shape} specified at `dynamic_shapes{keystr(path)}` "
                f"(expected either a list/tuple of dimensions, or a dict mapping indices to dimensions,"
                f" where each dimension is an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC)",
                case_name="dynamic_shapes_validation",
            )

    assert isinstance(dynamic_shapes, (dict, tuple, list))
    if isinstance(dynamic_shapes, dict):
        got_keys = list(dynamic_shapes.keys())
        expected_arg_names = list(combined_args.keys())
        if sorted(got_keys) != sorted(expected_arg_names):
            msg = (
                f"When `dynamic_shapes` is specified as a dict, its top-level keys "
                f"must be the arg names {expected_arg_names} of `inputs`, but "
                f"here they are {got_keys}. "
            )
            if (
                len(combined_args) == 1
                and expected_arg_names[0] not in got_keys
                and isinstance(combined_args[expected_arg_names[0]], dict)
            ):
                msg += (
                    "Since here `inputs` is a list/tuple enclosing a single dict, "
                    "maybe you just forgot to enclose `dynamic_shapes` in a list/tuple?"
                )
            else:
                msg += (
                    "Alternatively, you could also ignore arg names entirely "
                    "and specify `dynamic_shapes` as a list/tuple matching `inputs`."
                )
            raise UserError(
                UserErrorType.INVALID_INPUT, msg, case_name="dynamic_shapes_validation"
            )

    def check_shape(path, t, dynamic_shape):
        if isinstance(t, torch.Tensor):
            check_symbols(path, t, dynamic_shape)
        elif isinstance(t, _IntWrapper):
            if isinstance(dynamic_shape, _Dim):
                raise ValueError(
                    "Unable to specify input integers as dynamic through named "
                    "Dims. Please use Dim.AUTO/DYNAMIC instead."
                )
            assert dynamic_shape is None or isinstance(dynamic_shape, (int, _DimHint))
        else:
            if dynamic_shape is not None:
                rendered_path = keystr(path)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Cannot associate shape {dynamic_shape} specified at `dynamic_shapes{rendered_path}` "
                    f"to non-tensor type {type(t)} at `inputs{rendered_path}` (expected None)",
                    case_name="dynamic_shapes_validation",
                )

    _tree_map_with_path(check_shape, combined_args, dynamic_shapes, tree_name="inputs")


def _process_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
) -> list[Constraint]:
    """
    Reads the dynamic_shapes specification and produces a list of constraints.
    """
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        # we run with dynamic by default, so no need to produce constraints
        return []
    if isinstance(dynamic_shapes, (tuple, list)):
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]

    # map of Dim names representing input shape dimensions to constraints on them
    symbols: dict[str, list[Constraint]] = defaultdict(list)
    # track roots that do not directly represent input shape dimensions
    phantom_roots: dict[str, _PhantomRoot] = {}
    derived_constraints_with_phantom_root: list[_DerivedConstraint] = []
    # list of constraints to return
    constraints: list[Constraint] = []

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
                return int(solution[1])
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
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max),
                    warn_only=False,
                ),
                root,
                dim.fn,  # type: ignore[attr-defined]
            )
            if isinstance(root, _PhantomRoot):
                # NOTE(avik): since we have not processed all inputs yet, we may replace this
                # with a root that does represent an input shape dimension later (see below)
                derived_constraints_with_phantom_root.append(constraint)
        elif isinstance(dim, _StaticDim):
            constraint = _Constraint(  # type: ignore[assignment]
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.value, upper=dim.value),  # type: ignore[attr-defined]
                    warn_only=False,
                ),
            )
        else:
            assert isinstance(dim, Dim)
            constraint = _Constraint(  # type: ignore[assignment]
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max),  # type: ignore[attr-defined]
                    warn_only=False,
                ),
            )
        return constraint

    def _parse_tensor_dim(tensor, idx, dim) -> None:
        def _create_static_dim(tensor, i, value):
            return _StaticDim(value)

        if isinstance(dim, (int, Dim)):
            if isinstance(dim, int):
                dim = _create_static_dim(tensor, idx, dim)
            constraint = to_constraint(dim, tensor, idx)
            symbols[dim.__name__].append(constraint)
        elif isinstance(dim, _DimHint):
            if dim.type == _DimHintType.AUTO:
                torch._dynamo.maybe_mark_dynamic(tensor, idx)
            elif dim.type == _DimHintType.STATIC:
                torch._dynamo.mark_static(tensor, idx)
            elif dim.type == _DimHintType.DYNAMIC:
                torch._dynamo.mark_dynamic(tensor, idx)
            constraints.append(_RelaxedConstraint(id(tensor), idx))
        elif dim is None:
            torch._dynamo.mark_static(tensor, idx)

    def update_symbols(path, tensor, shape):
        # clean out decorators from user side, or previous export call
        # we also delete these attributes in non_strict_utils.py/make_constraints()
        tensor._dynamo_weak_dynamic_indices = set()
        tensor._dynamo_dynamic_indices = set()
        tensor._dynamo_dynamic_range = set()
        tensor._dynamo_static_indices = set()
        tensor._dynamo_unbacked_indices = set()

        if isinstance(shape, dict):
            for i, dim in shape.items():
                _parse_tensor_dim(tensor, i, dim)
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                _parse_tensor_dim(tensor, i, dim)
        elif shape is None:
            for i in range(tensor.dim()):
                _parse_tensor_dim(tensor, i, None)

    def assoc_shape(path, t, dynamic_shape):
        if isinstance(t, torch.Tensor):
            update_symbols(path, t, dynamic_shape)
        elif isinstance(t, _IntWrapper):
            # If tensor dimensions are marked as dynamic, the tensors themselves
            # get marked using mark_dynamic. However since we can't mark
            # integers as dynamic, we first wrap integers in this class, and
            # then set the `dim` field of the class with the dynamic shapes dim
            # to mark the integer as dynamic.
            t.dynamism = dynamic_shape

    _tree_map_with_path(assoc_shape, combined_args, dynamic_shapes, tree_name="inputs")

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
        constraints.extend(dynamic_dims)

    return constraints


def _get_dim_name_mapping(
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
):
    name_to_dim = {}
    for dim in tree_flatten(
        dynamic_shapes,
        is_leaf=lambda x: isinstance(x, Dim),
    )[0]:
        if dim is None:
            # NOTE: this must denote a non-Tensor or automatic at this point.
            continue
        if isinstance(dim, int):
            continue
        elif isinstance(dim, Dim):
            name_to_dim[dim.__name__] = dim
            if isinstance(dim, _DerivedDim):
                name_to_dim[dim.root.__name__] = dim.root  # type: ignore[attr-defined]
        else:
            assert isinstance(dim, _DimHint)
    return name_to_dim


def refine_dynamic_shapes_from_suggested_fixes(
    msg: str,
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any]],
) -> Union[dict[str, Any], tuple[Any], list[Any]]:
    """
    When exporting with :func:`dynamic_shapes`, export may fail with a ConstraintViolation error if the specification
    doesn't match the constraints inferred from tracing the model. The error message may provide suggested fixes -
    changes that can be made to :func:`dynamic_shapes` to export successfully.

    Example ConstraintViolation error message::

        Suggested fixes:

            dim = Dim('dim', min=3, max=6)  # this just refines the dim's range
            dim = 4  # this specializes to a constant
            dy = dx + 1  # dy was specified as an independent dim, but is actually tied to dx with this relation

    This is a helper function that takes the ConstraintViolation error message and the original :func:`dynamic_shapes` spec,
    and returns a new :func:`dynamic_shapes` spec that incorporates the suggested fixes.

    Example usage::

        try:
            ep = export(mod, args, dynamic_shapes=dynamic_shapes)
        except torch._dynamo.exc.UserError as exc:
            new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                exc.msg, dynamic_shapes
            )
            ep = export(mod, args, dynamic_shapes=new_shapes)

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
                # static, integer
                shape_fixes[name] = int(expr)  # type: ignore[assignment]
            else:
                # relation or derived dim
                shape_fixes[name] = expr

    name_to_dim = _get_dim_name_mapping(dynamic_shapes)

    # track derived dim roots
    roots: set[str] = set()
    for k, c in shape_fixes.items():
        assert isinstance(c, (int, Dim, _DerivedDim, sympy.Expr))
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
    derived_dim_cache: dict[str, _DerivedDim] = {}

    def apply_fixes(path, dim, dummy):
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
                    if symbol.name in shape_fixes:
                        root = shape_fixes[symbol.name]
                    else:
                        assert symbol.name in name_to_dim
                        root = name_to_dim[symbol.name]
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

    return _tree_map_with_path(apply_fixes, dynamic_shapes, dynamic_shapes)
