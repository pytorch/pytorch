# mypy: allow-untyped-defs
import dataclasses
import inspect
import logging
import sys
from collections import defaultdict
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

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
]


log = logging.getLogger(__name__)


class _DimHint(Enum):
    """
    Enum for dynamic shape hints.
    - AUTO means automatic inference of shape (static or dynamic).
    - STATIC means static shape (always specialized).
    - DYNAMIC means dynamic, will error out if specialized.
    """

    AUTO = auto()
    STATIC = auto()
    DYNAMIC = auto()


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
    assert name.isidentifier(), f"Dim name must be a valid identifier, got {name}"
    dim = _Dim(name, (int,), {"min": _min, "max": _max})
    dim.__module__ = getattr(
        inspect.getmodule(inspect.stack()[1][0]), "__name__", "__main__"
    )
    return dim


Dim.AUTO = _DimHint.AUTO  # type: ignore[attr-defined]
Dim.STATIC = _DimHint.STATIC  # type: ignore[attr-defined]
Dim.DYNAMIC = _DimHint.DYNAMIC  # type: ignore[attr-defined]


def dims(*names: str, min: Optional[int] = None, max: Optional[int] = None):
    """
    Util to create multiple :func:`Dim` types.
    """
    return tuple(Dim(name, min=min, max=max) for name in names)


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


def _process_equalities(
    constraint: Constraint,
    get_sources: Callable[[int, int], List["Source"]],
    shape_env: "ShapeEnv",
    names: Dict[str, Tuple[int, int]],
    source_pairs: List[Tuple["Source", "Source"]],
    derived_equalities: List[Tuple["Source", Union["Source", "Symbol"], Callable]],
    phantom_symbols: Dict[str, "Symbol"],
    relaxed_sources: Set["Source"],
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


def _combine_args(f, args, kwargs, _is_torch_jit_trace=False) -> Dict[str, Any]:
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


def _warn_on_None_dynamic_shape_dimension():
    msg = (
        "Using None as a dynamic shape dimension is deprecated. "
        "Please use Dim.STATIC instead"
    )
    # TODO(avik): raise an error in the future
    log.warning(msg)


def _check_dynamic_shapes(
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
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

    bounds: Dict[str, Tuple[int, int]] = {}

    def check_same_bounds(dim):
        if dim.__name__ in bounds:
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

    def check_symbols(path, tensor, shape):
        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, _Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension mapped to index {i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f" but got {dim} instead)",
                        case_name="dynamic_shapes_validation",
                    )
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                if isinstance(dim, _Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension #{i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f"but got {dim} instead)",
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
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
) -> List[Constraint]:
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
    symbols: Dict[str, List[Constraint]] = defaultdict(list)
    # track roots that do not directly represent input shape dimensions
    phantom_roots: Dict[str, _PhantomRoot] = {}
    derived_constraints_with_phantom_root: List[_DerivedConstraint] = []
    # list of constraints to return
    constraints: List[Constraint] = []

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
                    vr=ValueRanges(lower=dim.value, upper=dim.value), warn_only=False  # type: ignore[attr-defined]
                ),
            )
        else:
            assert isinstance(dim, _Dim)
            constraint = _Constraint(  # type: ignore[assignment]
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max), warn_only=False  # type: ignore[attr-defined]
                ),
            )
        return constraint

    def update_symbols(path, tensor, shape):
        def _create_static_dim(tensor, i, value):
            return _StaticDim(str(value), (int,), {"value": value})

        # clean out decorators from user side, or previous export call
        # we also delete these attributes in non_strict_utils.py/make_constraints()
        tensor._dynamo_weak_dynamic_indices = set()
        tensor._dynamo_dynamic_indices = set()
        tensor._dynamo_dynamic_range = set()
        tensor._dynamo_static_indices = set()
        tensor._dynamo_unbacked_indices = set()

        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, (int, _Dim)):
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                elif isinstance(dim, _DimHint):
                    if dim == _DimHint.AUTO:
                        torch._dynamo.maybe_mark_dynamic(tensor, i)
                    elif dim == _DimHint.STATIC:
                        torch._dynamo.mark_static(tensor, i)
                    elif dim == _DimHint.DYNAMIC:
                        torch._dynamo.mark_dynamic(tensor, i)
                    constraints.append(_RelaxedConstraint(id(tensor), i))
                elif dim is None:
                    torch._dynamo.mark_static(tensor, i)
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                if isinstance(dim, (int, _Dim)):
                    if isinstance(dim, int):
                        dim = _create_static_dim(tensor, i, dim)
                    constraint = to_constraint(dim, tensor, i)
                    symbols[dim.__name__].append(constraint)
                elif isinstance(dim, _DimHint):
                    if dim == _DimHint.AUTO:
                        torch._dynamo.maybe_mark_dynamic(tensor, i)
                    elif dim == _DimHint.STATIC:
                        torch._dynamo.mark_static(tensor, i)
                    elif dim == _DimHint.DYNAMIC:
                        torch._dynamo.mark_dynamic(tensor, i)
                    constraints.append(_RelaxedConstraint(id(tensor), i))
                elif dim is None:
                    torch._dynamo.mark_static(tensor, i)
        elif shape is None:
            for i in range(tensor.dim()):
                torch._dynamo.mark_static(tensor, i)

    def assoc_shape(path, t, dynamic_shape):
        if isinstance(t, torch.Tensor):
            update_symbols(path, t, dynamic_shape)

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

    return constraints  # type: ignore[return-value]


def _get_dim_name_mapping(
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None]
):
    name_to_dim = {}
    for dim in tree_flatten(
        dynamic_shapes,
        is_leaf=lambda x: isinstance(x, _Dim),
    )[0]:
        if dim is None:
            # NOTE: this must denote a non-Tensor or automatic at this point.
            continue
        if isinstance(dim, int):
            continue
        elif isinstance(dim, _Dim):
            name_to_dim[dim.__name__] = dim
            if isinstance(dim, _DerivedDim):
                name_to_dim[dim.root.__name__] = dim.root  # type: ignore[attr-defined]
        else:
            assert isinstance(dim, _DimHint)
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
                # static, integer
                shape_fixes[name] = int(expr)  # type: ignore[assignment]
            else:
                # relation or derived dim
                shape_fixes[name] = expr

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

    return _tree_map_with_path(apply_fixes, dynamic_shapes, dynamic_shapes)
