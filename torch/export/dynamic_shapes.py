import builtins
import dataclasses
import inspect
import math
import sys
import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch._subclasses.fake_tensor import FakeTensor
from .exported_program import ExportedProgram


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
    Metaclass that ensures a private constructor for :class:`Constraint`
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
    return Constraint._create(w_tensor, t_id, dim, constraint_range, shared, debug_name)


@dataclasses.dataclass
class Constraint(_ConstraintTarget, metaclass=_ConstraintFactory):
    """

    .. warning::
        Do not construct :class:`Constraint` directly, use :func:`dynamic_dim` instead.

    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.

    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: "StrictMinMaxConstraint"  # type: ignore[name-defined]
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
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraint's with logical connectives, "
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
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        if not isinstance(other, Constraint):
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


def dynamic_dim(t: torch.Tensor, index: int, debug_name: Optional[str] = None):
    """
    .. warning::
        (This feature is DEPRECATED. See :func:`Dim` instead.)

    :func:`dynamic_dim` constructs a :class:`Constraint` object that describes the dynamism of
    a dimension ``index`` of tensor ``t``. :class:`Constraint` objects should be passed to
    ``constraints`` argument of :func:`export`.

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A :class:`Constraint` object that describes shape dynamism. It can be passed to :func:`export` so
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


def _process_dynamic_shapes(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> Optional[List[Constraint]]:
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return None

    kwargs = kwargs if kwargs is not None else {}

    from collections.abc import Mapping, Sequence

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
        elif dataclasses.is_dataclass(combined_args):
            if not type(dynamic_shapes) == type(combined_args):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic_shapes of a {type(combined_args)} to be a {type(combined_args)}, "
                    f"got {dynamic_shapes} instead",
                )
            for f in dataclasses.fields(combined_args):
                yield from tree_zip(
                    getattr(combined_args, f.name), getattr(dynamic_shapes, f.name)
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

    def to_constraint(dim, tensor, i):
        constraint = dynamic_dim(tensor, i, debug_name=dim.__name__)
        if dim.min != 2:
            constraint = constraint >= dim.min
        if dim.max != sys.maxsize - 1:
            constraint = constraint <= dim.max
        return constraint

    from collections import defaultdict

    symbols = defaultdict(list)
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
                    symbols[dim.__name__].append(to_constraint(dim, tensor, i))
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
                    symbols[dim.__name__].append(to_constraint(dim, tensor, i))
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
    for dynamic_dims in symbols.values():
        primary, *others = dynamic_dims
        if others:
            for other in others:
                constraints.append(primary == other)
        else:
            constraints.append(primary)

    return constraints


def _process_constraints(
    graph_module: torch.fx.GraphModule,
    num_lifted_params_buffers: int,
    example_inputs: List[torch.Tensor],
) -> Tuple[Dict, List[Tuple[Any, Any]]]:
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

        equality_constraints (List[Tuple[InputDim, InputDim]]): List of tuples
            of (node, dim) to mark that these dimensions are equal.
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

    # Create list of (node name, dim) tuples to mark that they are equal
    equality_constraints: List[Tuple[InputDim, InputDim]] = []
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

            # Accumulate equality constraints
            if shared := constraint.get("shared", None):
                for other_node in tensor_id_to_nodes[shared["t_id"]]:
                    other_node_dim = InputDim(other_node, shared["dim"])
                    equality_constraints.append((node_dim, other_node_dim))

    # Create dict mapping symbol to a singular range (lower, upper)
    range_constraints: Dict[Any, ValueRanges] = {}

    # Add inline constraints to range_constraints
    range_constraints = {
        symbol: inline_constraints[symbol] for symbol in inline_constraints
    }

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
        symbol = symint.node._expr
        range_constraints[symbol] = ValueRanges(min_val, max_val)

    return range_constraints, equality_constraints
