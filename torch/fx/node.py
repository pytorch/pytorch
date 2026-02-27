# Nodes represent a definition of a value in our graph of operators.
import builtins
import inspect
import logging
import operator
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, TypeVar

import torch
from torch._C import _fx_map_aggregate, _fx_map_arg, _NodeBase
from torch.utils._dtype_abbrs import dtype_abbrs

from .._ops import ops as _ops
from ._compatibility import compatibility


if TYPE_CHECKING:
    from .graph import Graph

__all__ = ["Node", "map_arg", "map_aggregate", "has_side_effect"]

log = logging.getLogger(__name__)

BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
    torch.SymInt,
    torch.SymBool,
    torch.SymFloat,
]
base_types = typing.get_args(BaseArgumentTypes)

Target: TypeAlias = Union[Callable[..., Any], str]

Argument = Optional[
    Union[
        tuple["Argument", ...],
        Sequence["Argument"],
        Mapping[str, "Argument"],
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "Node",
        BaseArgumentTypes,
    ]
]
# pyrefly: ignore [invalid-annotation]
ArgumentT = TypeVar("ArgumentT", bound=Argument)
_P = ParamSpec("_P")
_R = TypeVar("_R")

_legal_ops = dict.fromkeys(
    [
        "placeholder",
        "call_method",
        "call_module",
        "call_function",
        "get_attr",
        "output",
        "root",
    ]
)

# Dynamo is unable to trace global set[Callable].__contains__.
# See https://github.com/pytorch/pytorch/issues/145761. Since we only have
# a handful of ops so switch to list of callables.
_side_effectful_need_to_be_preserved_pre_dispatch: list[Callable[..., Any]] = [
    torch._C._set_grad_enabled,
    torch.amp._enter_autocast,
    torch.amp._exit_autocast,
]

# TODO: Either refactor this into 2 functions 1 dce for functional graphs and 1 dce for all graphs,
# or add logic to correctly mark all inplace ops as side effectful.
#
# NOTE: For new operators, please do not add to this set!
# Instead, consider using the effects system via
# torch.library._register_effectful_op() for operators.
#
# This _side_effectful_functions set is only for:
# - Legacy functions that aren't operators (e.g., profiler ops, asserts)
# - Things that cannot be marked via the normal effects system
_side_effectful_functions: set[Callable[..., Any]] = {
    torch._assert,
    torch._assert_async,
    _ops.aten._assert_async.msg,
    _ops.aten._assert_scalar.default,
    _ops.aten._assert_tensor_metadata.default,
    _ops.aten.sym_constrain_range.default,
    _ops.aten.sym_constrain_range_for_size.default,
    _ops.profiler._record_function_enter,
    _ops.profiler._record_function_enter.default,
    _ops.profiler._record_function_enter_new,
    _ops.profiler._record_function_enter_new.default,
    _ops.profiler._record_function_exit,
    _ops.profiler._record_function_exit._RecordFunction,
    _ops.inductor.accumulate_grad_.default,
    operator.setitem,
    *_side_effectful_need_to_be_preserved_pre_dispatch,
}

if hasattr(_ops.inductor, "resize_storage_bytes_"):
    _side_effectful_functions.add(_ops.inductor.resize_storage_bytes_.default)


@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Registers a function to not be dead code eliminated by
    fx.graph.eliminate_dead_code

    NOTE: For new operators, please do not add to this set!
    Instead, consider using the effects system via
    torch.library._register_effectful_op() for operators.

    This _side_effectful_functions set is only for:
    - Legacy functions that aren't operators (e.g., profiler ops, asserts)
    - Things that cannot be marked via the normal effects system
    """
    _side_effectful_functions.add(fn)
    return fn


# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f"cannot find module for {orig_method}")


# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj: object) -> str:
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # Extension: If we don't ignore GenericAlias then `list[int]` will print
    # simply "list".
    if isinstance(obj, type) and not isinstance(obj, types.GenericAlias):
        if obj.__module__ == "builtins":
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    # torch.Tensor.{fn}
    if (
        isinstance(func, (types.MethodDescriptorType, types.WrapperDescriptorType))
        and func is getattr(torch.Tensor, func.__name__, None)
    ) or (
        func.__module__ == torch._tensor.__name__
        and func.__qualname__ == f"Tensor.{func.__name__}"
    ):
        return f"torch.Tensor.{func.__name__}"
    name = func.__name__

    if name == "<lambda>":
        # For lambdas, try to get their defining name in the module
        try:
            name = inspect.getsource(func).split("=")[0].strip()
        except Exception as e:
            raise RuntimeError("Unable to represent lambda") from e
    module = _find_module_of_method(func)
    module = module.replace(
        "torch._ops", "torch.ops"
    )  # WAR for bug in how torch.ops assigns module
    # Fixup segment_reduce mismatch
    if module == "torch" and name == "segment_reduce":
        name = "_" + name
    if module == "torch.nn.functional" and name in ("_ScalingType", "_SwizzleType"):
        name = name.removeprefix("_")
    return f"{module}.{name}"


def _format_arg(arg: object, max_list_len: float = float("inf")) -> str:
    if hasattr(arg, "_custom_fx_repr_fn"):
        return arg._custom_fx_repr_fn()
    elif isinstance(arg, list):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        return f"[{items}{maybe_len}]"
    elif isinstance(arg, tuple):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        maybe_comma = "," if len(arg) == 1 else ""
        return f"({items}{maybe_comma}{maybe_len})"
    elif isinstance(arg, dict):
        items_str = ", ".join(f"{k}: {_format_arg(v)}" for k, v in arg.items())
        return f"{{{items_str}}}"

    if isinstance(arg, Node):
        return "%" + str(arg)
    else:
        return str(arg)


# Node is now fully implemented in C++
# We create an alias here for backward compatibility and to add docstrings/type hints
@compatibility(is_backward_compatible=True)
class Node(_NodeBase):
    """
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """

    # Type annotations for the members (some implemented in C++ via _NodeBase)
    _args: tuple["Argument", ...]
    _kwargs: dict[str, "Argument"]
    graph: "Graph"
    name: str
    op: str
    target: "Target"
    _input_nodes: dict["Node", None]
    users: dict["Node", None]
    type: Optional[Any]
    _sort_key: Any
    _repr_fn: Optional[Callable[["Node"], str]]
    meta: dict[str, Any]

    # Properties implemented in C++ (_NodeBase) - type stubs in torch/_C/__init__.pyi
    args: tuple["Argument", ...]
    kwargs: dict[str, "Argument"]
    next: "Node"
    prev: "Node"
    all_input_nodes: list["Node"]
    stack_trace: Optional[str]

    @staticmethod
    def _pretty_print_target(target: object) -> str:
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
        if isinstance(target, str):
            return target
        if hasattr(target, "__module__"):
            name = getattr(target, "__name__", None)
            if name is None:
                # Just to be defensive, if we don't have `__name__`, get the
                # qualname. Not sure if this happens for any members of `operator`
                # or `builtins`. This fallback path is not as good, since e.g.
                # things in `operator` have `_operator` as their __module__.
                # TODO: THIS IS BROKEN: _get_qualified_name calls `__name__`
                return _get_qualified_name(target)  # type: ignore[arg-type]
            if target.__module__ == "builtins":
                return f"builtins.{name}"
            elif target.__module__ == "_operator":
                return f"operator.{name}"
        return _get_qualified_name(target)  # type: ignore[arg-type]


# Helper function for format_node, called from C++
def _format_node_impl(
    node: Node,
    placeholder_names: Optional[list[str]] = None,
    maybe_return_typename: Optional[list[str]] = None,
    *,
    include_tensor_metadata: bool = False,
) -> Optional[str]:
    """
    Return a descriptive string representation of ``node``.

    This method can be used with no arguments as a debugging
    utility.

    This function is also used internally in the ``__str__`` method
    of ``Graph``. Together, the strings in ``placeholder_names``
    and ``maybe_return_typename`` make up the signature of the
    autogenerated ``forward`` function in this Graph's surrounding
    GraphModule. ``placeholder_names`` and ``maybe_return_typename``
    should not be used otherwise.

    Args:
        placeholder_names: A list that will store formatted strings
            representing the placeholders in the generated
            ``forward`` function. Internal use only.
        maybe_return_typename: A single-element list that will store
            a formatted string representing the output of the
            generated ``forward`` function. Internal use only.
        include_tensor_metadata: Whether to include tensor metadata

    Returns:
        str: If 1) we're using ``format_node`` as an internal helper
            in the ``__str__`` method of ``Graph``, and 2) ``node``
            is a placeholder Node, return ``None``. Otherwise,
            return a  descriptive string representation of the
            current Node.
    """
    if node.op == "placeholder":
        arg_str = str(node.target)
        arg_str += arg_str + f": {_type_repr(node.type)}" if node.type else ""
        if placeholder_names:
            placeholder_names.append(arg_str)
            return None
        maybe_typename = f"{_type_repr(node.type)} " if node.type else ""
        default_val = "(default=" + str(node.args[0]) + ")" if node.args else ""
        return f"%{node.name} : {maybe_typename}[num_users={len(node.users)}] = {node.op}[target={node.target}]{default_val}"
    elif node.op == "get_attr":
        maybe_typename = f"{_type_repr(node.type)} " if node.type is not None else ""
        return (
            f"%{node.name} : {maybe_typename}[num_users={len(node.users)}] = "
            f"{node.op}[target={Node._pretty_print_target(node.target)}]"
        )
    elif node.op == "output":
        if node.type and maybe_return_typename:
            maybe_return_typename[0] = f" -> {_type_repr(node.type)}"
        return f"return {node.args[0]}"
    else:

        def stringify_shape(shape: Iterable) -> str:
            return f"[{', '.join([str(x) for x in shape])}]"

        meta_val = node.meta.get(
            "val",
            node.meta.get("tensor_meta", node.meta.get("example_value", None)),
        )
        type_annotation = ""
        if (
            include_tensor_metadata
            and isinstance(meta_val, torch.Tensor)
            and meta_val.layout
            not in (
                torch.sparse_csc,
                torch.sparse_csr,
            )
        ):
            stride_annotation = f"{stringify_shape(meta_val.stride())}"
            device_annotation = f"{meta_val.device}"
            type_annotation = (
                f'Tensor "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}'
                f'{stride_annotation}{device_annotation}"'
            )
        else:
            type_annotation = (
                f"{_type_repr(node.type)} " if node.type is not None else ""
            )
        return (
            f"%{node.name} : {type_annotation}[num_users={len(node.users)}] = "
            f"{node.op}[target={Node._pretty_print_target(node.target)}]("
            f"args = {_format_arg(node.args)}, kwargs = {_format_arg(node.kwargs)})"
        )


@compatibility(is_backward_compatible=True)
def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each Node appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    if not callable(fn):
        raise AssertionError("torch.fx.map_arg(a, fn): fn must be a callable")
    return _fx_map_arg(a, fn)


@compatibility(is_backward_compatible=True)
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each object appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    return _fx_map_aggregate(a, fn)
