# Nodes represent a definition of a value in our graph of operators.
import builtins
import inspect
import logging
import operator
import types
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch._C import _fx_map_aggregate, _fx_map_arg, _NodeBase
from torch.fx.operator_schemas import (
    ArgsKwargsPair,
    normalize_function,
    normalize_module,
)

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
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]

Target = Union[Callable[..., Any], str]

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
_side_effectful_functions: set[Callable[..., Any]] = {
    torch._assert,
    torch._assert_async,
    _ops.aten._assert_async.msg,
    _ops.aten._assert_scalar.default,
    _ops.aten._assert_tensor_metadata.default,
    _ops.aten.sym_constrain_range.default,
    _ops.aten.sym_constrain_range_for_size.default,
    _ops.profiler._record_function_enter,
    _ops.profiler._record_function_enter_new,
    _ops.profiler._record_function_exit,
    _ops.inductor.accumulate_grad_.default,
    operator.setitem,
    *_side_effectful_need_to_be_preserved_pre_dispatch,
}

if hasattr(_ops.inductor, "resize_storage_bytes_"):
    _side_effectful_functions.add(_ops.inductor.resize_storage_bytes_.default)


@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]:
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
    if isinstance(
        func, (types.MethodDescriptorType, types.WrapperDescriptorType)
    ) and func is getattr(torch.Tensor, func.__name__, None):
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

    _args: tuple["Argument", ...]
    _kwargs: dict[str, "Argument"]
    graph: "Graph"
    # unique name of value being created
    name: str
    # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
    op: str
    # for method/module/function, the name of the method/module/function/attr
    # being invoked, e.g add, layer1, or torch.add
    target: "Target"
    # All `Node`-valued inputs. Key is the Node, value is don't-care.
    # The public API for this is `all_input_nodes`, this private attribute
    # should not be accessed directly.
    _input_nodes: dict["Node", None]
    # All of the nodes that use the value produced by this Node
    # Note one user may correspond to several uses, e.g. the node fo ``x + x``
    # would appear once here, but represents two uses.
    # Is a dict to act as an "ordered set". Keys are significant, value dont-care
    users: dict["Node", None]
    # Type expression representing the output value of this node.
    # This should contain the same class of Type objects that would appear
    # as type annotations for function inputs/outputs.
    #
    # For placeholder nodes, this value will be used to type-annotate the
    # generated function parameters.
    # For the return node, this value will be used to type-annotate the
    # generated function return type. (Note this is a special case. ``return``
    # does not produce a value, it's more of a notation. Thus, this value
    # describes the type of args[0] in the ``return`` node.
    type: Optional[Any]
    _sort_key: Any
    # If set, use this fn to print this node
    _repr_fn: Optional[Callable[["Node"], str]]
    # Dictionary to store metadata passes need to do their
    # transformations. This metadata is preserved across node copies
    meta: dict[str, Any]

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: "Target",
        args: tuple["Argument", ...],
        kwargs: dict[str, "Argument"],
        return_type: Optional[Any] = None,
    ) -> None:
        """
        Instantiate an instance of ``Node``. Note: most often, you want to use the
        Graph APIs, i.e. ``Graph.call_module``, ``Graph.call_method``, etc. rather
        than instantiating a ``Node`` directly.

        Args:
            graph (Graph): The ``Graph`` to which this ``Node`` should belong.

            name (str): The name to which the output of this ``Node`` should be assigned

            op (str): The opcode for this ``Node``. Can be one of 'placeholder',
                'call_method', 'call_module', 'call_function', 'get_attr',
                'output'

            target ('Target'): The target this op should call. See the broader
                ``Node`` docstring for more details.

            args (Tuple['Argument']): The args to be passed to ``target``

            kwargs (Dict[str, 'Argument']): The kwargs to be passed to ``target``

            return_type (Optional[Any]): The python type expression representing the
                type of the output of this node. This field can be used for
                annotation of values in the generated code or for other types
                of analyses.
        """
        if op == "call_function":
            if not callable(target):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a Callable is expected"
                )
        else:
            assert op in _legal_ops
            if not isinstance(target, str):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a str is expected"
                )
        super().__init__(graph, name, op, target, return_type)
        self._update_args_kwargs(args, kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "graph": self.graph,
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "type": self.target,
            "_sort_key": self._sort_key,
            "_args": self._args,
            "_kwargs": self._kwargs,
            "_erased": self._erased,
            "_prev": self._prev,
            "_next": self._next,
            "_input_nodes": self._input_nodes,
            "users": self.users,
            "_repr_fn": self._repr_fn,
            "meta": self.meta,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def next(self) -> "Node":
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    @property
    def prev(self) -> "Node":
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev

    @compatibility(is_backward_compatible=True)
    def prepend(self, x: "Node") -> None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """
        assert self.graph == x.graph, "Attempting to move a Node into a different Graph"
        if self == x:
            log.debug(
                "Trying to prepend a node to itself. This behavior has no effect on the graph."
            )
            return
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

        # compute x._sort_key
        psk = x._prev._sort_key
        nsk = x._next._sort_key
        if len(psk) > len(nsk):
            idx: int
            *prefix, idx = psk[: len(nsk) + 1]
            x._sort_key = (*prefix, idx + 1)
        elif len(psk) < len(nsk):
            *prefix, idx = nsk[: len(psk) + 1]
            x._sort_key = (*prefix, idx - 1)
        else:  # same length, increase length by 1
            x._sort_key = (*psk, 0)

    def __gt__(self, other: "Node") -> bool:
        return self._sort_key > other._sort_key

    def __lt__(self, other: "Node") -> bool:
        return self._sort_key < other._sort_key

    def __ge__(self, other: "Node") -> bool:
        return self > other or self == other

    def __le__(self, other: "Node") -> bool:
        return self < other or self == other

    @compatibility(is_backward_compatible=True)
    def append(self, x: "Node") -> None:
        """
        Insert ``x`` after this node in the list of nodes in the graph.
        Equivalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """
        self._next.prepend(x)

    def _remove_from_list(self) -> None:
        p, n = self._prev, self._next
        p._next, n._prev = n, p

    @property
    def args(self) -> tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args

    @args.setter
    def args(self, a: tuple[Argument, ...]) -> None:
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.args = new_args`
        self._update_args_kwargs(a, self._kwargs)

    @property
    def kwargs(self) -> dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k: dict[str, Argument]) -> None:
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.kwargs = new_kwargs`
        self._update_args_kwargs(self._args, k)

    @property
    def all_input_nodes(self) -> list["Node"]:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    @compatibility(is_backward_compatible=True)
    def update_arg(self, idx: int, arg: Argument) -> None:
        """
        Update an existing positional argument to contain the new value
        ``arg``. After calling, ``self.args[idx] == arg``.

        Args:

            idx (int): The index into ``self.args`` of the element to update
            arg (Argument): The new argument value to write into ``args``
        """
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)

    @compatibility(is_backward_compatible=True)
    def insert_arg(self, idx: int, arg: Argument) -> None:
        """
        Insert an positional argument to the argument list with given index.

        Args:

            idx (int): The index of the element in ``self.args`` to be inserted before.
            arg (Argument): The new argument value to insert into ``args``
        """
        assert 0 <= idx <= len(self.args), (
            "insert_args index must be between 0 and len(self.args)"
        )
        args_left = self.args[:idx]
        args_right = self.args[idx:]

        self._args = args_left + (arg,) + args_right

        _new_input_nodes: dict[Node, None] = {}
        _fx_map_arg(arg, _new_input_nodes.setdefault)

        for new_use in _new_input_nodes.keys():
            if new_use not in self._input_nodes:
                self._input_nodes.setdefault(new_use)
                new_use.users.setdefault(self)

    @compatibility(is_backward_compatible=True)
    def update_kwarg(self, key: str, arg: Argument) -> None:
        """
        Update an existing keyword argument to contain the new value
        ``arg``. After calling, ``self.kwargs[key] == arg``.

        Args:

            key (str): The key in ``self.kwargs`` of the element to update
            arg (Argument): The new argument value to write into ``kwargs``
        """
        self.kwargs = {**self.kwargs, key: arg}

    @property
    def stack_trace(self) -> Optional[str]:
        """
        Return the Python stack trace that was recorded during tracing, if any.
        When traced with fx.Tracer, this property is usually populated by
        `Tracer.create_proxy`. To record stack traces during tracing for debug purposes,
        set `record_stack_traces = True` on the `Tracer` instance.
        When traced with dynamo, this property will be populated by default by
        `OutputGraph.create_proxy`.

        stack_trace would have the innermost frame at the end of the string.
        """
        return self.meta.get("stack_trace", None)

    @stack_trace.setter
    def stack_trace(self, trace: Optional[str]) -> None:
        self.meta["stack_trace"] = trace

    def __repr__(self) -> str:
        if self._repr_fn:
            return self._repr_fn(self)
        return self.name

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

    @compatibility(is_backward_compatible=True)
    def format_node(
        self,
        placeholder_names: Optional[list[str]] = None,
        maybe_return_typename: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Return a descriptive string representation of ``self``.

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

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a  descriptive string representation of the
                current Node.
        """
        if self.op == "placeholder":
            assert isinstance(self.target, str)
            arg_str = self.target
            arg_str += arg_str + f": {_type_repr(self.type)}" if self.type else ""
            if placeholder_names:
                placeholder_names.append(arg_str)
                return None
            maybe_typename = f"{_type_repr(self.type)} " if self.type else ""
            default_val = "(default=" + str(self.args[0]) + ")" if self.args else ""
            return f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = {self.op}[target={self.target}]{default_val}"
        elif self.op == "get_attr":
            maybe_typename = (
                f"{_type_repr(self.type)} " if self.type is not None else ""
            )
            return (
                f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]"
            )
        elif self.op == "output":
            if self.type and maybe_return_typename:
                maybe_return_typename[0] = f" -> {_type_repr(self.type)}"
            return f"return {self.args[0]}"
        else:
            maybe_typename = (
                f"{_type_repr(self.type)} " if self.type is not None else ""
            )
            return (
                f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]("
                f"args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})"
            )

    @compatibility(is_backward_compatible=True)
    def replace_all_uses_with(
        self,
        replace_with: "Node",
        delete_user_cb: Callable[["Node"], bool] = lambda user: True,
        *,
        propagate_meta: bool = False,
    ) -> list["Node"]:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.

        Returns:

            The list of Nodes on which this change was made.
        """
        if propagate_meta:
            assert len(replace_with.meta) == 0, (
                "Called node.replace_all_uses_with(replace_with, propagate_meta=True), "
                "but replace_with already has .meta keys"
            )
            for k, v in self.meta.items():
                replace_with.meta[k] = v
        to_process = list(self.users)
        skipped = []
        m = self.graph.owning_module
        for use_node in to_process:
            if not delete_user_cb(use_node):
                skipped.append(use_node)
                continue

            def maybe_replace_node(n: Node) -> Node:
                if n == self:
                    return replace_with
                else:
                    return n

            if getattr(m, "_replace_hooks", None):
                for replace_hook in m._replace_hooks:
                    replace_hook(old=self, new=replace_with.name, user=use_node)

            new_args = _fx_map_arg(use_node.args, maybe_replace_node)
            new_kwargs = _fx_map_arg(use_node.kwargs, maybe_replace_node)
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            use_node._update_args_kwargs(new_args, new_kwargs)

        assert len(self.users) - len(skipped) == 0
        return [n for n in to_process if n not in skipped]

    @compatibility(is_backward_compatible=False)
    def is_impure(self, impure_random: bool = True) -> bool:
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Args:
            impure_random (bool): Whether to treat rand op as impure.

        Returns:

            bool: If the op is impure or not.
        """
        if self.op in {"placeholder", "output"}:
            return True

        if self.op == "call_function":
            schema = getattr(self.target, "_schema", None)
            if schema is not None and schema.is_mutable:
                # impure since it mutates inputs
                return True

            if impure_random:
                if getattr(self.target, "_nondeterministic_seeded", False):
                    # impure since it mutates RNG state
                    return True

            return self.target in _side_effectful_functions

        # Check if an impure module.
        if self.op == "call_module":
            assert self.graph.owning_module is not None, (
                "self.graph.owning_module not set for purity check"
            )
            target_mod = self.graph.owning_module.get_submodule(self.target)
            assert target_mod is not None, (
                f"Did not find expected submodule target {self.target}"
            )
            return getattr(target_mod, "_is_impure", False)

        return False

    @compatibility(is_backward_compatible=False)
    def normalized_arguments(
        self,
        root: torch.nn.Module,
        arg_types: Optional[tuple[Any]] = None,
        kwarg_types: Optional[dict[str, Any]] = None,
        normalize_to_only_use_kwargs: bool = False,
    ) -> Optional[ArgsKwargsPair]:
        """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
        if self.op == "call_function":
            assert callable(self.target)
            return normalize_function(
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                arg_types,
                kwarg_types,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )
        elif self.op == "call_module":
            assert isinstance(self.target, str)
            return normalize_module(
                root,
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )

        return None

    @compatibility(is_backward_compatible=True)
    def replace_input_with(self, old_input: "Node", new_input: "Node") -> None:
        """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:

            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """

        def maybe_replace_node(n: Node) -> Node:
            return new_input if n == old_input else n

        m = self.graph.owning_module
        if getattr(m, "_replace_hooks", None):
            for replace_hook in m._replace_hooks:
                replace_hook(old=old_input, new=new_input.name, user=self)

        new_args = _fx_map_arg(self.args, maybe_replace_node)
        new_kwargs = _fx_map_arg(self.kwargs, maybe_replace_node)
        assert isinstance(new_args, tuple)
        assert isinstance(new_kwargs, dict)
        self._update_args_kwargs(new_args, new_kwargs)

    def _rename(self, candidate: str) -> None:
        if candidate == self.name:
            return
        name = self.graph._graph_namespace.create_name(candidate, None)
        self.name = name
        self.graph._graph_namespace._rename_object(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "name" and hasattr(self, "name"):
            m = self.graph.owning_module
            if getattr(m, "_replace_hooks", None):
                assert isinstance(value, str)
                for user in self.users:
                    for replace_hook in m._replace_hooks:
                        replace_hook(old=self, new=value, user=user)
        update = False
        if (
            hasattr(self, name)
            and hasattr(self.graph, "_find_nodes_lookup_table")
            and self in self.graph._find_nodes_lookup_table
        ):
            update = True
            self.graph._find_nodes_lookup_table.remove(self)
        object.__setattr__(self, name, value)
        if update:
            self.graph._find_nodes_lookup_table.insert(self)


@compatibility(is_backward_compatible=True)
def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each Node appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    assert callable(fn), "torch.fx.map_arg(a, fn): fn must be a callable"
    return _fx_map_arg(a, fn)


@compatibility(is_backward_compatible=True)
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each object appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    return _fx_map_aggregate(a, fn)
