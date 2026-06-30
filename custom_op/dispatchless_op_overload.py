"""The dispatch-less graph-object layer for custom_op.

``DispatchlessOpOverload`` and its packet are ``torch._ops`` subclasses that are
constructed directly in Python and never enter the C++ dispatcher, so they exist
only to carry a ``FunctionSchema`` and serve as FX graph markers.

Convention: these subclass ``torch._ops`` types. Mark overridden methods with
the ``@override`` decorator, and overriding attribute assignments (where a
decorator can't go) with a ``# @override`` comment. Keep these when editing.
"""

import contextlib
import dataclasses
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
import torch.fx.graph
import torch.utils._pytree as pytree

from .schema import _OVERLOAD_PREFIX, decode_overload_name


if TYPE_CHECKING:
    from .custom_op import CustomOp


class DispatchlessOpOverload(torch._ops.OpOverload):
    """A dispatch-less ``OpOverload`` used purely as an FX graph marker.

    Constructed directly in Python (no ``Library.define``), so it sidesteps the
    dispatcher's 64-argument limit and carries a ``FunctionSchema`` only for
    alias/mutation metadata.
    """

    def __init__(
        self,
        packet: "DispatchlessOpOverloadPacket",
        overload_name: str,
        schema: torch._C.FunctionSchema,
        in_spec: Any,
    ):
        # The overload is a graph-node marker; if ever called directly, route to
        # the CustomOp (normal dispatch happens through the packet).
        def _dispatch(*args: Any, **kwargs: Any) -> Any:
            return packet._custom_op(*args, **kwargs)

        super().__init__(packet, _dispatch, _dispatch, schema, tags=[])
        self._custom_op = packet._custom_op
        self._custom_op_overload_name = overload_name
        self._in_spec = in_spec  # node args are flat; this restructures printouts
        self.__name__ = self._custom_op._name  # @override
        self.__module__ = f"torch._ops.{self._custom_op._ns}"  # @override

    @override
    def name(self) -> str:
        return f"{self._namespace}::{self._opname}"

    @override
    def __repr__(self) -> str:
        return f"<DispatchlessOpOverload(op='{self._namespace}.{self._opname}')>"

    @override
    def __str__(self) -> str:
        return f"{self._namespace}.{self._opname}"

    def _pretty_print(self) -> str:
        return f"torch.ops.{self._namespace}.{self._opname}[*]"


class DispatchlessOpOverloadPacket(torch._ops.OpOverloadPacket):
    def __init__(self, custom_op: "CustomOp"):
        # We skip super().__init__() (its signature builds from a registered op),
        # so we set the OpOverloadPacket fields it would ourselves.
        self._custom_op = custom_op
        self._qualified_op_name = f"{custom_op._ns}::{custom_op._name}"  # @override
        self.__name__ = custom_op._name  # @override
        self._op = None  # @override: no registered dispatcher op
        self._has_torchbind_op_overload = False  # @override
        self.__module__ = f"torch.ops.{custom_op._ns}"  # @override

    @property
    @override
    def _schemas(self) -> dict[str, torch._C.FunctionSchema]:
        return {
            op._custom_op_overload_name: op._schema
            for op in self._custom_op._cache.values()
        }

    @property
    @override
    def _overload_names(self) -> list[str]:
        # On demand: overloads are generated per calling convention (conceptually
        # unbounded), so this is the list materialized (in use) so far.
        return [op._custom_op_overload_name for op in self._custom_op._cache.values()]

    @override
    def __getattr__(self, key: str) -> Any:
        # An overload name is a reversible encoding of the schema + pytree specs, so
        # packet.<name> reconstructs (and registers) the overload even for a calling
        # convention that hasn't been traced yet. _get_or_create_overload dedups, so
        # an already-materialized overload resolves to the same object.
        if not key.startswith(_OVERLOAD_PREFIX):
            raise AttributeError(key)
        try:
            meta = decode_overload_name(key)
            in_spec = pytree.treespec_loads(meta["in_spec"])
            out_spec = (
                None
                if meta["out_spec"] is None
                else pytree.treespec_loads(meta["out_spec"])
            )
        except Exception as e:
            raise AttributeError(key) from e
        return self._custom_op._get_or_create_overload(
            meta["schema"], in_spec, out_spec
        )

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._custom_op(*args, **kwargs)

    @override
    def __hash__(self) -> int:
        return hash(self._qualified_op_name)

    @override
    def __iter__(self):
        # Base iterates self._dir (accessed-so-far overloads); we don't track that,
        # so iterate the materialized overloads instead.
        return iter(self._overload_names)

    @override
    def overloads(self) -> list[str]:
        return self._overload_names


def _register_display_types(value: Any) -> None:
    # A pytree dataclass reprs as its constructor (Box(t=...)) with a bare,
    # unimported name; register the type as an FX custom builtin so that spelling
    # resolves in generated-code globals, making structured gm.code replayable.
    if isinstance(value, (tuple, list)):
        for v in value:
            _register_display_types(v)
    elif isinstance(value, dict):
        for v in value.values():
            _register_display_types(v)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if cls.__name__ not in torch.fx.graph._custom_builtins:
            torch.fx.graph._register_custom_builtin(
                cls.__name__, f"from {cls.__module__} import {cls.__name__}", cls
            )
        for field in dataclasses.fields(value):
            _register_display_types(getattr(value, field.name))


@contextlib.contextmanager
def _structured_display(graph: Any):
    # Node args are flat (for BC); temporarily restore the structured pytree form
    # (from in_spec) for printouts. No-op for flat calls, so compile graphs are
    # untouched.
    saved = []
    for node in graph.nodes:
        if not isinstance(node.target, DispatchlessOpOverload):
            continue
        new_args, new_kwargs = pytree.tree_unflatten(
            list(node.args), node.target._in_spec
        )
        if new_args != node.args or new_kwargs != node.kwargs:
            _register_display_types(new_args)
            _register_display_types(new_kwargs)
            saved.append((node, node.args, node.kwargs))
            node.args, node.kwargs = new_args, new_kwargs
    try:
        yield
    finally:
        for node, args, kwargs in saved:
            node.args, node.kwargs = args, kwargs


def _install_global_hooks() -> None:
    # str(gm.graph) renders a call target via the Node._pretty_print_target
    # staticmethod, which doesn't dispatch to the target -- delegate to ours.
    orig_pretty_print_target = torch.fx.Node._pretty_print_target

    def pretty_print_target(target):
        if isinstance(target, DispatchlessOpOverload):
            return target._pretty_print()
        return orig_pretty_print_target(target)

    torch.fx.Node._pretty_print_target = staticmethod(pretty_print_target)

    # Printouts restore the structured (pytree) arg form; node args stay flat.
    orig_python_code = torch.fx.Graph.python_code
    orig_graph_str = torch.fx.Graph.__str__

    def python_code(self, *args, **kwargs):
        with _structured_display(self):
            return orig_python_code(self, *args, **kwargs)

    def graph_str(self):
        with _structured_display(self):
            return orig_graph_str(self)

    torch.fx.Graph.python_code = python_code
    torch.fx.Graph.__str__ = graph_str


_install_global_hooks()
