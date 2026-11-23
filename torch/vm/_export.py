"""A graph of the engine as a ``torch.fx.GraphModule``, and what the passes over it make of it.

The engine describes a graph from one run of it on the tensors of a call
(``Path::describe`` in ``torch/csrc/cfg/rust/src/graph.rs``): the tensors it
takes, as they were in that run; its nodes, each with the call site that ran;
and the values it gives. ``build`` makes the ``GraphModule`` of such a
description, applies the passes over a graph in the order of the list, and
returns what the engine then calls in place of the nodes. Nothing is traced
and nothing runs here: there are no fake tensors and no symbolic sizes, and
the modules of a graph compiler are imported by the pass that is one.
"""

import operator
import os
import threading
import traceback
from collections.abc import Callable
from typing import Any

import torch

from . import _types


_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
}


class _State(threading.local):
    # The tensors of the call a graph is being exported from, while its
    # passes run.
    examples: list[torch.Tensor] | None = None


_state = _State()


def _overload(name: str, overload: str) -> Any:
    """``aten::add`` and ``Tensor`` as ``torch.ops.aten.add.Tensor``."""
    namespace, _, op = name.partition("::")
    return getattr(getattr(getattr(torch.ops, namespace), op), overload or "default")


def graph_module(description: dict[str, Any]) -> torch.fx.GraphModule:
    """The graph the engine describes: a placeholder per tensor it takes, a
    call of the operator overload per kernel call with the arguments the
    kernel was given, ``sym_size`` and the functions of ``operator`` for the
    integers it computes, and a tuple of the values it gives as its output:
    what the code behind the graph reads, and then the ranks and sizes that
    the engine checks of the tensors it does not get back.
    The type of what a kernel call returned in the run that was described is
    in the ``meta`` of its node, for :func:`torch.vm.type_of`."""
    graph = torch.fx.Graph()
    values: dict[int, Any] = {}

    def of(operand: tuple[str, Any]) -> Any:
        kind, value = operand
        return values[value] if kind == "v" else value

    def typed(node: Any, meta: Any) -> Any:
        if meta is not None:
            dtype, device, requires_grad, sizes = meta
            dims = tuple(_types.Interval(n, n) for n in sizes)
            found = _types.TensorType(dtype, device, requires_grad, dims)
            node.meta[_types._TYPE] = found
        return node

    for value, name in zip(description["input_ids"], description["input_names"]):
        values[value] = graph.placeholder(name)
    for kind, *parts in description["nodes"]:
        if kind == "aten":
            site, operands, value, meta = parts
            args: list[Any] = []
            kwargs: dict[str, Any] = {}
            for name, keyword_only, source, payload in site["args"]:
                if source == "operand":
                    given = of(operands[payload[0]])
                    if payload[1] == "int_as_list":
                        given = [given]
                elif source == "packed":
                    given = [of(operands[k]) for k in payload]
                else:
                    given = payload
                if keyword_only:
                    kwargs[name] = given
                else:
                    args.append(given)
            target = _overload(site["op"], site["overload"])
            node = graph.call_function(target, tuple(args), kwargs)
            values[value] = typed(node, meta)
        elif kind == "size":
            tensor, dim, value = parts
            size = torch.ops.aten.sym_size.int
            values[value] = graph.call_function(size, (of(tensor), dim))
        elif kind == "dim":
            tensor, value = parts
            rank = torch.ops.aten.dim.default
            values[value] = graph.call_function(rank, (of(tensor),))
        elif kind == "binary":
            symbol, a, b, value = parts
            values[value] = graph.call_function(_BINARY[symbol], (of(a), of(b)))
        elif kind == "neg":
            a, value = parts
            values[value] = graph.call_function(operator.neg, (of(a),))
        elif kind == "tuple":
            items, value = parts
            values[value] = tuple(of(item) for item in items)
        else:
            raise AssertionError(f"the engine describes no node of kind {kind!r}")
    given = [values[value] for value in description["outputs"]]
    asked = {"dim": torch.ops.aten.dim.default, "size": torch.ops.aten.sym_size.int}
    for kind, tensor, *dim in description["exits"]:
        given.append(graph.call_function(asked[kind], (of(tensor), *dim)))
    graph.output(tuple(given))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def examples(input_types: list[Any]) -> list[torch.Tensor]:
    """The tensors a pass that compiles a graph compiles it for: those of the
    call the graph is being exported from, which is what the engine checks
    later calls against (dtype, device, ``requires_grad``, sizes, strides).
    For a pass that is called on its own, tensors of zeros of the types."""
    found = _state.examples
    if found is not None and len(found) == len(input_types):
        return list(found)
    made = []
    for t in input_types:
        sizes = [dim.lo for dim in t.dims]
        zeros = torch.zeros(sizes, dtype=t.dtype, device=t.device)
        made.append(zeros.requires_grad_(t.requires_grad))
    return made


class _Compiled(torch.nn.Module):
    """What a pass compiled, as the one submodule of the graph it returns."""

    def __init__(self, compiled: Callable[..., Any]) -> None:
        super().__init__()
        self.compiled = compiled

    def forward(self, *args: Any) -> Any:
        return self.compiled(*args)


def calling(
    gm: torch.fx.GraphModule, compiled: Callable[..., Any]
) -> torch.fx.GraphModule:
    """A graph with the inputs of ``gm`` whose forward calls ``compiled`` with
    them: what a pass that compiles a graph returns."""
    root = torch.nn.Module()
    root.compiled = _Compiled(compiled)
    graph = torch.fx.Graph()
    names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
    args = tuple(graph.placeholder(name) for name in names)
    graph.output(graph.call_module("compiled", args))
    return torch.fx.GraphModule(root, graph)


def _callable(gm: torch.fx.GraphModule) -> Callable[..., Any]:
    """What the engine calls for a graph: the forward of the graph, or, where
    the graph is one call with its inputs of what a pass compiled, that. Not
    the module: a hook that is registered for every module is not for one
    the engine made."""
    nodes = list(gm.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    rest = nodes[len(placeholders) :]
    if len(rest) != 2 or rest[0].op != "call_module" or rest[1].op != "output":
        return gm.forward
    call, output = rest
    if call.args != tuple(placeholders) or call.kwargs or output.args != (call,):
        return gm.forward
    called = gm.get_submodule(call.target)
    return called.compiled if isinstance(called, _Compiled) else called.forward


def build(
    description: dict[str, Any], passes: tuple[Any, ...], runtime: Any
) -> tuple[Callable[..., Any], list[tuple[int, Any]]] | str:
    """What the passes over a graph make of the graph the engine describes:
    the callable that takes the tensors of the graph and returns a tuple of
    the values it gives, and the guards the passes brought, as (which input,
    check). Or, as a str, why there is none: a pass raised, or returned what
    is no graph of these inputs."""
    gm = graph_module(description)
    inputs = description["inputs"]
    types = [_types.type_of_value(tensor) for tensor in inputs]
    guards: list[tuple[int, Any]] = []
    for pass_ in passes:
        name = getattr(pass_, "__name__", type(pass_).__name__)
        _state.examples = inputs
        try:
            made = pass_(gm, types)
        except Exception as e:
            reason = f"the pass {name} raised {type(e).__name__}: {e}"
            reason = reason.split("\n")[0][:200]
            # Where it was raised, for a pass that raises from inside what
            # it calls.
            frames = traceback.extract_tb(e.__traceback__)
            if len(frames) > 2:
                last = frames[-1]
                reason += f" (in {last.name}, {os.path.basename(last.filename)}:{last.lineno})"
            return reason
        finally:
            _state.examples = None
        if not isinstance(made, torch.fx.GraphModule):
            return f"the pass {name} returned a {type(made).__name__}, not the graph"
        taken = sum(node.op == "placeholder" for node in made.graph.nodes)
        if taken != len(types):
            wanted = len(types)
            return f"the pass {name} returned a graph of {taken} inputs, not {wanted}"
        try:
            guards += _types.guards_of(made)
        except ValueError as e:
            return str(e)[:200]
        made.meta.pop(_types._GUARDS, None)
        made.recompile()
        gm = made
    compiles = any(getattr(pass_, "compiles", False) for pass_ in passes)
    if compiles and any(tensor.is_cuda for tensor in inputs):
        # A kernel on a GPU is started and not waited for: what measures a
        # function against the interpreter waits for the device from here on.
        runtime.trial_settles(torch._C._cuda_synchronize)
    return _callable(gm), guards


def exporter(passes: tuple[Any, ...], runtime: Any) -> Callable[[dict[str, Any]], Any]:
    """What the engine is given to export its graphs with: called with the
    description of a graph, it returns what :func:`build` returns for these
    passes."""

    def export(description: dict[str, Any]) -> Any:
        return build(description, passes, runtime)

    return export
