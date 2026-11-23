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
import sys
import threading
from collections.abc import Callable
from typing import Any

import torch

from . import _types, passes as listed


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


def _arguments(
    site: dict[str, Any], operand: Callable[[int], Any]
) -> tuple[list[Any], dict[str, Any]]:
    """The arguments a kernel call was given, as the operator takes them:
    ``operand(k)`` for the k-th value the call site takes, the constants of
    the call for the rest."""
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, keyword_only, source, payload in site["args"]:
        if source == "operand":
            given = operand(payload[0])
            if payload[1] == "int_as_list":
                given = [given]
        elif source == "packed":
            given = [operand(k) for k in payload]
        else:
            given = payload
        if keyword_only:
            kwargs[name] = given
        else:
            args.append(given)
    return args, kwargs


def graph_module(description: dict[str, Any]) -> torch.fx.GraphModule:
    """The graph the engine describes: a placeholder per tensor it takes, a
    call of the operator overload per kernel call with the arguments the
    kernel was given (or, for an operation that is no call site, a call of
    the function, method or subscript the code wrote), ``sym_size`` and the
    functions of ``operator`` for the integers it computes, and a tuple of
    the values it gives as its output:
    what the code behind the graph reads, and then the ranks and sizes that
    the engine checks of the tensors it does not get back.
    The type of what a kernel call returned in the run that was described is
    in the ``meta`` of its node, for :func:`torch.vm.type_of`."""
    graph = torch.fx.Graph()
    values: dict[int, Any] = {}

    def of(operand: tuple[str, Any]) -> Any:
        # A value of the graph, a list or a tuple of operands (what the
        # graph takes out of one it is handed), or a constant.
        kind, value = operand
        if kind == "v":
            return values[value]
        if kind == "l":
            return [of(item) for item in value]
        if kind == "t":
            return tuple(of(item) for item in value)
        return value

    def call(operands: list[Any], kwnames: Any) -> tuple[tuple[Any, ...], dict]:
        # The arguments of a call as the function wrote it: the last ones
        # by the names of ``kwnames``.
        given = [of(operand) for operand in operands]
        named = len(kwnames or ())
        positional = given[: len(given) - named]
        return tuple(positional), dict(zip(kwnames or (), given[len(positional) :]))

    def typed(node: Any, meta: Any) -> Any:
        if meta is not None:
            dtype, device, requires_grad, sizes = meta
            dims = tuple(_types.Interval(n, n) for n in sizes)
            found = _types.TensorType(dtype, device, requires_grad, dims)
            node.meta[_types._TYPE] = found
        return node

    taken: set[str] = set()
    for value, name in zip(description["input_ids"], description["input_names"]):
        # A variable of a loop names one input per time round.
        unique, k = name, 1
        while unique in taken:
            k += 1
            unique = f"{name}_{k}"
        taken.add(unique)
        values[value] = graph.placeholder(unique)
    for kind, *parts in description["nodes"]:
        if kind == "aten":
            site, operands, value, meta = parts
            args, kwargs = _arguments(site, lambda k: of(operands[k]))
            target = _overload(site["op"], site["overload"])
            node = graph.call_function(target, tuple(args), kwargs)
            values[value] = typed(node, meta)
        elif kind == "call":
            # An operation that is no call site: the function it called.
            callee, operands, kwnames, value, meta = parts
            positional, named = call(operands, kwnames)
            node = graph.call_function(of(callee), positional, named)
            values[value] = typed(node, meta)
        elif kind == "method":
            name, operands, kwnames, value, meta = parts
            positional, named = call(operands, kwnames)
            values[value] = typed(graph.call_method(name, positional, named), meta)
        elif kind == "getitem":
            _, (tensor, index), _, value, meta = parts
            node = graph.call_function(operator.getitem, (of(tensor), of(index)))
            values[value] = typed(node, meta)
        elif kind == "setitem":
            # A store into a tensor the graph made: it gives no value.
            _, (tensor, index, stored), _, _, _ = parts
            graph.call_function(operator.setitem, (of(tensor), of(index), of(stored)))
        elif kind == "same":
            # The results of an operation that has several, unpacked.
            (several, value) = parts
            values[value] = of(several)
        elif kind == "item":
            several, index, value = parts
            picked = (of(several), of(index))
            values[value] = graph.call_function(operator.getitem, picked)
        elif kind == "attr":
            several, name, value = parts
            values[value] = graph.call_function(getattr, (of(several), name))
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
        elif kind == "list":
            items, value = parts
            values[value] = [of(item) for item in items]
        elif kind == "slice":
            (start, stop, step), value = parts
            values[value] = slice(of(start), of(stop), of(step))
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


def plain(passes: tuple[Any, ...]) -> Callable[..., Callable[..., Any]]:
    """What the engine is given to hand the passes over graphs one of the two
    graphs of a form whose backward it derived (its forward, or the backward
    for the results that came with a gradient). It is called with the
    operations, as the engine's joint graph describes them (per operation
    the operator, its arguments, its keyword arguments, where what it gives
    goes, whether it gives one tensor, and its name), the values the graph
    takes and those it gives, and tensors like those it will be handed;
    it makes the ``GraphModule``, applies the passes, and returns what the
    engine then calls. There is nothing to derive in such a graph: its
    tensors need no gradient."""

    def of(arg: tuple[str, Any], values: dict[int, Any]) -> Any:
        kind, value = arg
        if kind == "v":
            return values[value]
        if kind in ("l", "t"):
            items = [of(item, values) for item in value]
            return items if kind == "l" else tuple(items)
        return value

    def make(
        ops: list[Any],
        takes: list[int],
        gives: list[int],
        examples: tuple[torch.Tensor, ...],
    ) -> Callable[..., Any]:
        graph = torch.fx.Graph()
        values: dict[int, Any] = {}
        for value in takes:
            values[value] = graph.placeholder(f"value_{value}")
        for function, args, kwargs, results, single, name in ops:
            given = [of(arg, values) for arg in args]
            named = {key: of(arg, values) for key, arg in kwargs.items()}
            if callable(function):
                target = function
            else:
                # A kernel the engine took down: its operator, with the
                # arguments that the operator takes by name only by name.
                op, _, overload = name.partition(".")
                target = _overload(op, overload)
                pairs = list(zip(target._schema.arguments, given))
                named.update({arg.name: item for arg, item in pairs if arg.kwarg_only})
                given = [item for arg, item in pairs if not arg.kwarg_only]
            node = graph.call_function(target, tuple(given), named)
            for place, value in results:
                item = (operator.getitem, (node, place))
                values[value] = node if single else graph.call_function(*item)
        graph.output(tuple(values[value] for value in gives))
        made = torch.fx.GraphModule(torch.nn.Module(), graph)
        types = [_types.type_of_value(tensor) for tensor in examples]
        for pass_ in passes:
            _state.examples = list(examples)
            try:
                made = pass_(made, types)
            finally:
                _state.examples = None
            if _types.guards_of(made):
                what = "an input of a graph of the engine's backward"
                raise ValueError(f"a pass asks for a check of {what}")
            made.meta.pop(_types._GUARDS, None)
            made.recompile()
        return _callable(made)

    return make


def build(
    description: dict[str, Any],
    passes: tuple[Any, ...],
    runtime: Any,
    backward: bool = False,
) -> tuple[Callable[..., Any], list[tuple[int, Any]]] | str:
    """What the passes over a graph make of the graph the engine describes:
    the callable that takes the tensors of the graph and returns a tuple of
    the values it gives, and the guards the passes brought, as (which input,
    check). Or, as a str, why there is none: a pass raised, or returned what
    is no graph of these inputs."""
    inputs = description["inputs"]
    graded = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in inputs)
    derived = any(isinstance(pass_, listed.AOTAutograd) for pass_ in passes)
    compiled = any(isinstance(pass_, listed.Inductor) for pass_ in passes)
    if backward and graded and not derived:
        # A graph whose inputs need a gradient gets its backward from the
        # engine (``backward.rs``), unless AOTAutograd is asked for by name.
        # The engine runs the graph itself, as it describes it; with passes
        # over graphs it hands them its two graphs when it has them.
        return runtime.Staged(description, plain(passes) if passes else None), []
    if derived and compiled:
        # AOTAutograd() in front of Inductor(): Inductor compiles the graph
        # and, for tensors that need a gradient, the backward AOTAutograd
        # traces for it. That is one compilation, Inductor's.
        aot = listed.AOTAutograd
        passes = tuple(pass_ for pass_ in passes if not isinstance(pass_, aot))
    elif compiled and graded:
        return (
            "Inductor() compiles graphs whose tensors need no gradient: the engine "
            "derives the backward of a graph (Graphs()) and hands it two such graphs, "
            "or AOTAutograd() in front of it derives one"
        )
    if not passes:
        return "no pass over graphs: the operations run as they are"
    gm = graph_module(description)
    types = [_types.type_of_value(tensor) for tensor in inputs]
    guards: list[tuple[int, Any]] = []
    for pass_ in passes:
        name = getattr(pass_, "__name__", type(pass_).__name__)
        _state.examples = inputs
        try:
            made = pass_(gm, types)
        except Exception as e:
            # What the handler of a signal raised while the pass ran says
            # nothing about the pass: it leaves the call, as it does
            # wherever else in the call the signal comes.
            if runtime.interrupted(e):
                raise
            # What it says is asked only where no code of anybody's makes
            # the text: eager mode never asks it.
            reason = f"the pass {name} raised {runtime.said(e)}"[:200]
            # Where it was raised, for a pass that raises from inside what
            # it calls: read off the traceback as it is, without the lines
            # of the source (a file that is opened is seen by an audit
            # hook).
            last, depth = sys.exc_info()[2], 1
            while last.tb_next is not None:
                last, depth = last.tb_next, depth + 1
            if depth > 2:
                code = last.tb_frame.f_code
                reason += f" (in {code.co_name}, {os.path.basename(code.co_filename)}:{last.tb_lineno})"
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
            return runtime.said(e)[:200]
        made.meta.pop(_types._GUARDS, None)
        made.recompile()
        gm = made
    compiles = any(getattr(pass_, "compiles", False) for pass_ in passes)
    if compiles and any(tensor.is_cuda for tensor in inputs):
        # A kernel on a GPU is started and not waited for: what measures a
        # function against the interpreter waits for the device from here on.
        runtime.trial_settles(torch._C._cuda_synchronize)
    return _callable(gm), guards


def exporter(
    passes: tuple[Any, ...], runtime: Any, backward: bool = False
) -> Callable[[dict[str, Any]], Any]:
    """What the engine is given to export its graphs with: called with the
    description of a graph, it returns what :func:`build` returns for these
    passes. With ``backward``, a graph whose inputs need a gradient gets its
    backward from the engine."""

    def export(description: dict[str, Any]) -> Any:
        return build(description, passes, runtime, backward)

    return export


def _ask_fx() -> None:
    """``torch.fx`` finds some of its classes when the first node of a process
    needs them, by importing their modules. That is done here, where this
    module is imported, and not in the call of a compiled function in which
    the first graph is made: an import calls ``builtins.__import__``, which
    eager mode would not call there."""
    graph = torch.fx.Graph()
    taken = graph.placeholder("taken")
    item = graph.call_function(operator.getitem, ([taken], 0), {"held": {"by": taken}})
    graph.output(item)


_ask_fx()
