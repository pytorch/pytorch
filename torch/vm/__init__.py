"""torch.vm: a function run in a VM as a control-flow graph, with an ordered list of passes.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.vm(fn)`` turns the bytecode of ``fn`` into one control-flow graph in
SSA form, branches and loops included, runs it in a small VM, and applies
passes to it. With no passes the function does what Python does, instruction
for instruction; every pass is a rewrite that leaves the result what it was.
There is no symbolic tracing and there are no fake tensors: the first run of
each basic block is its trace. It runs on the real tensors, with a hook in the
dispatcher recording which kernel every tensor operation resolved to. The
default passes then specialize the block so that those operations call their
post-dispatch kernel directly, guarded on the dispatch keys of their operands::

    @torch.vm
    def f(x, y):
        while x.sum() < 10:  # data-dependent control flow stays in the graph
            x = x * 2 + y
        return x


    f(torch.ones(3), torch.ones(3))
    print(f.dump())  # the SSA graph and its specialized blocks

**The default never changes a result.** What ``@torch.vm`` returns computes
what the function computes, bit for bit: the same kernels, on the same values,
in the same order. A configuration that is faster and may differ from eager in
the last bits (a graph compiler fuses and reorders arithmetic) is never the
default: it is asked for, by naming its passes. :func:`is_exact` tells which of
the two a configuration is.

Nothing is specialized on shapes unless a pass asks for it: every dimension is
dynamic. A pass can narrow a dimension to a range by attaching a guard; when
such a guard fails, only the basic block that carries it is rewritten again.

Calls are followed. A call of a Python function of the same module is inlined
under a guard on the function; any other Python function, method or
``nn.Module`` is rewritten the first time rewritten code calls it; and
``for i in range(k)`` with a small constant ``k`` is unrolled. Closures,
``try``/``except``/``finally``, ``with``, imports and class statements are
handled. A rewritten function runs under a frame of the interpreter that has
its variables, so what looks for them there finds them: ``locals()``, ``eval``,
``exec``, ``super()`` without arguments, a traceback, a debugger. Generators
and coroutines run in the interpreter, and so does what they call; so does what
is called while a trace function (``sys.settrace``) is set. A Python side
effect does not split the function: it is one more instruction of the graph,
between the specialized ones. A whole training step can therefore be one
rewritten function, with the same results bit for bit as the interpreter::

    @torch.vm
    def step(model, opt, x, y):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()  # the autograd engine, on the tape the forward recorded
        opt.step()
        return loss


    step(model, opt, x, y)
    step.stats()["compiled_callees"]  # Module._call_impl, Linear.forward, SGD.step, ...

What is done to a function is one ordered list of passes, ``passes=``, and a
pass says what it is over. One over the control-flow graph rewrites the
function once, before it runs: calls are inlined, loops unrolled. One over a
basic block rewrites a block after its first runs were watched: tensor
operations become direct calls of their kernels. The bare decorator is the
default list::

    @torch.vm(passes=torch.vm.DEFAULT_PASSES)
    def forward(model, x): ...

A list of one's own is the default one with more passes. The engine makes
graphs of the tensor work of each block when it is asked to, and runs them
itself, with the results of eager mode; the pass that hands such a graph to
Inductor is the one to ask for when the last bits of a result may differ::

    P = torch.vm.passes
    passes = (*torch.vm.DEFAULT_PASSES, P.Graphs(), P.Inductor())


    @torch.vm(passes=passes)
    def forward(model, x): ...

A pass that has a setting is made with it::

    # loops of up to 16 iterations, where the default is 8
    torch.vm.passes.Unroll(16)

A pass of one's own goes in the list as well, and its signature says what it
is over. A function of two arguments is a pass over a graph: it gets the
``torch.fx.GraphModule`` of a graph the engine made of the tensor work of a
block and the types of its inputs (:class:`TensorType`: dtype, device, and
per dimension the sizes the graph is for), and returns the graph, rewritten.
What its rewrite is only right for, it says with :func:`guard`: a call that
fails a guard runs the code the pass was given::

    def my_kernel(gm: torch.fx.GraphModule, input_types: list[torch.vm.Type]):
        x = next(node for node in gm.graph.nodes if node.op == "placeholder")
        if 32 not in input_types[0].dim(0):
            return gm  # not the sizes the kernel is for
        torch.vm.guard(gm, x, torch.vm.Dim(0, range(16, 33)))
        for node in gm.graph.nodes:
            if node.target is torch.ops.aten.silu.default:
                node.target = torch.ops.mylib.silu.default
        return gm


    @torch.vm(passes=(*torch.vm.DEFAULT_PASSES, my_kernel))
    def forward(model, x): ...

A callable of one argument is a pass over the instructions of one basic block
of the VM, a ``Region`` (:func:`swap` makes one that puts a kernel in the
place of a run of operations); such a pass can also be written in Rust and
loaded from a library (:func:`load_passes`).
"""

import dataclasses
import functools
import importlib
import inspect
import os
import sys
import types
import warnings
from collections.abc import Iterable
from typing import Any

import torch

from . import _shortcuts, passes
from ._passes import fuse_checked, Mismatch, swap
from ._types import (
    Aligned,
    Check,
    Contiguous,
    Device,
    Dim,
    Dtype,
    guard,
    Interval,
    IntType,
    materialize,
    Stride,
    TensorType,
    Type,
    type_of,
)
from .passes import BuiltinPass, Graphs, Inline, Unroll


__all__ = [
    "Aligned",
    "Check",
    "Contiguous",
    "DEFAULT_PASSES",
    "Device",
    "Dim",
    "Dtype",
    "IntType",
    "Interval",
    "Mismatch",
    "Stride",
    "TensorType",
    "Type",
    "Unsupported",
    "fuse_checked",
    "guard",
    "is_available",
    "is_exact",
    "load_passes",
    "materialize",
    "memory",
    "passes",
    "regions",
    "swap",
    "type_of",
    "vm",
]


class Unsupported(RuntimeError):
    """The function uses something the prototype cannot rewrite."""


DEFAULT_PASSES: tuple[BuiltinPass, ...] = (
    passes.Inline(),
    passes.Unroll(),
    passes.ModuleShortcuts(),
    passes.ResolveDispatch(),
    passes.TensorAccessors(),
    passes.MethodCalls(),
    passes.Fold(),
    passes.DCE(),
)
"""What ``@torch.vm`` does to a function: the passes over its control-flow graph,
once, and then the passes over each basic block once it was profiled, in this
order and again until no pass changes the block anymore."""


_runtime: Any = None


def _load_runtime() -> Any:
    global _runtime
    if _runtime is None:
        try:
            # Not a static import: the extension only exists in builds made
            # with USE_TORCH_VM=1.
            _cfg_rs = importlib.import_module("torch._cfg_rs")
        except ImportError as e:
            raise RuntimeError(
                "torch.vm needs its Rust runtime (torch._cfg_rs), which this build "
                "of PyTorch does not include. Build PyTorch with USE_TORCH_VM=1 "
                "(requires cargo)."
            ) from e
        _cfg_rs.init(torch._C._cfg_c_api())
        _cfg_rs.leave_to_interpreter(_compilers())
        _runtime = _cfg_rs
    return _runtime


def _compilers() -> list[str]:
    """Where the code is that compiles, and that runs what was compiled.

    A rewritten function reaches it when it calls a function that
    ``torch.compile`` made, or when autograd runs the backward of a compiled
    graph. It is not rewritten: the interpreter runs it, with all it calls.
    """
    places = ["_inductor/", "_functorch/_aot_autograd/", "vm/"]
    places += ["_functorch/aot_autograd.py", "_functorch/compilers.py"]
    # What enters Dynamo: the function torch.compile returns.
    entry = "_TorchDynamoContext.__call__.<locals>.compile_wrapper"
    places += [f"_dynamo/eval_frame.py::{entry}"]
    root = os.path.dirname(torch.__file__)
    return [os.path.join(root, place) for place in places]


def load_passes(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads passes written in Rust from a library, by name.

    The library is a ``cdylib`` crate that depends on the crate
    ``torch-vm-pass`` (``torch/csrc/cfg/rust/sdk`` in the source tree),
    implements its ``Pass`` trait and ends with ``export_passes!``. Such a
    pass sees a basic block as a Python pass does (the instructions, the
    kernel each was seen to call, the values they define and use) and can
    replace a run of them by a call. A library links against neither Python
    nor PyTorch, so one that was built once keeps working::

        passes = torch.vm.load_passes("target/release/libmine.so")
        fuse = passes["fuse_rmsnorm"].bind(kernel)  # what the pass puts in
        torch.vm(fn, passes=[fuse, *torch.vm.DEFAULT_PASSES])

    ``bind`` returns the pass with an object that is handed to it on every
    run: the kernel, as a rule, which stays a Python callable.
    """
    return _load_runtime().load_passes(os.fspath(path))


def is_available() -> bool:
    """Whether this build of PyTorch has the torch.vm runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def _over_graph(one: Any) -> bool:
    """Whether a pass is over a graph: one of the engine's says so, and the
    signature of one's own does, ``(gm, input_types) -> gm``."""
    if isinstance(one, BuiltinPass):
        return one.kind == "graph"
    try:
        parameters = inspect.signature(one).parameters.values()
    except (TypeError, ValueError):
        return False
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    needed = [p for p in parameters if p.kind in kinds and p.default is p.empty]
    return len(needed) == 2


def _plan(
    given: Iterable[Any],
) -> tuple[list[Any], dict[str, bool], Graphs | None, dict[str, Any], list[Any]]:
    """Checks a list of passes, and sorts it into what the engine is given.

    Returns the passes over a block in their order (the engine's own by name),
    which of the passes over the control-flow graph are asked for, the graph
    pass if there is one, what the passes were made with, and the passes over
    a graph in their order.
    """
    listed = tuple(given)
    for one in listed:
        if not (isinstance(one, BuiltinPass) or callable(one)):
            kind = type(one).__name__
            raise TypeError(
                f"passes has a {kind}: a pass is one of torch.vm.passes "
                "(torch.vm.passes.Fold()), or a callable taking a region"
            )
    own = [type(one) for one in listed if isinstance(one, BuiltinPass)]
    twice = sorted({kind.__name__ for kind in own if own.count(kind) > 1})
    if twice:
        raise ValueError(f"torch.vm.passes.{twice[0]} is given more than once")
    if Inline in own and Unroll in own and own.index(Unroll) < own.index(Inline):
        raise ValueError(
            "passes has Unroll in front of Inline: the engine inlines first, so "
            "that the loops of an inlined function are unrolled too"
        )
    whole = dict.fromkeys(("inline", "unroll", "shortcuts"), False)
    settings: dict[str, Any] = {}
    for one in listed:
        if isinstance(one, BuiltinPass) and one.kind == "cfg":
            whole[one.engine] = True
            settings.update(dataclasses.asdict(one))

    def said(one: Any) -> str:
        if isinstance(one, BuiltinPass):
            return repr(one)
        return getattr(one, "__name__", type(one).__name__)

    def place(one: Any) -> int:
        return next(i for i, other in enumerate(listed) if other is one)

    found = next((one for one in listed if isinstance(one, Graphs)), None)
    over_graph = [one for one in listed if _over_graph(one)]
    if over_graph and found is None:
        # The graphs they are over are made for them.
        found = Graphs()
        at = place(over_graph[0])
        listed = (*listed[:at], found, *listed[at:])
    for one in over_graph:
        if place(one) < place(found):
            raise ValueError(
                f"passes has {said(one)} in front of {found!r}: a pass over a graph "
                "is given the graphs that Graphs makes, so it goes behind it"
            )
    compiling = [one for one in over_graph if getattr(one, "compiles", False)]
    if len(compiling) > 1:
        raise ValueError(
            f"passes has {said(compiling[0])} and {said(compiling[1])}: a graph is "
            "compiled once, by the last pass over it"
        )
    if compiling and over_graph[-1] is not compiling[0]:
        raise ValueError(
            f"passes has {said(over_graph[-1])} behind {said(compiling[0])}: what a "
            "pass that compiles a graph returns is run, not rewritten, so it is the "
            "last pass over a graph"
        )
    block = [
        one
        for one in listed
        if not (isinstance(one, BuiltinPass) and one.kind == "cfg")
        and all(one is not other for other in over_graph)
    ]
    engine = [one.engine if isinstance(one, BuiltinPass) else one for one in block]
    return engine, whole, found, settings, over_graph


def vm(
    fn: Any = None,
    *,
    passes: Iterable[Any] | None = None,
    fallback: bool = True,
    skip_autograd: bool = True,
    unboxed: bool = True,
    callees: bool = True,
    demote: bool | None = None,
    max_versions: int = 4,
    max_respecializations: int = 6,
    keep_memory: bool | None = None,
) -> Any:
    """Runs ``fn`` as an SSA control-flow graph in the VM, with passes applied.

    ``torch.vm`` itself is this function: ``@torch.vm``, ``@torch.vm(...)`` and
    ``torch.vm(fn, ...)`` all work.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (which is
            called as ``module(...)`` calls it, hooks included). Can be omitted
            to use ``torch.vm`` as a decorator with options.
        passes: what is done to the function: one ordered list. Defaults to
            :data:`DEFAULT_PASSES`. The passes over the control-flow graph of
            the function are applied once, before it runs; the passes over a
            basic block in the order of the list, and again until none changes
            the block, once the block has run and was watched. An entry is one
            of :mod:`torch.vm.passes` (``torch.vm.passes.Fold()``), or a pass
            of one's own, whose signature says what it is over. A function
            ``pass_(gm, input_types) -> gm`` is a pass over a graph: the
            ``torch.fx.GraphModule`` of a graph that
            :class:`torch.vm.passes.Graphs` made of the tensor work of a
            block, with a :class:`Type` per input, to be returned rewritten,
            under the guards it adds with :func:`guard`. The passes over a
            graph go behind ``Graphs`` (which is taken to be in the list when
            one of them is) and are applied in their order; one that compiles
            the graph (:class:`torch.vm.passes.AOTAutograd`,
            :class:`torch.vm.passes.Inductor`) is the last of them. A
            callable ``pass_(region) -> bool`` looks at the instructions of one
            basic block and at what was observed of them, rewrites them, and
            returns whether it changed anything; it can be written in Rust
            (:func:`load_passes`). With an empty list nothing is rewritten: the
            function runs in the generic tier of the VM, which does what the
            interpreter does, without profiling and without direct kernel
            calls.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.
        skip_autograd: let a call site enter below the Autograd kernel when the
            call it was captured from tracked no gradient. Such a site is
            guarded on that still being the case.
        unboxed: keep the tensors that kernels return without a Python object
            until Python code needs one. A tensor that only flows from one
            kernel call into others then never gets one.
        callees: rewrite the Python functions that are called while the
            function runs, the first time each is called, with the same
            options. That is every call of a Python function on the thread,
            whoever makes it: the rewritten code, C code that it calls (the
            ``__init__`` of a class that is instantiated, a property, an
            operator such as ``__add__``, the ``key`` of ``sorted``,
            ``functools.partial``, a hook of autograd), or a function below it
            that the interpreter runs, a generator for one. The calls that C
            code and the interpreter make are caught where the interpreter
            starts a frame (PEP 523), which it does through this runtime only
            while a rewritten function is running (``TORCH_VM_EVAL_FRAME=0``
            leaves those calls to the interpreter). What the runtime itself
            calls is not rewritten: passes, compiled graphs, imports. A callee
            the frontend does not support runs in the interpreter.
        demote: measure the function and each of its callees, once warmed up,
            both rewritten and in the interpreter, and leave to the interpreter
            what runs faster there: the VM gains on tensor operations and loses
            on plain Python, and library code is often mostly the latter.
            ``stats()["demoted_callees"]`` lists the callees that were left,
            ``stats()["demoted"]`` says whether the function itself was, and
            ``stats()["measuring"]`` whether that is still open for the
            function, or for calls that took a noticeable part (1/64) of its
            last call. On by default; the environment variable
            ``TORCH_VM_DEMOTE=0`` turns the default off, for looking at the VM
            itself.
        max_versions: how many specialized versions a block can have at once.
        max_respecializations: how many times a block may be profiled and
            specialized again before it settles on code without guards.
        keep_memory: give the function a workspace: the memory of the CPU
            tensors its calls make stays with it when they are freed, and the
            next call uses it again instead of having the system map and fault
            in the same pages. It holds what the tensors of a call hold at
            their peak and up to half as much again, 256 MB at most; what a few
            calls in a row did not ask for is given back.
            ``torch.vm.memory(fn)`` reports it. On by default
            (``TORCH_VM_KEEP_MEMORY=0`` turns it off).

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text, ``stats()`` and ``sites()``.
    """
    if demote is None:
        demote = os.environ.get("TORCH_VM_DEMOTE", "1") != "0"
    if keep_memory is None:
        keep_memory = os.environ.get("TORCH_VM_KEEP_MEMORY", "1") != "0"
    passes = DEFAULT_PASSES if passes is None else tuple(passes)
    options: dict[str, Any] = {
        "passes": passes,
        "fallback": fallback,
        "skip_autograd": skip_autograd,
        "unboxed": unboxed,
        "callees": callees,
        "demote": demote,
        "max_versions": max_versions,
        "max_respecializations": max_respecializations,
        "keep_memory": keep_memory,
    }
    if fn is None:
        return functools.partial(vm, **options)
    if isinstance(fn, torch.nn.Module):
        module = fn

        # The module is called, not its forward: its hooks run as they do in
        # the interpreter.
        def call(*args: Any, **kwargs: Any) -> Any:
            return module(*args, **kwargs)

        return vm(call, **options)
    if isinstance(fn, types.MethodType):
        return types.MethodType(vm(fn.__func__, **options), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"torch.vm expects a function, not {type(fn).__name__}")
    block, whole, found, settings, over_graph = _plan(passes)
    runtime = _load_runtime()
    export = None
    if over_graph:
        # Only a pass over graphs has the engine hand a graph to Python.
        _export = importlib.import_module("torch.vm._export")
        export = _export.exporter(tuple(over_graph), runtime)
    try:
        return runtime.compile(
            fn,
            block,
            specialize=bool(block),
            skip_autograd=skip_autograd,
            unboxed=unboxed,
            inline=whole["inline"],
            unroll=whole["unroll"],
            max_trip_count=settings.get("max_trip_count", 8),
            callees=callees,
            shortcuts=whole["shortcuts"] and _shortcuts.register(runtime),
            demote=demote,
            max_versions=max_versions,
            max_respecializations=max_respecializations,
            workspace=torch._C._cfg_workspace() if keep_memory else None,
            hook=os.environ.get("TORCH_VM_EVAL_FRAME", "1") != "0",
            graphs=None if found is None else (found.min_ops, found.max_variants),
            export=export,
        )
    except NotImplementedError as e:
        if not fallback:
            raise Unsupported(f"{fn.__qualname__}: {e}") from None
        warnings.warn(f"torch.vm left {fn.__qualname__} as it is: {e}", stacklevel=2)
        return fn


def regions(fn: Any) -> list[dict[str, Any]]:
    """What :class:`passes.Graphs` made of a rewritten function and of what it calls.

    One dict per graph, with ``where`` (function, block and instruction of its
    first operation), ``state``, ``ops`` (how many
    operations it has) and ``operators`` (which, as the dispatcher names
    them), ``inputs`` and ``outputs`` (how many values it takes and gives),
    ``variants`` (how many times it was exported), and ``hits`` and
    ``misses`` (the calls that ran what was made of it, and those that came
    to it and ran its operations). The ``state`` is ``"built"`` for a graph
    the engine runs itself; with a pass over graphs in the list, ``"waiting"``
    for one that is exported at its next call and ``"compiled"`` for one that
    what the passes returned runs in place of (with a ``reason`` if further
    kinds of calls are not exported); and ``"vm"`` for one that is not
    exported, or not for the calls it has seen, with the ``reason``. And one
    dict per instruction or run of instructions with tensor work that is in
    no graph, with ``state`` ``"vm"``, the ``reason``, and for one
    instruction its ``text``.
    """
    fn = getattr(fn, "__func__", fn)
    graphs = getattr(fn, "graphs", None)
    return graphs() if graphs is not None else []


def memory(fn: Any) -> dict[str, int] | None:
    """What a compiled function keeps of the memory of its calls.

    With ``keep_memory`` a compiled function has a workspace: while it runs,
    the CPU tensors that are made get their memory from it and give it back
    to it, so that the next call finds blocks of the sizes it needs instead
    of asking the system for pages again. Returns ``calls``, ``hits`` and
    ``misses`` (blocks found in the workspace, and blocks malloc had to
    make), and in bytes ``kept`` (held for the next call), ``live`` (held
    by tensors now) and ``peak``; or None for a function without one.
    """
    fn = getattr(fn, "__func__", fn)
    workspace = getattr(fn, "workspace", None)
    return None if workspace is None else torch._C._cfg_workspace_stats(workspace)


def is_exact(passes: Iterable[Any] | None = None) -> bool:
    """Whether a list of passes computes what the interpreter computes, bit for
    bit: ``torch.vm.is_exact()`` is true, and with
    ``torch.vm.passes.Inductor()`` in the list it is not.

    True when every pass is one of the engine's own that leaves every result
    what it was: they change how the same kernels are reached, not which ones
    run or in what order. ``torch.vm.passes.Graphs()`` is such a pass: the
    engine runs its graphs with those kernels. A pass that compiles a graph
    is not, and a pass of one's own is not taken to be one.
    """
    listed = DEFAULT_PASSES if passes is None else tuple(passes)
    return all(getattr(one, "exact", False) is True for one in listed)


class _CallableModule(types.ModuleType):
    # torch.vm is both the entry point and the namespace of the prototype:
    # torch.vm(fn) and torch.vm.is_available().
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return vm(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
