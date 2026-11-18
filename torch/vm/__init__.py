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

Nothing is specialized on shapes unless a pass asks for it: every dimension is
dynamic. A pass can narrow a dimension to a range by attaching a guard; when
such a guard fails, only the basic block that carries it is rewritten again.

Calls are followed. A call of a Python function of the same module is inlined
under a guard on the function; any other Python function, method or
``nn.Module`` is rewritten the first time rewritten code calls it; and
``for i in range(k)`` with a small constant ``k`` is unrolled. Closures,
``try``/``except``/``finally``, ``with``, imports and class statements are
handled, and so are ``locals()``, ``vars()`` and ``dir()``, which get the
variables from where rewritten code keeps them. Generators, coroutines and
functions that read their own frame some other way (``eval``, ``exec``,
``super()`` without arguments) run in the interpreter, and so does what they
call. A Python side effect does not split the function: it is one more
instruction of the graph, between the specialized ones. A whole training step
can therefore be one rewritten function, with the same results bit for bit as
the interpreter::

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

A list of one's own is the default one with more passes; the pass that hands
tensor work to Inductor is the one to ask for when the last bits of a result
may differ::

    passes = (*torch.vm.DEFAULT_PASSES, torch.vm.passes.Graphs("inductor"))


    @torch.vm(passes=passes)
    def forward(model, x): ...

A pass of one's own goes in the list as well: a Python callable taking a
``Region``, the instructions of one basic block.
"""

import functools
import importlib
import os
import sys
import types
import warnings
import weakref
from collections.abc import Iterable
from typing import Any

import torch

from . import _regions, _shortcuts, passes
from .passes import BuiltinPass, Graphs, Inline, Unroll


__all__ = ["DEFAULT_PASSES", "Unsupported", "is_available", "passes", "regions", "vm"]


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
        _runtime = _cfg_rs
    return _runtime


def is_available() -> bool:
    """Whether this build of PyTorch has the torch.vm runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def _plan(given: Iterable[Any]) -> tuple[list[Any], dict[str, bool], Graphs | None]:
    """Checks a list of passes, and sorts it into what the engine is given.

    Returns the passes over a block in their order (the engine's own by name),
    which of the passes over the control-flow graph are asked for, and the
    graph pass if there is one.
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
    for one in listed:
        if isinstance(one, BuiltinPass) and one.kind == "cfg":
            whole[one.engine] = True
    found = next((one for one in listed if isinstance(one, Graphs)), None)
    block = [
        one
        for one in listed
        if not (isinstance(one, BuiltinPass) and one.kind == "cfg")
    ]
    # The graph pass sees a block as the function wrote it: it is the first.
    block = [one for one in block if one is found] + [
        one for one in block if one is not found
    ]
    engine = [
        one.engine if isinstance(one, BuiltinPass) and one is not found else one
        for one in block
    ]
    return engine, whole, found


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
) -> Any:
    """Runs ``fn`` as an SSA control-flow graph in the VM, with passes applied.

    ``torch.vm`` itself is this function: ``@torch.vm``, ``@torch.vm(...)`` and
    ``torch.vm(fn, ...)`` all work.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (its
            ``forward`` is taken; calling the result skips module hooks). Can
            be omitted to use ``torch.vm`` as a decorator with options.
        passes: what is done to the function: one ordered list. Defaults to
            :data:`DEFAULT_PASSES`. The passes over the control-flow graph of
            the function are applied once, before it runs; the passes over a
            basic block in the order of the list, and again until none changes
            the block, once the block has run and was watched. An entry is one
            of :mod:`torch.vm.passes` (``torch.vm.passes.Fold()``), or a pass
            of one's own: a callable ``pass_(region) -> bool`` that looks at
            the instructions of one basic block and at what was observed of
            them, rewrites them, and returns whether it changed anything. With
            an empty list nothing is rewritten: the function runs in the
            generic tier of the VM, which does what the interpreter does,
            without profiling and without direct kernel calls.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.
        skip_autograd: let a call site enter below the Autograd kernel when the
            call it was captured from tracked no gradient. Such a site is
            guarded on that still being the case.
        unboxed: keep the tensors that kernels return without a Python object
            until Python code needs one. A tensor that only flows from one
            kernel call into others then never gets one.
        callees: rewrite the Python functions that the rewritten code calls
            (functions, methods, and objects whose ``__call__`` is a Python
            function) the first time each is called, with the same options. A
            callee the frontend does not support runs in the interpreter.
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

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text, ``stats()`` and ``sites()``.
    """
    if demote is None:
        demote = os.environ.get("TORCH_VM_DEMOTE", "1") != "0"
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
    }
    if fn is None:
        return functools.partial(vm, **options)
    if isinstance(fn, torch.nn.Module):
        return types.MethodType(vm(type(fn).forward, **options), fn)
    if isinstance(fn, types.MethodType):
        return types.MethodType(vm(fn.__func__, **options), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"torch.vm expects a function, not {type(fn).__name__}")
    block, whole, found = _plan(passes)
    runtime = _load_runtime()
    regions = None
    if found is not None:
        regions = _regions.make_pass(
            runtime, found.backend, found.min_ops, found.max_variants, 0
        )
        block[0] = regions
    try:
        compiled = runtime.compile(
            fn,
            block,
            specialize=bool(block),
            skip_autograd=skip_autograd,
            unboxed=unboxed,
            inline=whole["inline"],
            unroll=whole["unroll"],
            callees=callees,
            shortcuts=whole["shortcuts"] and _shortcuts.register(runtime),
            demote=demote,
            max_versions=max_versions,
            max_respecializations=max_respecializations,
        )
        if regions is not None:
            _region_passes[compiled] = regions
        return compiled
    except NotImplementedError as e:
        if not fallback:
            raise Unsupported(f"{fn.__qualname__}: {e}") from None
        warnings.warn(f"torch.vm left {fn.__qualname__} as it is: {e}", stacklevel=2)
        return fn


# The region pass of each compiled function that has a backend.
_region_passes: "weakref.WeakKeyDictionary[Any, _regions.Regions]" = (
    weakref.WeakKeyDictionary()
)


def regions(fn: Any) -> list[dict[str, Any]]:
    """What :class:`passes.Graphs` made of a rewritten function and of what it calls.

    One dict per region, with ``where`` (function, block and instructions),
    ``state`` (``"compiled"``, or ``"waiting"`` for a region that has not been
    tried yet), ``variants``, ``compiles`` (how many of the variants went to
    the backend: the others had the graph of an earlier one), ``ops`` and
    ``seconds`` spent tracing and compiling; and one per instruction or run of
    instructions that was left to the VM, with ``state`` ``"vm"`` and the
    ``reason``.
    """
    fn = getattr(fn, "__func__", fn)
    found = _region_passes.get(fn)
    return found.report() if found is not None else []


class _CallableModule(types.ModuleType):
    # torch.vm is both the entry point and the namespace of the prototype:
    # torch.vm(fn) and torch.vm.is_available().
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return vm(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
