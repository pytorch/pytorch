"""torch.rewrite: a function as a control-flow graph, and a list of rewrites.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.rewrite(fn)`` turns the bytecode of ``fn`` into one control-flow
graph in SSA form, branches and loops included, runs it in a small VM, and
applies a list of passes to it. With no passes the function does what
Python does, instruction for instruction; every pass is a rewrite that
leaves the result what it was. There is no symbolic tracing and there are no
fake tensors: the first run of each basic block is its trace. It runs on the
real tensors, with a hook in the dispatcher recording which kernel every
tensor operation resolved to. The standard passes then specialize the block
so that those operations call their post-dispatch kernel directly, guarded
on the dispatch keys of their operands::

    @torch.rewrite
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
handled. A rewritten function runs under a frame of the interpreter that has
its variables, so what looks for them there finds them: ``locals()``,
``eval``, ``exec``, ``super()`` without arguments, a traceback, a debugger.
Generators and coroutines run in the interpreter, and so does what they
call; so does what is called while a trace function (``sys.settrace``) is
set. A Python side effect does not split the function: it is
one more instruction of the graph, between the specialized ones. A whole
training step can therefore be one rewritten function, with the same results
bit for bit as the interpreter::

    @torch.rewrite
    def step(model, opt, x, y):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()  # the autograd engine, on the tape the forward recorded
        opt.step()
        return loss


    step(model, opt, x, y)
    step.stats()["compiled_callees"]  # Module._call_impl, Linear.forward, SGD.step, ...

What is done to a function is its list of passes, ``passes=``, in the order
they are applied. :data:`canonical` is the standard list;
:data:`inductor` is the pass that hands tensor work to AOTAutograd and
Inductor; and a pass of one's own is a Python callable taking a ``Region``,
or one written in Rust and loaded from a library::

    @torch.rewrite(passes=[torch.rewrite.inductor, *torch.rewrite.canonical])
    def forward(model, x): ...
"""

import functools
import importlib
import os
import sys
import types
import warnings
import weakref
from collections.abc import Callable, Iterable
from typing import Any

import torch

from . import _regions, _shortcuts
from ._passes import Builtin, fuse_checked, Mismatch, swap


__all__ = [
    "Graphs",
    "Mismatch",
    "Unsupported",
    "canonical",
    "graphs",
    "fuse_checked",
    "inductor",
    "is_available",
    "load_passes",
    "memory",
    "regions",
    "rewrite",
    "swap",
]


class Unsupported(RuntimeError):
    """The function uses something the prototype cannot rewrite."""


class Graphs:
    """The pass that hands tensor work to a graph compiler; see :func:`graphs`."""

    __slots__ = ("backend", "freeze", "max_variants", "min_ops")

    def __init__(
        self,
        backend: str | Callable[..., Any],
        min_ops: int = 4,
        max_variants: int = 32,
        freeze: bool | None = None,
    ) -> None:
        self.backend = backend
        self.min_ops = min_ops
        self.max_variants = max_variants
        self.freeze = freeze

    def __repr__(self) -> str:
        name = self.backend if isinstance(self.backend, str) else "a callable"
        return f"torch.rewrite.graphs({name!r})"


def graphs(
    backend: str | Callable[..., Any] = "inductor",
    *,
    min_ops: int = 4,
    max_variants: int = 32,
    freeze: bool | None = None,
) -> Graphs:
    """A pass that hands the tensor work of each basic block to a graph compiler.

    ``"inductor"`` traces every run of instructions of a block that a trace
    can capture, together with what those instructions call, into one ATen
    graph, and compiles it with AOTAutograd and Inductor: with gradients
    enabled, the tensors such a region returns have one compiled autograd
    function as their ``grad_fn``. The outermost region that traces is the
    one that is compiled; where an instruction does not trace, the VM runs it
    and the blocks of what it calls are tried in turn. ``"aot_eager"`` builds
    the same graphs and runs them with the eager kernels. A callable
    ``backend(graph_module, example_inputs)`` returning a callable is used as
    given. ``torch.rewrite.regions(fn)`` reports what was compiled and what
    was left to the VM, with the reason. See ``torch/rewrite/_regions.py``
    for what a compiled region checks on every call and what it does not.

    The pass traces the instructions as the function wrote them, so it comes
    before the passes that rewrite single instructions:
    ``passes=[torch.rewrite.inductor, *torch.rewrite.canonical]``.

    Args:
        backend: see above.
        min_ops: a region with fewer tensor operations than this is left to
            the VM, whose direct kernel calls are cheaper than entering a
            compiled graph.
        max_variants: how many graphs a region gives the backend before
            further kinds of inputs (dtypes, ranks, modules, modes) run its
            instructions as they are. Inputs that trace to the same graph, as
            the layers of a model do when a region is called with one after
            the other, share what the backend made of it, and sizes that
            differ from call to call are left open in one graph rather than
            compiled one by one.
        freeze: with ``"inductor"``, build the parameters and buffers of the
            modules a region reads into its graph when the call is one of
            inference (no gradients recorded, every module in eval mode), so
            that Inductor folds what follows from them alone (a batch norm
            into the convolution before it) and lays convolution weights out
            for the kernels. Such a graph is checked on every call: a tensor
            that was written to since (in place, through ``.data``, by an
            optimizer) or replaced makes the region compile again with that
            tensor as an input of the graph. Not built in: a tensor the
            region itself writes, the matrix of a matrix product or of an
            embedding lookup (nothing folds, and it would be a second copy),
            and memory PyTorch does not see the writes to (shared with
            another process, or a NumPy array's). ``regions()`` reports
            ``frozen`` per region. On by default
            (``TORCH_REWRITE_FREEZE=0`` turns it off).
    """
    return Graphs(backend, min_ops, max_variants, freeze)


Pass = str | Callable[[Any], bool] | Graphs | Builtin

# The passes that rewrite the whole function once, before the others see it.
_WHOLE = ("inline", "unroll", "shortcuts")
# The built-in passes over a basic block.
_BLOCK = (
    "resolve_dispatch",
    "tensor_accessors",
    "method_calls",
    "narrow_dims",
    "fold",
    "dce",
)

canonical: tuple[Pass, ...] = (
    "inline",
    "unroll",
    "shortcuts",
    "resolve_dispatch",
    "tensor_accessors",
    "method_calls",
    "fold",
    "dce",
)
"""The standard passes, in the order they are applied.

Three rewrite the function as a whole, once:

- ``inline``: calls of Python functions defined in the same module are
  replaced by their graph, under a guard on the callee; the call itself
  stays as the other side of that guard.
- ``unroll``: ``for i in range(k)`` loops with a small constant ``k`` are
  replaced by ``k`` copies of their body, under a guard on ``range``.
- ``shortcuts``: the code of ``nn.Module`` that sits in front of every call
  of a module and every read of a parameter, buffer or submodule is skipped
  whenever a few checks show what it would do: a call of a module without
  hooks goes straight to its ``forward``, and the three dicts are looked at
  without calling ``Module.__getattr__``. ``Module.__call__`` then does not
  appear on the stack.

The others run over each basic block once it was profiled, in order, and
the whole list is repeated until no pass changes the block anymore:

- ``resolve_dispatch``: operations the profile found to be a single
  dispatcher call become direct calls of the post-dispatch kernel.
- ``tensor_accessors``: ``x.shape[i]``, ``x.size(i)``, ``x.dim()`` and
  ``x.ndim`` on tensors become plain SSA integers.
- ``method_calls``: ``x.f(...)`` calls that stay generic are made without
  creating a bound method object.
- ``fold``: folds truthiness tests and branches on constants.
- ``dce``: drops instructions whose result is not needed.

One more built-in pass is not in the list: ``narrow_dims`` folds a comparison
of a dimension with a constant to the outcome last observed, under a guard
that pins the dimension to the range of sizes with that outcome. It goes in
front of ``fold``.
"""

inductor: Graphs = Graphs("inductor")
"""``graphs("inductor")``: tensor work goes to AOTAutograd and Inductor."""


_runtime: Any = None


def _load_runtime() -> Any:
    global _runtime
    if _runtime is None:
        try:
            # Not a static import: the extension only exists in builds made
            # with USE_CFG_COMPILE=1.
            _cfg_rs = importlib.import_module("torch._cfg_rs")
        except ImportError as e:
            raise RuntimeError(
                "torch.rewrite needs its Rust runtime (torch._cfg_rs), which this build "
                "of PyTorch does not include. Build PyTorch with USE_CFG_COMPILE=1 "
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
    places = ["_inductor/", "_functorch/_aot_autograd/", "rewrite/"]
    places += ["_functorch/aot_autograd.py", "_functorch/compilers.py"]
    # What enters Dynamo: the function torch.compile returns.
    entry = "_TorchDynamoContext.__call__.<locals>.compile_wrapper"
    places += [f"_dynamo/eval_frame.py::{entry}"]
    root = os.path.dirname(torch.__file__)
    return [os.path.join(root, place) for place in places]


def load_passes(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads passes written in Rust from a library, by name.

    The library is a ``cdylib`` crate that depends on the crate
    ``torch-rewrite-pass`` (``torch/csrc/cfg/rust/sdk`` in the source tree),
    implements its ``Pass`` trait and ends with ``export_passes!``. Such a
    pass sees a basic block as a Python pass does (the instructions, the
    kernel each was seen to call, the values they define and use) and can
    replace a run of them by a call. A library links against neither Python
    nor PyTorch, so one that was built once keeps working::

        passes = torch.rewrite.load_passes("target/release/libmine.so")
        fuse = passes["fuse_rmsnorm"].bind(kernel)  # what the pass puts in
        torch.rewrite(fn, passes=[fuse, *torch.rewrite.canonical])

    ``bind`` returns the pass with an object that is handed to it on every
    run: the kernel, as a rule, which stays a Python callable.
    """
    return _load_runtime().load_passes(os.fspath(path))


def is_available() -> bool:
    """Whether this build of PyTorch has the torch.rewrite runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def _plan(
    passes: Iterable[Pass] | None,
) -> tuple[dict[str, bool], list[Any], Graphs | None]:
    """Sorts a list of passes into what the engine is given.

    Returns which of the passes over the whole function are asked for, the
    passes over a block in their order, and which of those is the graph pass.
    """
    whole = dict.fromkeys(_WHOLE, False)
    block: list[Any] = []
    found: Graphs | None = None
    for one in canonical if passes is None else passes:
        if isinstance(one, Builtin):
            # What torch.vm makes.
            one = one.name
        if isinstance(one, Graphs):
            if found is not None:
                raise ValueError(f"passes= has {found!r} and {one!r}: one of them")
            builtin = next((p for p in block if isinstance(p, str)), None)
            if builtin is not None:
                raise ValueError(
                    f"{one!r} comes after {builtin!r} in passes=: it traces the "
                    "instructions as the function wrote them, so it goes in front "
                    "of the passes that rewrite single instructions"
                )
            found = one
            block.append(one)
        elif isinstance(one, str):
            if one in whole:
                whole[one] = True
            elif one in _BLOCK:
                block.append(one)
            else:
                names = ", ".join((*_WHOLE, *_BLOCK))
                raise ValueError(
                    f"unknown pass {one!r}; the built-in passes are {names}"
                )
        elif callable(one):
            block.append(one)
        else:
            raise TypeError(
                f"a pass is a name, a callable or torch.rewrite.graphs(...), not {type(one).__name__}"
            )
    return whole, block, found


def rewrite(
    fn: Any = None,
    *,
    passes: Iterable[Pass] | None = None,
    fallback: bool = True,
    skip_autograd: bool = True,
    unboxed: bool = True,
    callees: bool = True,
    demote: bool | None = None,
    max_versions: int = 4,
    max_respecializations: int = 6,
    keep_memory: bool | None = None,
) -> Any:
    """Runs ``fn`` as an SSA control-flow graph with a list of rewrites applied.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (whose
            ``forward`` is rewritten; calling the result skips module hooks).
            Can be omitted to use ``rewrite`` as a decorator with options.
        passes: what is done to the function, in the order it is done.
            Defaults to :data:`canonical`. An entry is the name of a built-in
            pass (or what ``torch.vm`` makes for one, such as
            ``torch.vm.inlining_pass()``), ``torch.rewrite.graphs(...)``,
            or a callable
            ``pass_(region) -> bool`` that looks at the instructions of one
            basic block and at what was observed of them, rewrites them, and
            returns whether it changed anything. With an empty list nothing
            is rewritten: the function runs in the generic tier of the VM,
            which does what the interpreter does, without profiling and
            without direct kernel calls.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.
        skip_autograd: let a call site enter below the Autograd kernel when
            the call it was captured from tracked no gradient. Such a site is
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
            ``functools.partial``, a hook of autograd), or a function below
            it that the interpreter runs, a generator for one. The calls
            that C code and the interpreter make are caught where the
            interpreter starts a frame (PEP 523), which it does through this
            runtime only while a rewritten function is running
            (``TORCH_REWRITE_EVAL_FRAME=0`` leaves those calls to the
            interpreter). What the runtime itself calls is not rewritten:
            passes, compiled graphs, imports.
        demote: measure the function and each of its callees, once warmed up,
            both rewritten and in the interpreter, and leave to the interpreter
            what runs faster there: the VM gains on tensor operations and loses
            on plain Python, and library code is often mostly the latter.
            ``stats()["demoted_callees"]`` lists the callees that were left,
            ``stats()["demoted"]`` says whether the function itself was, and
            ``stats()["measuring"]`` whether that is still open for the
            function, or for calls that took a noticeable part (1/64) of
            its last call.
            On by default; the environment variable
            ``TORCH_REWRITE_DEMOTE=0`` turns the default off, for looking
            at the VM itself.
            A callee the frontend does not support runs in the interpreter.
        max_versions: how many specialized versions a block can have at once.
        max_respecializations: how many times a block may be profiled and
            specialized again before it settles on code without guards.
        keep_memory: give the function a workspace: the memory of the CPU
            tensors its calls make stays with it when they are freed, and
            the next call uses it again instead of having the system map and
            fault in the same pages. It holds what the tensors of a call
            hold at their peak and up to half as much again, 256 MB at
            most; what a few calls in a row did not ask for is given back.
            ``torch.rewrite.memory(fn)`` reports it. On by
            default (``TORCH_REWRITE_KEEP_MEMORY=0`` turns it off).

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text, ``stats()`` and ``sites()``.
    """
    if demote is None:
        demote = os.environ.get("TORCH_REWRITE_DEMOTE", "1") != "0"
    if keep_memory is None:
        keep_memory = os.environ.get("TORCH_REWRITE_KEEP_MEMORY", "1") != "0"
    if passes is not None:
        passes = tuple(passes)
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
        return functools.partial(rewrite, **options)
    if isinstance(fn, torch.nn.Module):
        return types.MethodType(rewrite(type(fn).forward, **options), fn)
    if isinstance(fn, types.MethodType):
        return types.MethodType(rewrite(fn.__func__, **options), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"torch.rewrite expects a function, not {type(fn).__name__}")
    whole, block, found = _plan(passes)
    runtime = _load_runtime()
    regions = None
    if found is not None:
        freeze = found.freeze
        if freeze is None:
            freeze = os.environ.get("TORCH_REWRITE_FREEZE", "1") != "0"
        regions = _regions.make_pass(
            runtime, found.backend, found.min_ops, found.max_variants, 0, freeze
        )
        block = [regions if one is found else one for one in block]
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
            workspace=torch._C._cfg_workspace() if keep_memory else None,
            hook=os.environ.get("TORCH_REWRITE_EVAL_FRAME", "1") != "0",
        )
        if regions is not None:
            _region_passes[compiled] = regions
        return compiled
    except NotImplementedError as e:
        if not fallback:
            raise Unsupported(f"{fn.__qualname__}: {e}") from None
        warnings.warn(
            f"torch.rewrite left {fn.__qualname__} as it is: {e}", stacklevel=2
        )
        return fn


# The region pass of each compiled function that has a backend.
_region_passes: "weakref.WeakKeyDictionary[Any, _regions.Regions]" = (
    weakref.WeakKeyDictionary()
)


def regions(fn: Any) -> list[dict[str, Any]]:
    """What a ``graphs`` pass made of a rewritten function and of what it calls.

    One dict per region, with ``where`` (function, block and instructions),
    ``state`` (``"compiled"``, or ``"waiting"`` for a region that has not been
    tried yet), ``variants``, ``compiles`` (how many of the variants went to
    the backend: the others had the graph of an earlier one), ``open`` (per
    variant that leaves sizes open, the conditions it holds under),
    ``frozen`` (how many of the variants have the tensors of their modules
    built in, see ``freeze``), ``ops`` and ``seconds`` spent tracing and
    compiling; and one per instruction or
    run of instructions that was left to the VM, with ``state`` ``"vm"`` and
    the ``reason``.
    """
    fn = getattr(fn, "__func__", fn)
    found = _region_passes.get(fn)
    return found.report() if found is not None else []


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


class _CallableModule(types.ModuleType):
    # torch.rewrite is both the entry point and the namespace of the
    # prototype: torch.rewrite(fn) and torch.rewrite.canonical.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return rewrite(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
