"""torch.cfg_compile: a prototype compiler that captures control-flow graphs.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.cfg_compile(fn)`` turns the bytecode of ``fn`` into one control-flow
graph in SSA form, branches and loops included, and runs it in a small VM.
There is no symbolic tracing and there are no fake tensors: the first run of
each basic block is its trace. It runs on the real tensors, with a hook in the
dispatcher recording which kernel every tensor operation resolved to. The
block is then specialized so that those operations call their post-dispatch
kernel directly, guarded on the dispatch keys of their operands::

    @torch.cfg_compile
    def f(x, y):
        while x.sum() < 10:  # data-dependent control flow stays in the graph
            x = x * 2 + y
        return x


    f(torch.ones(3), torch.ones(3))
    print(f.dump())  # the SSA graph and its specialized blocks

Nothing is specialized on shapes unless a pass asks for it: every dimension is
dynamic. A pass can narrow a dimension to a range by attaching a guard; when
such a guard fails, only the basic block that carries it is recompiled.

Calls are followed. A call of a Python function of the same module is inlined
under a guard on the function; any other Python function, method or
``nn.Module`` is compiled the first time compiled code calls it; and
``for i in range(k)`` with a small constant ``k`` is unrolled. Closures,
``try``/``except``/``finally``, ``with``, imports and class statements are
compiled. Generators, coroutines and functions that read their own frame
(``locals()``, ``eval``, ``exec`` and the like) run in the interpreter, and so
does what they call. A Python side effect does not split the function: it is
one more instruction of the graph, between the specialized ones. A whole
training step can therefore be one compiled function, with the same results
bit for bit as the interpreter::

    @torch.cfg_compile
    def step(model, opt, x, y):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()  # the autograd engine, on the tape the forward recorded
        opt.step()
        return loss


    step(model, opt, x, y)
    step.stats()["compiled_callees"]  # Module._call_impl, Linear.forward, SGD.step, ...

What is compiled is described by a :class:`Pipeline`, an ordered list of
passes that is run over each block until it stops changing. Passes are
built-in ones, named by a string, or Python callables taking a ``Region``.
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


__all__ = ["Pipeline", "Unsupported", "cfg_compile", "is_available", "regions"]

Pass = str | Callable[[Any], bool]


class Unsupported(RuntimeError):
    """The function uses something the prototype cannot compile."""


class Pipeline:
    """An ordered list of CFG passes.

    The passes run in order over a region of the graph, and the whole list is
    repeated until no pass changes the region anymore. A pass is the name of a
    built-in pass or a callable ``pass_(region) -> bool`` that returns whether
    it changed the region.

    Built-in passes:

    - ``resolve_dispatch``: operations the profile found to be a single
      dispatcher call become direct calls of the post-dispatch kernel.
    - ``method_calls``: ``x.f(...)`` calls that stay generic are made without
      creating a bound method object.
    - ``tensor_accessors``: ``x.shape[i]``, ``x.size(i)``, ``x.dim()`` and
      ``x.ndim`` on tensors become plain SSA integers.
    - ``narrow_dims``: a comparison of a dimension with a constant is folded to
      the outcome last observed, under a guard that pins the dimension to the
      range of sizes with that outcome. Not part of the default pipeline.
    - ``fold``: folds truthiness tests and branches on constants.
    - ``dce``: drops instructions whose result is not needed.

    Pipelines are immutable; the methods return new ones.
    """

    def __init__(self, passes: Iterable[Pass]) -> None:
        self.passes: tuple[Pass, ...] = tuple(passes)
        for p in self.passes:
            if not isinstance(p, str) and not callable(p):
                raise TypeError(f"a pass is a name or a callable, not {type(p)}")

    @classmethod
    def default(cls) -> "Pipeline":
        """Direct kernel calls, with every dimension left dynamic."""
        return cls(
            ["resolve_dispatch", "tensor_accessors", "method_calls", "fold", "dce"]
        )

    @classmethod
    def narrowing(cls) -> "Pipeline":
        """The default pipeline plus ``narrow_dims``."""
        return cls.default().insert_before("fold", "narrow_dims")

    def append(self, pass_: Pass) -> "Pipeline":
        """A pipeline with ``pass_`` added at the end."""
        return Pipeline([*self.passes, pass_])

    def insert_before(self, name: str, pass_: Pass) -> "Pipeline":
        """A pipeline with ``pass_`` added in front of the pass called ``name``."""
        i = self.passes.index(name)
        return Pipeline([*self.passes[:i], pass_, *self.passes[i:]])

    def insert_after(self, name: str, pass_: Pass) -> "Pipeline":
        """A pipeline with ``pass_`` added behind the pass called ``name``."""
        i = self.passes.index(name) + 1
        return Pipeline([*self.passes[:i], pass_, *self.passes[i:]])

    def remove(self, name: str) -> "Pipeline":
        """A pipeline without the pass called ``name``."""
        return Pipeline([p for p in self.passes if p != name])

    def __repr__(self) -> str:
        names = [
            p if isinstance(p, str) else getattr(p, "__name__", repr(p))
            for p in self.passes
        ]
        return f"Pipeline({names})"


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
                "torch.cfg_compile needs its Rust runtime (torch._cfg_rs), which this build "
                "of PyTorch does not include. Build PyTorch with USE_CFG_COMPILE=1 "
                "(requires cargo)."
            ) from e
        _cfg_rs.init(torch._C._cfg_c_api())
        _runtime = _cfg_rs
    return _runtime


def is_available() -> bool:
    """Whether this build of PyTorch has the cfg_compile runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def cfg_compile(
    fn: Any = None,
    *,
    pipeline: Pipeline | None = None,
    fallback: bool = True,
    specialize: bool = True,
    skip_autograd: bool = True,
    unboxed: bool = True,
    inline: bool = True,
    unroll: bool = True,
    callees: bool = True,
    shortcuts: bool = True,
    demote: bool | None = None,
    backend: str | Callable[..., Any] | None = None,
    region_min_ops: int = 4,
    region_max_variants: int = 8,
    max_versions: int = 4,
    max_respecializations: int = 6,
) -> Any:
    """Compiles ``fn`` to an SSA control-flow graph that runs in the cfg VM.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (whose
            ``forward`` is compiled; calling the result skips module hooks).
            Can be omitted to use ``cfg_compile`` as a decorator with options.
        pipeline: the passes that specialize the graph. Defaults to
            :meth:`Pipeline.default`.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.
        specialize: if ``False``, only run the generic tier of the VM: no
            profiling, no direct kernel calls.
        skip_autograd: let a call site enter below the Autograd kernel when
            the call it was captured from tracked no gradient. Such a site is
            guarded on that still being the case.
        unboxed: keep the tensors that kernels return without a Python object
            until Python code needs one. A tensor that only flows from one
            kernel call into others then never gets one.
        inline: replace calls of Python functions defined in the same module
            by their graph, under a guard on the callee; the call itself stays
            as the other side of that guard.
        unroll: replace ``for i in range(k)`` loops with a small constant ``k``
            by ``k`` copies of their body, under a guard on ``range``.
        callees: compile the Python functions that the compiled code calls
            (functions, methods, and objects whose ``__call__`` is a Python
            function) the first time each is called, with the same options.
        shortcuts: skip the code of ``nn.Module`` that sits in front of every
            call of a module and every read of a parameter, buffer or
            submodule, whenever a few checks show what it would do: a call of
            a module without hooks goes straight to its ``forward``, and the
            three dicts are looked at without calling ``Module.__getattr__``.
            ``Module.__call__`` then does not appear on the stack.
        demote: measure the function and each of its callees, once warmed up,
            both compiled and in the interpreter, and leave to the interpreter
            what runs faster there: the VM gains on tensor operations and loses
            on plain Python, and library code is often mostly the latter.
            ``stats()["demoted_callees"]`` lists the callees that were left,
            and ``stats()["demoted"]`` says whether the function itself was.
            On by default; the environment variable
            ``TORCH_CFG_COMPILE_DEMOTE=0`` turns the default off, for looking
            at the VM itself.
            A callee the compiler does not support runs in the interpreter.
        backend: hand the tensor work of each basic block to a graph compiler.
            ``"inductor"`` traces every run of instructions of a block that a
            trace can capture, together with what those instructions call,
            into one ATen graph, and compiles it with AOTAutograd and
            Inductor: with gradients enabled, the tensors such a region
            returns have one compiled autograd function as their ``grad_fn``.
            The outermost region that traces is the one that is compiled;
            where an instruction does not trace, the VM runs it and the blocks
            of what it calls are tried in turn. ``"aot_eager"`` builds the
            same graphs and runs them with the eager kernels. A callable
            ``backend(graph_module, example_inputs)`` returning a callable is
            used as given. ``regions()`` on the result reports what was
            compiled and what was left to the VM, with the reason. See
            ``torch/cfg_compile/_regions.py`` for what a compiled region
            checks on every call and what it does not.
        region_min_ops: a region with fewer tensor operations than this is
            left to the VM, whose direct kernel calls are cheaper than
            entering a compiled graph.
        region_max_variants: how many combinations of inputs (shapes, dtypes,
            modules, modes) a region is compiled for before further ones run
            its instructions as they are.
        max_versions: how many specialized versions a block can have at once.
        max_respecializations: how many times a block may be profiled and
            specialized again before it settles on code without guards.

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text, ``stats()`` and ``sites()``.
    """
    if demote is None:
        demote = os.environ.get("TORCH_CFG_COMPILE_DEMOTE", "1") != "0"
    options: dict[str, Any] = {
        "pipeline": pipeline,
        "fallback": fallback,
        "specialize": specialize,
        "skip_autograd": skip_autograd,
        "unboxed": unboxed,
        "inline": inline,
        "unroll": unroll,
        "callees": callees,
        "shortcuts": shortcuts,
        "demote": demote,
        "backend": backend,
        "region_min_ops": region_min_ops,
        "region_max_variants": region_max_variants,
        "max_versions": max_versions,
        "max_respecializations": max_respecializations,
    }
    if fn is None:
        return functools.partial(cfg_compile, **options)
    if isinstance(fn, torch.nn.Module):
        return types.MethodType(cfg_compile(type(fn).forward, **options), fn)
    if isinstance(fn, types.MethodType):
        return types.MethodType(cfg_compile(fn.__func__, **options), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(
            f"torch.cfg_compile expects a function, not {type(fn).__name__}"
        )
    if pipeline is None:
        pipeline = Pipeline.default()
    runtime = _load_runtime()
    regions = None
    if backend is not None:
        if not specialize:
            raise ValueError("backend= needs specialize=True: regions are a pass")
        regions = _regions.make_pass(
            runtime, backend, region_min_ops, region_max_variants, warmup=0
        )
        # First, while the instructions are still the ones of the block.
        pipeline = Pipeline([regions, *pipeline.passes])
    try:
        compiled = runtime.compile(
            fn,
            list(pipeline.passes),
            specialize=specialize,
            skip_autograd=skip_autograd,
            unboxed=unboxed,
            inline=inline,
            unroll=unroll,
            callees=callees,
            shortcuts=shortcuts and _shortcuts.register(runtime),
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
        warnings.warn(
            f"torch.cfg_compile left {fn.__qualname__} uncompiled: {e}", stacklevel=2
        )
        return fn


# The region pass of each compiled function that has a backend.
_region_passes: "weakref.WeakKeyDictionary[Any, _regions.Regions]" = (
    weakref.WeakKeyDictionary()
)


def regions(fn: Any) -> list[dict[str, Any]]:
    """What ``backend=`` made of a compiled function and of what it calls.

    One dict per region, with ``where`` (function, block and instructions),
    ``state`` (``"compiled"``, or ``"waiting"`` for a region that has not been
    tried yet), ``variants``, ``ops`` and ``seconds`` spent tracing and
    compiling; and one per instruction or run of instructions that was left
    to the VM, with ``state`` ``"vm"`` and the ``reason``.
    """
    fn = getattr(fn, "__func__", fn)
    found = _region_passes.get(fn)
    return found.report() if found is not None else []


class _CallableModule(types.ModuleType):
    # torch.cfg_compile is both the entry point and the namespace of the
    # prototype: torch.cfg_compile(fn) and torch.cfg_compile.Pipeline.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return cfg_compile(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
