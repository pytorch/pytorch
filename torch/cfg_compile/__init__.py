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

What is compiled is described by a :class:`Pipeline`, an ordered list of
passes that is run over each block until it stops changing. Passes are
built-in ones, named by a string, or Python callables taking a ``Region``.
"""

import functools
import importlib
import sys
import types
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch


__all__ = ["Pipeline", "Unsupported", "cfg_compile", "is_available"]

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
        return Pipeline([*self.passes, pass_])

    def insert_before(self, name: str, pass_: Pass) -> "Pipeline":
        i = self.passes.index(name)
        return Pipeline([*self.passes[:i], pass_, *self.passes[i:]])

    def insert_after(self, name: str, pass_: Pass) -> "Pipeline":
        i = self.passes.index(name) + 1
        return Pipeline([*self.passes[:i], pass_, *self.passes[i:]])

    def remove(self, name: str) -> "Pipeline":
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
        max_versions: how many specialized versions a block can have at once.
        max_respecializations: how many times a block may be profiled and
            specialized again before it settles on code without guards.

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text, ``stats()`` and ``sites()``.
    """
    options: dict[str, Any] = {
        "pipeline": pipeline,
        "fallback": fallback,
        "specialize": specialize,
        "skip_autograd": skip_autograd,
        "unboxed": unboxed,
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
    try:
        return runtime.compile(
            fn,
            list(pipeline.passes),
            specialize=specialize,
            skip_autograd=skip_autograd,
            unboxed=unboxed,
            max_versions=max_versions,
            max_respecializations=max_respecializations,
        )
    except NotImplementedError as e:
        if not fallback:
            raise Unsupported(f"{fn.__qualname__}: {e}") from None
        warnings.warn(
            f"torch.cfg_compile left {fn.__qualname__} uncompiled: {e}", stacklevel=2
        )
        return fn


class _CallableModule(types.ModuleType):
    # torch.cfg_compile is both the entry point and the namespace of the
    # prototype: torch.cfg_compile(fn) and torch.cfg_compile.Pipeline.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return cfg_compile(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
