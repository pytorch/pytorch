"""torch.cfg_compile: a prototype compiler that captures control-flow graphs.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.cfg_compile(fn)`` turns the bytecode of ``fn`` into one control-flow
graph in SSA form, branches and loops included, and runs it in a small VM::

    @torch.cfg_compile
    def f(x, y):
        while x.sum() < 10:  # data-dependent control flow stays in the graph
            x = x * 2 + y
        return x


    f(torch.ones(3), torch.ones(3))
    print(f.dump())  # the SSA graph
"""

import functools
import importlib
import sys
import types
import warnings
from typing import Any

import torch


__all__ = ["Unsupported", "cfg_compile", "is_available"]


class Unsupported(RuntimeError):
    """The function uses something the prototype cannot compile."""


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
        _runtime = _cfg_rs
    return _runtime


def is_available() -> bool:
    """Whether this build of PyTorch has the cfg_compile runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def cfg_compile(fn: Any = None, *, fallback: bool = True) -> Any:
    """Compiles ``fn`` to an SSA control-flow graph that runs in the cfg VM.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (whose
            ``forward`` is compiled; calling the result skips module hooks).
            Can be omitted to use ``cfg_compile`` as a decorator with options.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text.
    """
    options: dict[str, Any] = {"fallback": fallback}
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
    runtime = _load_runtime()
    try:
        return runtime.compile(fn)
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
