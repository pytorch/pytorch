"""torch.vm: a function run in a VM as a control-flow graph, with an ordered list of passes.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.vm(fn)`` turns the bytecode of ``fn`` into one control-flow graph in
SSA form, branches and loops included, and runs it in a small VM::

    @torch.vm
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


__all__ = ["Unsupported", "is_available", "vm"]


class Unsupported(RuntimeError):
    """The function uses something the prototype cannot rewrite."""


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
        _runtime = _cfg_rs
    return _runtime


def is_available() -> bool:
    """Whether this build of PyTorch has the torch.vm runtime."""
    try:
        _load_runtime()
    except RuntimeError:
        return False
    return True


def vm(
    fn: Any = None,
    *,
    fallback: bool = True,
) -> Any:
    """Runs ``fn`` as an SSA control-flow graph in the VM.

    ``torch.vm`` itself is this function: ``@torch.vm``, ``@torch.vm(...)`` and
    ``torch.vm(fn, ...)`` all work.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (its
            ``forward`` is taken; calling the result skips module hooks). Can
            be omitted to use ``torch.vm`` as a decorator with options.
        fallback: if the function uses something that is not supported, warn
            and return it unchanged instead of raising :class:`Unsupported`.

    Returns:
        A callable with the signature of ``fn``. It has ``dump()``, returning
        the graph as text.
    """
    options: dict[str, Any] = {
        "fallback": fallback,
    }
    if fn is None:
        return functools.partial(vm, **options)
    if isinstance(fn, torch.nn.Module):
        return types.MethodType(vm(type(fn).forward, **options), fn)
    if isinstance(fn, types.MethodType):
        return types.MethodType(vm(fn.__func__, **options), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"torch.vm expects a function, not {type(fn).__name__}")
    runtime = _load_runtime()
    try:
        return runtime.compile(fn)
    except NotImplementedError as e:
        if not fallback:
            raise Unsupported(f"{fn.__qualname__}: {e}") from None
        warnings.warn(f"torch.vm left {fn.__qualname__} as it is: {e}", stacklevel=2)
        return fn


class _CallableModule(types.ModuleType):
    # torch.vm is both the entry point and the namespace of the prototype:
    # torch.vm(fn) and torch.vm.is_available().
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return vm(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
