import functools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from typing_extensions import Protocol

import torch
from torch import fx


class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ...


CompilerFn = Callable[[fx.GraphModule, List[torch.Tensor]], CompiledFn]

_BACKENDS: Dict[str, CompilerFn] = dict()


def register_backend(
    compiler_fn: Optional[CompilerFn] = None,
    name: Optional[str] = None,
    tags: Sequence[str] = (),
):
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if compiler_fn is None:
        # @register_backend(name="") syntax
        return functools.partial(register_backend, name=name, tags=tags)
    assert callable(compiler_fn)
    name = name or compiler_fn.__name__
    assert name not in _BACKENDS, f"duplicate name: {name}"
    _BACKENDS[name] = compiler_fn
    compiler_fn._tags = tuple(tags)
    return compiler_fn


def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        compiler_fn = _BACKENDS[compiler_fn]
    return compiler_fn


def list_backends():
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    return sorted(_BACKENDS.keys())


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    # TODO(jansel): refactor backends defined in other places
    from .. import debug_utils
    from ..optimizations import backends

    assert backends is not None
    assert debug_utils is not None
