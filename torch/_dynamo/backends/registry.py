import functools
import sys
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

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


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            _lazy_import_entry_point(compiler_fn)
        if compiler_fn not in _BACKENDS:
            from ..exc import InvalidBackend

            raise InvalidBackend(name=compiler_fn)
        compiler_fn = _BACKENDS[compiler_fn]
    return compiler_fn


def list_backends(exclude_tags=("debug", "experimental")):
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags = set(exclude_tags or ())
    return sorted(
        [
            name
            for name, backend in _BACKENDS.items()
            if not exclude_tags.intersection(backend._tags)
        ]
    )


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    assert dynamo_minifier_backend is not None


@functools.lru_cache(None)
def _lazy_import_entry_point(backend_name: str):
    from importlib.metadata import entry_points

    compiler_fn = None
    group_name = "torch_dynamo_backends"
    if sys.version_info < (3, 10):
        backend_eps = entry_points()
        eps = [ep for ep in backend_eps.get(group_name, ()) if ep.name == backend_name]
        if len(eps) > 0:
            compiler_fn = eps[0].load()
    else:
        backend_eps = entry_points(group=group_name)
        if backend_name in backend_eps.names:
            compiler_fn = backend_eps[backend_name].load()

    if compiler_fn is not None and backend_name not in list_backends(tuple()):
        register_backend(compiler_fn=compiler_fn, name=backend_name)
