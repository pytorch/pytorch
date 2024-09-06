# mypy: ignore-errors

import functools
import logging
import sys
from importlib.metadata import EntryPoint
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
from torch import fx


log = logging.getLogger(__name__)


class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ...


CompilerFn = Callable[[fx.GraphModule, List[torch.Tensor]], CompiledFn]

_BACKENDS: Dict[str, Optional[EntryPoint]] = {}
_COMPILER_FNS: Dict[str, CompilerFn] = {}


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
    assert name not in _COMPILER_FNS, f"duplicate name: {name}"
    if compiler_fn not in _BACKENDS:
        _BACKENDS[name] = None
    _COMPILER_FNS[name] = compiler_fn
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
            from ..exc import InvalidBackend

            raise InvalidBackend(name=compiler_fn)

        if compiler_fn not in _COMPILER_FNS:
            entry_point = _BACKENDS[compiler_fn]
            register_backend(compiler_fn=entry_point.load(), name=compiler_fn)
        compiler_fn = _COMPILER_FNS[compiler_fn]
    return compiler_fn


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags = set(exclude_tags or ())

    backends = [
        name
        for name in _BACKENDS.keys()
        if name not in _COMPILER_FNS
        or not exclude_tags.intersection(_COMPILER_FNS[name]._tags)
    ]
    return sorted(backends)


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    assert dynamo_minifier_backend is not None

    _discover_entrypoint_backends()


@functools.lru_cache(None)
def _discover_entrypoint_backends():
    # importing here so it will pick up the mocked version in test_backends.py
    from importlib.metadata import entry_points

    group_name = "torch_dynamo_backends"
    if sys.version_info < (3, 10):
        eps = entry_points()
        eps = eps[group_name] if group_name in eps else []
        eps = {ep.name: ep for ep in eps}
    else:
        eps = entry_points(group=group_name)
        eps = {name: eps[name] for name in eps.names}
    for backend_name in eps:
        _BACKENDS[backend_name] = eps[backend_name]
