"""
This module implements TorchDynamo's backend registry system for managing compiler backends.

The registry provides a centralized way to register, discover and manage different compiler
backends that can be used with torch.compile(). It handles:

- Backend registration and discovery through decorators and entry points
- Lazy loading of backend implementations
- Lookup and validation of backend names
- Categorization of backends using tags (debug, experimental, etc.)

Key components:
- CompilerFn: Type for backend compiler functions that transform FX graphs
- _BACKENDS: Registry mapping backend names to entry points
- _COMPILER_FNS: Registry mapping backend names to loaded compiler functions

Example usage:
    @register_backend
    def my_compiler(fx_graph, example_inputs):
        # Transform FX graph into optimized implementation
        return compiled_fn

    # Use registered backend
    torch.compile(model, backend="my_compiler")

The registry also supports discovering backends through setuptools entry points
in the "torch_dynamo_backends" group. Example:
```
setup.py
---
from setuptools import setup

setup(
    name='my_torch_backend',
    version='0.1',
    packages=['my_torch_backend'],
    entry_points={
        'torch_dynamo_backends': [
            # name = path to entry point of backend implementation
            'my_compiler = my_torch_backend.compiler:my_compiler_function',
        ],
    },
)
```
```
my_torch_backend/compiler.py
---
def my_compiler_function(fx_graph, example_inputs):
    # Transform FX graph into optimized implementation
    return compiled_fn
```
Using `my_compiler` backend:
```
import torch

model = ...  # Your PyTorch model
optimized_model = torch.compile(model, backend="my_compiler")
```
"""

import functools
import logging
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any, Callable, Optional, Protocol, Union

import torch
from torch import fx


log = logging.getLogger(__name__)


class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


CompilerFn = Callable[[fx.GraphModule, list[torch.Tensor]], CompiledFn]

_BACKENDS: dict[str, Optional[EntryPoint]] = {}
_COMPILER_FNS: dict[str, CompilerFn] = {}


def register_backend(
    compiler_fn: Optional[CompilerFn] = None,
    name: Optional[str] = None,
    tags: Sequence[str] = (),
) -> Callable[..., Any]:
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
        return functools.partial(register_backend, name=name, tags=tags)  # type: ignore[return-value]
    assert callable(compiler_fn)
    name = name or compiler_fn.__name__
    assert name not in _COMPILER_FNS, f"duplicate name: {name}"
    if compiler_fn not in _BACKENDS:
        _BACKENDS[name] = None
    _COMPILER_FNS[name] = compiler_fn
    compiler_fn._tags = tuple(tags)  # type: ignore[attr-defined]
    return compiler_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn: Union[str, CompilerFn]) -> CompilerFn:
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            from ..exc import InvalidBackend

            raise InvalidBackend(name=compiler_fn)

        if compiler_fn not in _COMPILER_FNS:
            entry_point = _BACKENDS[compiler_fn]
            if entry_point is not None:
                register_backend(compiler_fn=entry_point.load(), name=compiler_fn)
        compiler_fn = _COMPILER_FNS[compiler_fn]
    return compiler_fn


# NOTE: can't type this due to public api mismatch; follow up with dev team
def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:  # type: ignore[no-untyped-def]
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags_set = set(exclude_tags or ())

    backends = [
        name
        for name in _BACKENDS.keys()
        if name not in _COMPILER_FNS
        or not exclude_tags_set.intersection(_COMPILER_FNS[name]._tags)  # type: ignore[attr-defined]
    ]
    return sorted(backends)


@functools.cache
def _lazy_import() -> None:
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    assert dynamo_minifier_backend is not None

    _discover_entrypoint_backends()


@functools.cache
def _discover_entrypoint_backends() -> None:
    # importing here so it will pick up the mocked version in test_backends.py
    from importlib.metadata import entry_points

    group_name = "torch_dynamo_backends"
    eps = entry_points(group=group_name)
    eps_dict = {name: eps[name] for name in eps.names}
    for backend_name in eps_dict:
        _BACKENDS[backend_name] = eps_dict[backend_name]
