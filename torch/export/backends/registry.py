from __future__ import annotations

import functools
import importlib
import os
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = []  # type: ignore[var-annotated]

ExportFn = Callable[
    [Callable, List[Any], Optional[Dict[str, Any]], Optional[Any]], "ExportedProgram"  # type: ignore[name-defined]
]
_BACKENDS: Dict[str, ExportFn] = dict()


class InvalidTorchExportBackend(RuntimeError):
    def __init__(self, name):
        super().__init__(
            f"Invalid backend: {name!r}, see `torch._export.list_backends()` for available backends."
        )


def _lookup_backend(export_fn):
    """Expand backend strings to functions"""
    if isinstance(export_fn, str):
        if export_fn not in _BACKENDS:
            _lazy_import()
        if export_fn not in _BACKENDS:
            _lazy_import_entry_point(export_fn)
        if export_fn not in _BACKENDS:
            raise InvalidTorchExportBackend(name=export_fn)
        export_fn = _BACKENDS[export_fn]
    return export_fn


def register_backend(
    export_fn: Optional[ExportFn] = None,
    name: Optional[str] = None,
    tags: Sequence[str] = (),
):
    """
    Decorator to add a given export function to the registry to allow calling
    `torch.export` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        export_fn: Callable taking a :class:`torch.nn.Module` model, inputs and options
        name: Optional name, defaults to `export_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if export_fn is None:
        # @register_backend(name="") syntax
        return functools.partial(register_backend, name=name, tags=tags)
    assert callable(export_fn)
    name = name or export_fn.__name__
    assert name not in _BACKENDS, f"duplicate backend name: {name}"
    _BACKENDS[name] = export_fn
    export_fn._tags = tuple(tags)  # type: ignore[attr-defined]
    return export_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.export(..., backend="name")`.
    """
    _lazy_import()
    exclude_tags = set(exclude_tags or ())

    return sorted(
        [
            name
            for name, backend in _BACKENDS.items()
            if not exclude_tags.intersection(backend._tags)  # type: ignore[attr-defined]
        ]
    )


@functools.lru_cache(None)
def _lazy_import_entry_point(backend_name: str):
    # TODO: Document similar to <torch>/docs/source/torch.compiler_custom_backends.rst
    from importlib.metadata import entry_points

    export_fn = None
    group_name = "torch_export_backends"
    if sys.version_info < (3, 10):
        backend_eps = entry_points()
        eps = [ep for ep in backend_eps.get(group_name, ()) if ep.name == backend_name]
        if len(eps) > 0:
            export_fn = eps[0].load()
    else:
        backend_eps = entry_points(group=group_name)
        if backend_name in backend_eps.names:
            export_fn = backend_eps[backend_name].load()

    if export_fn is not None and backend_name not in list_backends():
        register_backend(export_fn=export_fn, name=backend_name)


def _import_submodule(mod: types.ModuleType):
    """
    Ensure all the files in a given submodule are imported
    """
    for filename in sorted(os.listdir(os.path.dirname(mod.__file__))):  # type: ignore[type-var]
        if filename.endswith(".py") and filename[0] != "_":
            importlib.import_module(f"{mod.__name__}.{filename[:-3]}")


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends

    _import_submodule(backends)
