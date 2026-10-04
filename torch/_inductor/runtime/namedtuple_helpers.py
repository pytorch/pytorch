"""Pickle-safe NamedTuple reconstruction for Inductor-generated modules.

User ``tl.constexpr`` NamedTuples are embedded via ``repr`` into generated
Triton modules (``triton_meta={...!r}``). Those modules live under
``torch._inductor.runtime.compile_tasks.<hash>``, so a locally rebuilt
``collections.namedtuple`` class is not pickleable when a compile worker
returns the kernel (``TORCHINDUCTOR_WORKER_START=fork``).

Types constructed here are registered on this importable module and reduce
via ``namedtuple_type``, so they survive worker pickle and remote-cache
reloads without importing the user's defining module.

Generated modules import this helper by path
(``from torch._inductor.runtime.namedtuple_helpers import namedtuple_type``)
and call ``namedtuple_type(name, fields)``. Renaming the module or changing
that signature is a cache compatibility break for already-generated sources.
"""

from __future__ import annotations

import collections
import hashlib
import sys
from typing import Any


_CACHE: dict[tuple[str, tuple[str, ...]], type] = {}


def _attr_name(name: str, fields: tuple[str, ...]) -> str:
    payload = (
        name.encode("utf-8") + b"\0" + b"\0".join(f.encode("utf-8") for f in fields)
    )
    digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:20]
    return f"_NamedTuple_{digest}"


def _rebuild_namedtuple_instance(
    name: str, fields: tuple[str, ...], values: tuple[Any, ...]
) -> Any:
    return namedtuple_type(name, fields)(*values)


def namedtuple_type(name: str, fields: tuple[str, ...] | list[str]) -> type:
    """Return a cached ``collections.namedtuple`` registered for pickling.

    ``name`` is kept as ``cls.__name__`` so ``repr`` still evals as
    ``Name(field=...)`` in generated modules. Pickle uses a custom
    ``__reduce_ex__`` that reconstructs through this helper, so the class
    need not already exist in the unpickling process.
    """
    fields_t = tuple(fields)
    key = (name, fields_t)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    attr = _attr_name(name, fields_t)
    cls = collections.namedtuple(name, fields_t)  # type: ignore[misc]
    cls.__module__ = __name__
    cls.__qualname__ = attr

    def __reduce_ex__(self, protocol: int) -> tuple[Any, ...]:
        return (_rebuild_namedtuple_instance, (name, fields_t, tuple(self)))

    cls.__reduce_ex__ = __reduce_ex__  # type: ignore[method-assign, assignment]
    setattr(sys.modules[__name__], attr, cls)
    _CACHE[key] = cls
    return cls
