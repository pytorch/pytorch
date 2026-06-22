# Copyright (c) 2025-2026, Tri Dao.
"""Persistent kernel-cache + compile-only utilities for QuACK.

Public API
----------

Persistent ``.o`` cache:
* :func:`jit_cache` \u2014 decorator that wraps a kernel-compile function with
  in-memory + persistent ``.o`` caching (see :mod:`quack.cache.jit`).
* :data:`CACHE_ENABLED`, :data:`CACHE_DIR`, :data:`EXTRA_SOURCE_DIRS` \u2014
  static-config flags (see "Flag semantics" below).
* :class:`FileLock`, :func:`get_cache_path`, :class:`CacheInfo` \u2014
  supporting types.

Compile-only cache warming:
* :class:`CompileOnlyFakeTensorMode`, :func:`compile_only_mode`,
  :func:`is_compile_only` \u2014 helpers for populating the cache via
  :class:`~torch._subclasses.fake_tensor.FakeTensorMode` (no GPU memory)
  before the real run. See :mod:`quack.cache.compile_only`.

Flag semantics
--------------

The **compile-only flag is a stack**, backed by a :class:`contextvars.ContextVar`
counter. There is no module-level boolean to assign to. The only sanctioned
mutation path is :func:`compile_only_mode` (push on enter, pop on exit, leak-free
by construction). Reads go through :func:`is_compile_only`, which always
reflects the live stack depth.

For backward compatibility, *attribute reads* of ``quack.cache.COMPILE_ONLY``
still work via the module ``__getattr__`` hook and return a live bool on
each access. This covers patterns like::

    import quack.cache
    if quack.cache.COMPILE_ONLY: ...        # live read on every access

``from quack.cache import COMPILE_ONLY`` does **NOT** give a live read \u2014
PEP 562's module ``__getattr__`` fires on attribute *access*, not on import
binding. ``from ... import COMPILE_ONLY`` snapshots the value once at import
time; subsequent reads of the local name don't re-enter ``__getattr__``.
New code should call :func:`is_compile_only` instead.

Direct *writes* (``quack.cache.COMPILE_ONLY = True``) are forbidden \u2014 a
custom module ``__setattr__`` raises :class:`AttributeError` with a pointer
to :func:`compile_only_mode`.

Why the change: the old design used a plain module attribute. Two failure
modes followed:

* Capture-at-import: ``pytestmark = pytest.mark.skipif(quack.cache.COMPILE_ONLY, ...)``
  reads at decorator-application time. Under xdist worksteal the worker can
  import the module before the plugin's ``pytest_configure`` flips the flag,
  so the skip captures ``False`` and never fires. (The :func:`is_compile_only`
  function call avoids this for code under our control; test-file ``pytestmark``
  patterns still need the marker introduced in R2.)
* Leak-on-reset: any caller that did ``COMPILE_ONLY = True; ... finally:
  COMPILE_ONLY = False`` clobbered the *outer* session value to ``False`` on
  exit, breaking the plugin's invariant for the rest of the xdist worker.
  Token-based push/pop on a ``ContextVar`` removes that footgun by design.

Recommended access patterns:

  >>> from quack.cache import is_compile_only, compile_only_mode
  >>> if is_compile_only(): ...               # live read
  >>> with compile_only_mode():                # push, auto-pop on exit
  ...     run_kernels()
"""

from __future__ import annotations

import contextvars
import os
import sys
import types
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Runtime flags. Source of truth.
#
# CRITICAL ORDERING: ``CACHE_ENABLED``/``CACHE_DIR``/``EXTRA_SOURCE_DIRS`` and
# the compile-only ContextVar MUST be defined before the ``from
# quack.cache.jit import ...`` block below. ``quack/cache/jit.py`` does
# ``import quack.cache as _state`` at its module top; Python returns the
# partially-initialized package object, and lookups inside ``jit_cache``'s
# wrapper rely on these names already existing at that checkpoint.
# Reordering the imports here, even via an auto-formatter, will break the
# first kernel compile with ``AttributeError``.
#
# The defensive unit tests in ``tests/test_cache.py`` exercise this path
# end-to-end so a reordering bug surfaces immediately.
# ---------------------------------------------------------------------------

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: Optional[str] = os.getenv("QUACK_CACHE_DIR", None)

# Stack depth for compile-only mode. ``compile_only_mode()`` pushes
# (depth += 1) on enter and pops (reset to prior token) on exit, so:
#   * Nested ``with compile_only_mode():`` blocks stay True throughout.
#   * An exception inside the block still pops correctly (context-manager
#     finally + ContextVar token semantics).
#   * No caller can leak ``False`` back to the outer session by accident,
#     because there is no boolean to assign to \u2014 only a token to reset.
# Read via :func:`quack.cache.compile_only.is_compile_only`.
_COMPILE_ONLY_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "quack_compile_only_depth", default=0
)

#: Downstream projects can append directories here to include their sources
#: in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: List[Path] = []


# ---------------------------------------------------------------------------
# Module attribute hooks: live read for legacy ``quack.cache.COMPILE_ONLY``,
# hard-stop on write to surface migration callers.
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Resolve legacy ``COMPILE_ONLY`` attribute reads to a live boolean.

    Triggers on attribute access \u2014 ``quack.cache.COMPILE_ONLY``, the
    ``hasattr(quack.cache, "COMPILE_ONLY")`` probe, etc. Returns the current
    stack depth as a bool. There is no module-level attribute to shadow it
    (so the hook always wins), and *writes* are intercepted by the custom
    module ``__setattr__`` installed below.

    Important caveat: ``from quack.cache import COMPILE_ONLY`` performs the
    attribute lookup *once* at import time and binds the *value* in the
    importer's namespace; subsequent reads of the local name do NOT re-enter
    this hook. New code should call :func:`is_compile_only` instead of
    relying on the legacy ``from`` import for live semantics.
    """
    if name == "COMPILE_ONLY":
        return _COMPILE_ONLY_DEPTH.get() > 0
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _CacheModule(types.ModuleType):
    """Custom module class that forbids assignment to ``COMPILE_ONLY``.

    Without this guard, ``quack.cache.COMPILE_ONLY = True`` would place a
    real attribute in ``__dict__`` and shadow the ``__getattr__`` proxy
    forever (Python looks in ``__dict__`` before falling through). That's
    the exact leak mode the ContextVar redesign exists to prevent, so we
    raise instead of silently breaking the invariant.
    """

    def __setattr__(self, name: str, value) -> None:
        if name == "COMPILE_ONLY":
            raise AttributeError(
                "quack.cache.COMPILE_ONLY is now a read-only stack-backed flag. "
                "Use `with quack.cache.compile_only_mode(): ...` to enable it "
                "for a scope (auto-restores on exit, exception-safe, leak-free). "
                "See quack/cache/__init__.py docstring for migration notes."
            )
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CacheModule


class CompileOnlyStrictError(RuntimeError):
    """Raised by precompile helpers under ``QUACK_COMPILE_ONLY_STRICT=1``.

    Wraps the underlying exception so the reusable pytest plugin's blanket
    swallow hooks (which exist to ignore expected FakeTensor-incompatibility
    errors *after* the kernel has dispatched) can let strict-mode failures
    surface as real test failures.

    Without this distinction, ``QUACK_COMPILE_ONLY_STRICT=1`` looks like it
    works (the inner ``try/except`` does re-raise) but the surrounding
    pytest plugin would force-pass the test anyway, defeating the strict
    mode entirely.
    """


# ---------------------------------------------------------------------------
# Public API surface. Imported AFTER the flags are defined.
# ---------------------------------------------------------------------------

from .jit import (  # noqa: E402
    EXPORT_FUNC_NAME,
    LOCK_TIMEOUT,
    CacheInfo,
    FileLock,
    get_cache_path,
    jit_cache,
)
from .compile_only import (  # noqa: E402
    CompileOnlyFakeTensorMode,
    compile_only_mode,
    is_compile_only,
)

# ``__all__`` advertises the *recommended* public API only. Static-config
# flags (``CACHE_ENABLED``, ``CACHE_DIR``, ``EXTRA_SOURCE_DIRS``) and the
# legacy ``COMPILE_ONLY`` live-read proxy are intentionally *not* listed: the
# recommended patterns for compile-only mode are :func:`is_compile_only`
# (read) and :func:`compile_only_mode` (push/pop, context-managed).
__all__ = [
    # Persistent .o cache.
    "jit_cache",
    "CacheInfo",
    "EXPORT_FUNC_NAME",
    "LOCK_TIMEOUT",
    "FileLock",
    "get_cache_path",
    # Compile-only cache warming.
    "CompileOnlyFakeTensorMode",
    "CompileOnlyStrictError",
    "compile_only_mode",
    "is_compile_only",
]
