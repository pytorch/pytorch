"""
Ahead-of-time precompilation of a callable into MANY graphs.

``torch.compile(fn, fullgraph=True).aot_compile(...)`` produces exactly one
graph and rejects graph breaks. This module is the multi-graph counterpart: it
captures every frame Dynamo produces -- the entry frame, each
``torch_dynamo_resume_in_*`` continuation created by a graph break, and every
recompiled variant of each -- into a single serializable artifact.

Usage::

    session = precompile_capture(model, backend="inductor")
    with session as compiled:
        for variant in variants:  # every path you want covered
            with variant:
                compiled(*args)
    session.save(path)

    # later, in a fresh process
    compiled = precompile_load(model, path, backend="inductor")
    with serving():  # no compilation permitted
        compiled(*args)

Capture is by execution: a resume function only exists once the frame ahead of
it has actually run, so every variant must be exercised. Whatever you do not
run is not in the artifact.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING
from typing_extensions import Self

import torch
import torch._functorch.config as functorch_config

from .exc import PackageError
from .guards import CheckFunctionManager
from .package import CompilePackage, DiskDynamoStore


if TYPE_CHECKING:
    from .package import _DynamoCacheEntry
    from .types import GuardFilterEntry


log = logging.getLogger(__name__)

__all__ = [
    "PrecompileSession",
    "PrecompileSummary",
    "PrecompiledCallable",
    "precompile_capture",
    "precompile_load",
    "serving",
]


def default_guard_filter_fn(
    guard_entries: Sequence[GuardFilterEntry],
) -> Sequence[bool]:
    """
    Keep every guard we can actually serialize.

    Dropping a guard is not free: it does not fail at serving time, it makes a
    graph traced under one condition get reused under another. So drop only the
    types that cannot be written to disk at all, and let serialization raise for
    anything else.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """What a capture actually produced. Assert on this in a build step."""

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.bypassed

    def __str__(self) -> str:
        base = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        return base


def _summarize(entry: _DynamoCacheEntry) -> PrecompileSummary:
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(entry.backend_ids),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
    )


class PrecompileSession:
    """
    A capture in progress. Use as a context manager to get the callable to
    exercise, then ``save()``.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
        | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
    ) -> None:
        self._fn = fn
        self._backend = backend
        self._guard_filter_fn = guard_filter_fn or default_guard_filter_fn
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        self._package = CompilePackage(self._entry_fn)
        self._stack: contextlib.ExitStack | None = None
        self._compiled: Callable[..., object] | None = None

    def __enter__(self) -> Callable[..., object]:
        if self._stack is not None:
            raise RuntimeError("PrecompileSession is already active")
        stack = contextlib.ExitStack()
        # Backends must serialize into the artifact rather than into the
        # process-local inductor cache.
        stack.enter_context(functorch_config.patch("bundled_autograd_cache", True))
        self._stack = stack
        self._compiled = torch._dynamo.optimize(
            self._backend,
            package=self._package,
            guard_filter_fn=self._guard_filter_fn,
            recompile_limit=self._recompile_limit,
            dynamic=self._dynamic,
        )(self._fn)
        return self._compiled

    def __exit__(self, *exc: object) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def summary(self) -> PrecompileSummary:
        return _summarize(self._package.cache_entry())

    def save(self, path: str, *, require_complete: bool = True) -> PrecompileSummary:
        """
        Write the artifact. Raises if any frame was bypassed, which is what
        happens when a frame exceeds ``recompile_limit`` -- that frame is pinned
        to eager and its remaining variants were never captured, so the artifact
        would silently stop matching at serving time.
        """
        if self._stack is not None:
            raise RuntimeError("save() must be called after the capture block exits")
        summary = self.summary()
        if require_complete and not summary.complete:
            raise PackageError(
                f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                f"were bypassed and will fall back to eager: {list(summary.bypassed)}. "
                f"This usually means a frame exceeded recompile_limit "
                f"(currently {self._recompile_limit}); a frame needs one slot per "
                f"variant, and frames shared across module instances accumulate "
                f"them. Raise recompile_limit, or pass require_complete=False to "
                f"accept a partial artifact."
            )
        store = DiskDynamoStore()
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be handed to the store explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                store.record_eager_backend(backend_id, backend)
        store.save_package(self._package, path)
        log.info("precompile: saved %s to %s", summary, path)
        return summary


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
) -> PrecompileSession:
    """
    Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
    )


def precompile_load(
    fn: Callable[..., object],
    path: str,
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
) -> PrecompiledCallable:
    """
    Load an artifact and return a callable ready to serve it.

    The wiring is order-sensitive -- the package has to be attached to the
    optimize context before its globals and guarded codes are installed -- so it
    is done here rather than left to callers.

    Installing mutates global state on the underlying code objects, which
    ``torch._dynamo.reset()`` does not undo, so the result is also a context
    manager that unloads on exit.
    """
    store = DiskDynamoStore()
    package, backends = store.load_package(_entry_fn_of(fn), path)
    compiled = torch._dynamo.optimize(
        backend,
        package=package,
        guard_filter_fn=guard_filter_fn or default_guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
    )(fn)
    package.install(backends)
    return PrecompiledCallable(compiled, package)


class PrecompiledCallable:
    """A loaded artifact. Call it, or use it as a context manager to scope it."""

    def __init__(
        self, compiled: Callable[..., object], package: CompilePackage
    ) -> None:
        self._compiled = compiled
        self._package = package

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._compiled(*args, **kwargs)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove installed globals and precompile entries from the code objects."""
        self._package.uninstall()


@contextlib.contextmanager
def serving() -> Iterator[None]:
    """
    Forbid compilation, so a call the artifact does not cover raises instead of
    quietly recompiling. This is process-wide, not a property of the artifact.
    """
    with torch.compiler.set_stance("fail_on_recompile"):
        yield


@functools.singledispatch
def _entry_fn_of(fn: object) -> Callable[..., object]:
    if not callable(fn):
        raise TypeError(f"expected a callable or nn.Module, got {type(fn).__name__}")
    return fn  # type: ignore[return-value]


@_entry_fn_of.register
def _(fn: torch.nn.Module) -> Callable[..., object]:
    return fn.forward
