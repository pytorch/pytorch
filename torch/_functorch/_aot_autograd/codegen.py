"""Generic codegen surface shared by AOTAutograd's runtime wrappers.

``_compile_and_exec_source`` -- the chokepoint that compiles a generated wrapper
source string into a live function -- lives in codegen_utils.py alongside
PySourceBuilder; it is re-exported here so importers have a stably-named
``codegen`` module for the generic exec primitive without reaching into the
PySourceBuilder module. This module also hosts the optional thread-local capture
sink that records each codegen'd wrapper's source, which AOT-to-Python lowering
(to_standalone_python.py) uses to compose the wrappers into one standalone module.
codegen_utils._compile_and_exec_source appends to the sink (lazy import) when one
is installed. Kept as a leaf module (stdlib + torch only) so it is safe to import
anywhere.
"""

import contextlib
import threading
from collections.abc import Iterator

from .codegen_utils import _compile_and_exec_source


__all__ = [
    "_compile_and_exec_source",
    "GeneratedSource",
    "capture_generated_sources",
]


# Optional sink for the source of every runtime-wrapper function codegen'd via
# ``_compile_and_exec_source``. When a sink is installed (see
# ``capture_generated_sources``), each codegen'd wrapper appends a
# ``GeneratedSource`` to it; this lets AOT-to-Python lowering compose the wrappers
# into one standalone module. Thread-local (NOT a process global) so a concurrent
# compile on another thread cannot splice its wrappers into this capture -- and,
# conversely, so an unrelated concurrent compile on another thread is never mistaken for
# offloaded capture work and aborted (``_compile_and_exec_source`` runs on EVERY
# AOTAutograd compile, not just this one). Absent (zero overhead, no behavior change)
# during ordinary compilation. Mirrors the threading.local used by
# _saved_tensor_hook_context in graph_compile.py.
_capture_tls = threading.local()


def _current_capture_sink() -> "list[GeneratedSource] | None":
    return getattr(_capture_tls, "sink", None)


class GeneratedSource:
    """One codegen'd runtime-wrapper function: its source, the exec'd function
    object, and the globals it closes over (which include the inner ``compiled_fn``
    it chains to, plus any baked metadata). The function object lets a composer wire
    cross-wrapper references by identity. Recorded by ``_compile_and_exec_source``
    when capture is active.

    ``globals_dict`` is a PRE-EXEC snapshot of the declared closure globals: it does
    NOT contain the interpreter ``__builtins__`` that exec() would otherwise inject,
    so the standalone composer can reconstruct every entry as source without having
    to special-case ``__builtins__``."""

    def __init__(
        self,
        artifact_name: str,
        fn_name: str,
        source: str,
        globals_dict: dict[str, object],
        fn: object,
        origin_id: int | None = None,
    ) -> None:
        self.artifact_name = artifact_name
        self.fn_name = fn_name
        self.source = source
        self.globals_dict = globals_dict
        self.fn = fn
        # Identity of the TracingContext this wrapper was codegen'd under. The capture
        # sink is duration-scoped over one inductor compile, so a re-entrant on-thread
        # lowering that codegen's into THIS sink during that window would append its
        # wrappers too; the composer filters by this id, which separates such a foreign
        # lowering when it ran under a DISTINCT TracingContext. A same-context re-entrant
        # lowering shares this id and is caught instead by the composer's
        # orchestration-count guard.
        self.origin_id = origin_id


@contextlib.contextmanager
def capture_generated_sources(into: "list[GeneratedSource]") -> "Iterator[None]":
    """Within this context, record every codegen'd runtime-wrapper function's source
    into ``into`` (in codegen order). A no-op when not entered.

    THREADING: the sink is thread-local, so every wrapper captured into ``into`` must be
    codegen'd ON THIS THREAD. There is deliberately NO cross-thread tripwire here: a
    process-global owner check would abort an unrelated concurrent compile on another
    thread, since ``_compile_and_exec_source`` runs on every AOTAutograd compile and is not
    serialized against this capture. If a future change offloads wrapper codegen to a
    worker thread, those wrappers are simply not captured; a forward that loses its
    orchestration wrapper this way is rejected by the composer (it requires exactly one),
    so the common case fails loudly rather than emitting an empty module.
    """
    prev = getattr(_capture_tls, "sink", None)
    _capture_tls.sink = into
    try:
        yield
    finally:
        _capture_tls.sink = prev
