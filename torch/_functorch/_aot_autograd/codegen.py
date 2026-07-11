"""
Generic codegen for AOTAutograd runtime wrappers.

PySourceBuilder accumulates indented Python source together with the globals the
generated code closes over and a fresh-name counter; _compile_and_exec_source is
the single chokepoint that turns generated source into a live function (logging it
as a structured-trace artifact for tlparse and giving it a stable traceback
filename). Every runtime-wrapper codegen routes through it -- the subclass wrappers
in subclass_codegen.py and the orchestration / alias / mutation epilogues in
runtime_wrappers.py -- so it knows nothing about any particular wrapper kind. This
module also hosts the optional thread-local capture sink that records each codegen'd
wrapper's source, which AOT-to-Python lowering (to_standalone_python.py) uses to
compose the wrappers into one standalone module. Kept as a leaf module (stdlib +
torch only) so it is safe to import anywhere.
"""

import contextlib
import functools
import logging
import threading
from collections.abc import Callable, Iterator

import torch


log = logging.getLogger(__name__)


class PySourceBuilder:
    """Builds indented Python source for compile/exec, along with the globals
    the generated code closes over and a monotonic fresh-name counter.

    Body lines are written WITHOUT leading whitespace; indentation is managed
    by the ``indent()`` context manager so call sites read as plain code. Pass
    ``fn_name``/``artifact_name`` to emit a ``def`` header and enable
    ``build()``, which routes through _compile_and_exec_source.
    """

    def __init__(
        self,
        fn_name: str | None = None,
        *,
        args: str = "args",
        artifact_name: str | None = None,
    ) -> None:
        self.lines: list[str] = []
        self.globals: dict[str, object] = {}
        self._name_counter: int = 0
        self._indent: int = 0
        self._fn_name = fn_name
        self._artifact_name = artifact_name
        if fn_name is not None:
            self.writeline(f"def {fn_name}({args}):")

    @contextlib.contextmanager
    def indent(self, offset: int = 1) -> Iterator[None]:
        self._indent += offset
        try:
            yield
        finally:
            self._indent -= offset

    def writeline(self, line: str) -> None:
        """Append one line at the current indent level (paired with indent())."""
        self.lines.append("    " * self._indent + line)

    def emit(self, line: str, indent: int = 1) -> None:
        """Append one line at an explicit absolute indent level.

        Convenient for recursive generators that thread an indent depth instead
        of nesting indent() context managers.
        """
        self.lines.append("    " * indent + line)

    def fresh_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._name_counter}"
        self._name_counter += 1
        return name

    def add_global(self, name: str, value: object) -> str:
        """Bind a live object into the exec globals under ``name``, by reference."""
        self.globals[name] = value
        return name

    def bind(self, **values: object) -> None:
        """Bind live objects into the exec globals by keyword, by reference."""
        self.globals.update(values)

    def bind_value(self, prefix: str, value: object) -> str:
        """Bind a value under a fresh unique name and return that name."""
        return self.add_global(self.fresh_name(prefix), value)

    def getvalue(self) -> str:
        return "\n".join(self.lines)

    def build(
        self, *, wrapped_fn: Callable[..., object] | None = None
    ) -> Callable[..., object]:
        assert self._fn_name is not None and self._artifact_name is not None, (  # noqa: S101
            "build() requires fn_name and artifact_name"
        )
        return _compile_and_exec_source(
            self.getvalue(),
            self.globals,
            self._fn_name,
            self._artifact_name,
            wrapped_fn=wrapped_fn,
        )


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
    prev = _current_capture_sink()
    _capture_tls.sink = into
    try:
        yield
    finally:
        _capture_tls.sink = prev


def _compile_and_exec_source(
    source: str,
    globals_dict: dict[str, object],
    fn_name: str,
    artifact_name: str,
    wrapped_fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """Compile generated source, exec it, and return the named function.

    If wrapped_fn is provided, applies functools.update_wrapper so that
    __wrapped__ and __dict__ (e.g. _fx_graph_cache_key) propagate to the
    generated function.
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Generated %s:\n%s", artifact_name, source)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: source,
    )

    # Use a path under torch/_functorch/ so the code object is recognized by
    # dynamo's MOD_SKIPLIST. The eval frame hook stays active during the entire
    # torch.compile(fn)(*args) call (to handle graph breaks and resume functions),
    # so codegen'd functions called during backward get intercepted even though
    # no tracing is active. A real path makes them skip automatically.
    code = compile(source, f"{__file__}:codegen({artifact_name})", "exec")
    local_dict: dict[str, object] = {}
    # exec() mutates ``globals_dict`` in place, injecting ``__builtins__`` (and any
    # name the source binds at module scope). When capturing, snapshot the declared
    # closed-over names BEFORE that happens so the captured GeneratedSource holds only
    # the intended closure globals -- not the post-exec dict with the interpreter's
    # ``__builtins__`` -- which is what the standalone composer reconstructs as source.
    sink = _current_capture_sink()
    captured_globals = dict(globals_dict) if sink is not None else globals_dict
    exec(code, globals_dict, local_dict)
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]

    if sink is not None:
        # Tag with the current TracingContext identity so the composer can drop any
        # wrapper a re-entrant lowering appended during the capture window (see the
        # origin_id note on GeneratedSource).
        ctx = torch._guards.TracingContext.try_get()
        origin_id = id(ctx) if ctx is not None else None
        sink.append(
            GeneratedSource(
                artifact_name, fn_name, source, captured_globals, fn, origin_id
            )
        )

    return fn  # type: ignore[return-value]
