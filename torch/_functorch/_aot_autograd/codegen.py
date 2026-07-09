"""
Generic codegen utilities for AOTAutograd runtime wrappers.

PySourceBuilder accumulates indented Python source together with the globals
the generated code closes over and a fresh-name counter; _compile_and_exec_source
turns generated source into a callable (logging it as a structured-trace artifact
for tlparse and giving it a stable traceback filename). Both are independent of
any particular wrapper, so they live here rather than in subclass_codegen.
"""

import contextlib
import functools
import logging
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
    exec(code, globals_dict, local_dict)
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]
    return fn  # type: ignore[return-value]
