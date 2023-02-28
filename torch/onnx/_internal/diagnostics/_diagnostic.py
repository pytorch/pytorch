"""Diagnostic components for PyTorch ONNX export."""
from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Optional

import torch

from torch.onnx._internal.diagnostics import infra
from torch.utils import cpp_backtrace


def _cpp_call_stack(frames_to_skip: int = 0, frames_to_log: int = 32) -> infra.Stack:
    """Returns the current C++ call stack.

    This function utilizes `torch.utils.cpp_backtrace` to get the current C++ call stack.
    The returned C++ call stack is a concatenated string of the C++ call stack frames.
    Each frame is separated by a newline character, in the same format of
    r"frame #[0-9]+: (?P<frame_info>.*)". More info at `c10/util/Backtrace.cpp`.

    """
    # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
    frames = cpp_backtrace.get_cpp_backtrace(frames_to_skip, frames_to_log).split("\n")
    frame_messages = []
    for frame in frames:
        segments = frame.split(":", 1)
        if len(segments) == 2:
            frame_messages.append(segments[1].strip())
        else:
            frame_messages.append("<unknown frame>")
    return infra.Stack(
        frames=[
            infra.StackFrame(location=infra.Location(message=message))
            for message in frame_messages
        ]
    )


class ExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    python_call_stack: Optional[infra.Stack] = None
    cpp_call_stack: Optional[infra.Stack] = None

    def __init__(
        self,
        *args,
        frames_to_skip: int = 1,
        cpp_stack: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.python_call_stack = self.record_python_call_stack(
            frames_to_skip=frames_to_skip
        )
        if cpp_stack:
            self.cpp_call_stack = self.record_cpp_call_stack(
                frames_to_skip=frames_to_skip
            )

    def record_cpp_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current C++ call stack in the diagnostic."""
        # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
        # No need to skip this function because python frame is not recorded
        # in cpp call stack.
        stack = _cpp_call_stack(frames_to_skip=frames_to_skip)
        stack.message = "C++ call stack"
        self.with_stack(stack)
        return stack

    def record_fx_graphmodule(self, gm: torch.fx.GraphModule) -> None:
        self.with_graph(infra.Graph(gm.print_readable(False), gm.__class__.__name__))


class ExportDiagnosticEngine(infra.DiagnosticEngine):
    """PyTorch ONNX Export diagnostic engine.

    The only purpose of creating this class instead of using the base class directly
    is to provide a background context for `diagnose` calls inside exporter.

    By design, one `torch.onnx.export` call should initialize one diagnostic context.
    All `diagnose` calls inside exporter should be made in the context of that export.
    However, since diagnostic context is currently being accessed via a global variable,
    there is no guarantee that the context is properly initialized. Therefore, we need
    to provide a default background context to fallback to, otherwise any invocation of
    exporter internals, e.g. unit tests, will fail due to missing diagnostic context.
    This can be removed once the pipeline for context to flow through the exporter is
    established.
    """

    _background_context: infra.DiagnosticContext

    def __init__(self) -> None:
        super().__init__()
        self._background_context = infra.DiagnosticContext(
            name="torch.onnx",
            version=torch.__version__,
            diagnostic_type=ExportDiagnostic,
        )

    @property
    def background_context(self) -> infra.DiagnosticContext:
        return self._background_context

    def clear(self):
        super().clear()
        self._background_context.diagnostics.clear()

    def sarif_log(self):
        log = super().sarif_log()
        log.runs.append(self._background_context.sarif())
        return log


engine = ExportDiagnosticEngine()
_context = engine.background_context


@contextlib.contextmanager
def create_export_diagnostic_context() -> Generator[
    infra.DiagnosticContext, None, None
]:
    """Create a diagnostic context for export.

    This is a workaround for code robustness since diagnostic context is accessed by
    export internals via global variable. See `ExportDiagnosticEngine` for more details.
    """
    global _context
    assert (
        _context == engine.background_context
    ), "Export context is already set. Nested export is not supported."
    _context = engine.create_diagnostic_context(
        "torch.onnx.export", torch.__version__, diagnostic_type=ExportDiagnostic
    )
    try:
        yield _context
    finally:
        _context.pretty_print(_context.options.log_verbose, _context.options.log_level)
        _context = engine.background_context


def diagnose(
    rule: infra.Rule,
    level: infra.Level,
    message: Optional[str] = None,
    frames_to_skip: int = 2,
    **kwargs,
) -> ExportDiagnostic:
    """Creates a diagnostic and record it in the global diagnostic context.

    This is a wrapper around `context.record` that uses the global diagnostic context.
    """
    # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
    diagnostic = ExportDiagnostic(
        rule, level, message, frames_to_skip=frames_to_skip, **kwargs
    )
    export_context().add_diagnostic(diagnostic)
    return diagnostic


def export_context() -> infra.DiagnosticContext:
    global _context
    return _context
