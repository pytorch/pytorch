"""Diagnostic components for PyTorch ONNX export."""

import contextlib
from typing import Optional, TypeVar

import torch
from torch.onnx._internal.diagnostics import infra

# This is a workaround for mypy not supporting Self from typing_extensions.
_ExportDiagnostic = TypeVar("_ExportDiagnostic", bound="ExportDiagnostic")


class ExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def with_cpp_stack(self: _ExportDiagnostic) -> _ExportDiagnostic:
        # TODO: Implement this.
        # self.stacks.append(...)
        raise NotImplementedError()
        return self

    def with_python_stack(self: _ExportDiagnostic) -> _ExportDiagnostic:
        # TODO: Implement this.
        # self.stacks.append(...)
        raise NotImplementedError()
        return self

    def with_model_source_location(
        self: _ExportDiagnostic,
    ) -> _ExportDiagnostic:
        # TODO: Implement this.
        # self.locations.append(...)
        raise NotImplementedError()
        return self

    def with_export_source_location(
        self: _ExportDiagnostic,
    ) -> _ExportDiagnostic:
        # TODO: Implement this.
        # self.locations.append(...)
        raise NotImplementedError()
        return self


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
            options=None,
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
context = engine.background_context


@contextlib.contextmanager
def create_export_diagnostic_context():
    """Create a diagnostic context for export.

    This is a workaround for code robustness since diagnostic context is accessed by
    export internals via global variable. See `ExportDiagnosticEngine` for more details.
    """
    global context
    context = engine.create_diagnostic_context(
        "torch.onnx.export", torch.__version__, diagnostic_type=ExportDiagnostic
    )
    try:
        yield context
    finally:
        context = engine.background_context


def diagnose(
    rule: infra.Rule,
    level: infra.Level,
    message: Optional[str] = None,
    **kwargs,
) -> ExportDiagnostic:
    """Creates a diagnostic and record it in the global diagnostic context.

    This is a wrapper around `context.record` that uses the global diagnostic context.
    """
    global context
    diagnostic = ExportDiagnostic(rule, level, message, **kwargs)
    context.add_diagnostic(diagnostic)
    return diagnostic
