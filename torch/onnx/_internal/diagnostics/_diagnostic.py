"""Diagnostic components for PyTorch ONNX export."""

from typing import Any, Optional, Tuple

import torch
from torch.onnx._internal.diagnostics import _rules, infra

levels = infra.Level


class ExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    def __init__(
        self,
        rule: infra.Rule,
        level: infra.Level,
        message_args: Optional[Tuple[Any, ...]],
        **kwargs,
    ) -> None:
        super().__init__(rule, level, message_args, **kwargs)

    def with_cpp_stack(self) -> "ExportDiagnostic":
        # TODO: Implement this.
        # self.stacks.append(...)
        return self

    def with_python_stack(self) -> "ExportDiagnostic":
        # TODO: Implement this.
        # self.stacks.append(...)
        return self

    def with_model_source_location(self) -> "ExportDiagnostic":
        # TODO: Implement this.
        # self.locations.append(...)
        return self

    def with_export_source_location(self) -> "ExportDiagnostic":
        # TODO: Implement this.
        # self.locations.append(...)
        return self


class ExportDiagnosticTool(infra.DiagnosticTool):
    """Base class for all export diagnostic tools.

    This class is used to represent all export diagnostic tools. It is a subclass
    of infra.DiagnosticTool.
    """

    def __init__(self) -> None:
        super().__init__(
            name="torch.onnx.export",
            version=torch.__version__,
            rules=_rules.rules,
            diagnostic_type=ExportDiagnostic,
        )


class ExportDiagnosticEngine(infra.DiagnosticEngine):
    """PyTorch ONNX Export diagnostic engine.

    The only purpose of this class is to provide a background context for export diagnose calls.
    Since diagnostic context is currently being accessed as a global variable, we need to provide a default background context
    to fallback to.
    """

    _background_context: infra.DiagnosticContext

    def __init__(self) -> None:
        super().__init__()
        self._background_context = infra.DiagnosticContext(
            ExportDiagnosticTool(), options=None
        )

    @property
    def background_context(self) -> infra.DiagnosticContext:
        return self._background_context

    def sarif_log(self):
        runs = [context.sarif() for context in self._contexts]
        runs.append(self._background_context.sarif())
        return self._sarif_log(runs)


engine = ExportDiagnosticEngine()
context = engine.background_context
