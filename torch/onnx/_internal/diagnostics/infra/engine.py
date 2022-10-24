"""A diagnostic engine based on SARIF."""

from __future__ import annotations

from typing import List, Optional

from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version


class DiagnosticEngine:
    """A generic diagnostic engine based on SARIF.

    This class is the main interface for diagnostics. It manages the creation of diagnostic contexts.
    A DiagnosticContext provides the entry point for recording Diagnostics.
    Each DiagnosticContext is powered by a DiagnosticTool, which can be customized with
    custom RuleCollection and Diagnostic type.
    See infra.DiagnosticContext and infra.DiagnosticTool for more details.

    Examples:
        Step 1: Create a set of rules.
        >>> rules = infra.RuleCollection.from_list(
        ...     "CustomRuleCollection",
        ...     [
        ...         infra.Rule(
        ...             id="r1",
        ...             name="rule-1",
        ...             message_default_template="Mising xxx",
        ...         ),
        ...     ],
        ... )

        Step 2: Create a diagnostic tool.
        >>> tool = infra.DiagnosticTool(
        ...     name="tool",
        ...     version="1.0.0",
        ...     rules=rules,
        ... )

        Step 3: Create a diagnostic engine.
        >>> engine = DiagnosticEngine()

        Step 4: Start a new diagnostic context.
        >>> with engine.start_diagnostic_context(tool) as context:

        Step 5: Add diagnostics in your code.
        ...     context.diagnose(rules.rule1, infra.Level.ERROR)

        Step 6: Afterwards, get the SARIF log.
        >>> sarif_log = engine.sarif_log()
    """

    _contexts: List[infra.DiagnosticContext]

    def __init__(self) -> None:
        self._contexts = []

    def sarif_log(self) -> sarif.SarifLog:
        return sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            runs=[context.sarif() for context in self._contexts],
        )

    def __str__(self) -> str:
        # TODO: pretty print.
        return self.to_json()

    def __repr__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        return formatter.sarif_to_json(self.sarif_log())

    def clear(self) -> None:
        """Clears all diagnostic contexts."""
        self._contexts.clear()

    def create_diagnostic_context(
        self,
        tool: infra.DiagnosticTool,
        options: Optional[infra.DiagnosticOptions] = None,
    ) -> infra.DiagnosticContext:
        context = infra.DiagnosticContext(tool, options)
        self._contexts.append(context)
        return context
