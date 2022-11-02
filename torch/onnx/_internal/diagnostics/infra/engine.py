"""A diagnostic engine based on SARIF."""

from __future__ import annotations

from typing import List, Optional, Type

from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version


class DiagnosticEngine:
    """A generic diagnostic engine based on SARIF.

    This class is the main interface for diagnostics. It manages the creation of diagnostic contexts.
    A DiagnosticContext provides the entry point for recording Diagnostics.
    See infra.DiagnosticContext for more details.

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

        Step 2: Create a diagnostic engine.
        >>> engine = DiagnosticEngine()

        Step 3: Start a new diagnostic context.
        >>> with engine.create_diagnostic_context("torch.onnx.export", version="1.0") as context:

        Step 4: Add diagnostics in your code.
        ...     context.diagnose(rules.rule1, infra.Level.ERROR)

        Step 5: Afterwards, get the SARIF log.
        >>> sarif_log = engine.sarif_log()
    """

    contexts: List[infra.DiagnosticContext]

    def __init__(self) -> None:
        self.contexts = []

    def sarif_log(self) -> sarif.SarifLog:
        return sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            runs=[context.sarif() for context in self.contexts],
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
        self.contexts.clear()

    def create_diagnostic_context(
        self,
        name: str,
        version: str,
        options: Optional[infra.DiagnosticOptions] = None,
        diagnostic_type: Type[infra.Diagnostic] = infra.Diagnostic,
    ) -> infra.DiagnosticContext:
        """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
        context = infra.DiagnosticContext(
            name, version, options, diagnostic_type=diagnostic_type
        )
        self.contexts.append(context)
        return context
