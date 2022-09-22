"""A diagnostic engine based on SARIF."""

import contextlib
from typing import Any, List, Optional, Tuple

from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif_om
from torch.onnx._internal.diagnostics.infra.options import (
    DiagnosticOptions,  # class import for readability
)
from torch.onnx._internal.diagnostics.infra.sarif_om import version as sarif_version


class DiagnosticEngine:
    """A generic diagnostic engine based on SARIF.

    This class is the main interface for diagnostics. It manages the start and finish of diagnostic runs.
    A Run can have multiple Diagnostics. Each run is powered by a DiagnosticTool.
    See infra.Run and infra.DiagnosticTool for more details.

    Attributes:

        options (DiagnosticOptions): The options for the diagnostic engine.
        sarif_log (sarif_om.Log): The SARIF log object.


    Examples:
        Step 1: Create a set of rules.
        >>> @dataclasses.dataclass(frozen=True)
        ... class _Rules:
        ...     rule1 = infra.Rule(
        ...         id="r1",
        ...         name="rule-1",
        ...         short_description={"text": "This is rule 1."}
        ...     )
        >>> rules = _Rules()

        Step 2: Create a diagnostic tool.
        >>> tool = infra.DiagnosticTool(
        ...     name="tool",
        ...     version="1.0.0",
        ...     rules=list(dataclasses.asdict(_Rules).values()),
        ... )

        Step 3: Create a diagnostic engine.
        >>> engine = DiagnosticEngine()

        Step 4: Start a new diagnostic run.
        >>> with engine.start_new_run(tool):

        Step 5: Add diagnostics to the run.
        ...     engine.diagnose(rules.rule1, infra.Level.ERROR)

        Step 6: End the diagnostic run and get the SARIF log.
        >>> sarif_log = engine.sarif_log
    """

    _runs: List[infra.Run]
    _current_run: Optional[infra.Run]
    _options: DiagnosticOptions
    _sarif_version: str = sarif_version.SARIF_VERSION
    _sarif_schema_uri: str = sarif_version.SARIF_SCHEMA_LINK

    def __init__(self) -> None:
        self._initialize()

    @property
    def options(self) -> DiagnosticOptions:
        """The options for the diagnostic engine."""
        return self._options

    @options.setter
    def options(self, new_options: DiagnosticOptions) -> None:
        """Set the options for the diagnostic engine."""
        self._options = new_options

    @property
    def sarif_log(self) -> sarif_om.SarifLog:
        return sarif_om.SarifLog(
            runs=[run.sarif() for run in self._runs],
            version=self._sarif_version,
            schema_uri=self._sarif_schema_uri,
        )

    def diagnose(
        self,
        rule: infra.Rule,
        level: infra.Level,
        message_args: Optional[Tuple[Any, ...]] = None,
        **additional_kwargs,
    ) -> infra.Diagnostic:
        if not self._current_run:
            raise RuntimeError(
                "Cannot add diagnostic to run. No run is currently active. "
                "Use start_new_run() to start a new run."
            )

        return self._current_run.add_diagnostic(
            rule=rule,
            level=level,
            message_args=message_args,
        )

    def __str__(self) -> str:
        return self.to_json()

    def pretty_print(self) -> str:
        # TODO: Implement this.
        return self.to_json()

    def to_json(self) -> str:
        return formatter.sarif_to_json(self.sarif_log)

    def reset(self) -> None:
        self._initialize()

    def _initialize(self, options: Optional[DiagnosticOptions] = None) -> None:
        self.options = options if options else DiagnosticOptions()
        self._runs = []
        self._current_run = None

    @contextlib.contextmanager
    def start_new_run(self, tool: infra.DiagnosticTool):
        previous_run = self._current_run
        self._current_run = infra.Run(tool)
        self._runs.append(self._current_run)
        try:
            yield
        finally:
            self._end_current_run(previous_run)

    def _end_current_run(self, previous_run=None) -> None:
        self._current_run = previous_run
