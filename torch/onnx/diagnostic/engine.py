import json
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Set

import attr

import torch
from torch.onnx import diagnostic, sarif_om
from torch.onnx.diagnostic import formatter
from torch.onnx.diagnostic.options import (
    DiagnosticOptions,  # class import for readability
)


class DiagnosticEngine:
    _sarif_log: sarif_om.SarifLog
    _options: DiagnosticOptions
    _used_rules: Set[str]
    _current_run: Optional[sarif_om.Run]

    @property
    def options(self) -> DiagnosticOptions:
        return self._options

    @options.setter
    def options(self, new_options: DiagnosticOptions) -> None:
        self._options = new_options

    @property
    def sarif_log(self) -> sarif_om.SarifLog:
        return self._sarif_log

    def __init__(self) -> None:
        self._initialize()

    def diagnose(
        self,
        rule: diagnostic.Rule,
        level: diagnostic.Level,
        src_location=None,
        export_location=None,
        message_args=None,
        *additional_args,
    ) -> diagnostic.Diagnostic:
        if message_args is None:
            message_args = ()
        sarif_result = diagnostic.Diagnostic(
            rule_id=rule.id,
            level=level.value,
            # src_location,
            # export_location,
            message=rule.message_strings["default"]["text"].format(
                *message_args
            ),
        )
        sarif_run = self._current_run
        if not sarif_run:
            raise RuntimeError(
                "SARIF log 'runs' uninitialized. "
                "'diagnose' must be called within 'torch.onnx.export'."
            )

        sarif_run.results.append(sarif_result)
        # update rules
        if rule.id not in self._used_rules:
            self._used_rules.add(rule.id)
            sarif_run.tool.driver.rules.append(rule)

        return sarif_result

    def __str__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        return formatter.sarif_om_to_json(self._sarif_log)

    def reset(self) -> None:
        self._initialize()

    def _initialize(self, options: Optional[DiagnosticOptions] = None) -> None:
        self.options = options if options else DiagnosticOptions()
        # TOOD: load sarif attributes from somewhere.
        self._sarif_log = sarif_om.SarifLog(
            runs=[],
            version="2.1.0",
            schema_uri="http://docs.oasis-open.org/sarif/sarif/v2.1.0/cs01/schemas/sarif-schema-2.1.0.json",
        )
        self._sarif_log.runs = []
        self._used_rules = set()
        self._current_run = None

    def start_new_run(self) -> None:
        runs = self._sarif_log.runs
        runs.append(
            sarif_om.Run(
                tool=sarif_om.Tool(
                    driver=sarif_om.ToolComponent(
                        name="PyTorch.ONNX",
                        version=torch.__version__,
                        rules=[],
                    )
                ),
                results=[],
            )
        )
        self._current_run = runs[-1]
        self._used_rules.clear()

    def end_current_run(self) -> None:
        self._current_run = None
        self._used_rules.clear()


engine = DiagnosticEngine()
