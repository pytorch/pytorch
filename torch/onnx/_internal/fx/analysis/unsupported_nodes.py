from __future__ import annotations

import dataclasses
from typing import Dict, List, Set

from torch.onnx._internal.fx import _pass, diagnostics


@dataclasses.dataclass
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    unsupported_op_to_target_mapping: Dict[str, Set[str]]


class UnsupportedFxNodesAnalysis(_pass.Analysis):
    """An analysis that detects unsupported FX nodes in the graph."""

    def _lint(
        self,
        analysis_result: UnsupportedFxNodesAnalysisResult,
        diagnostic_level: diagnostics.infra.Level,
    ):
        """Lint the graph and emit diagnostics if unsupported FX nodes are found."""
        if not analysis_result.unsupported_op_to_target_mapping:
            return

        normalized_op_targets_map = {
            op: [str(target) for target in targets]
            for op, targets in analysis_result.unsupported_op_to_target_mapping.items()
        }

        rule = diagnostics.rules.unsupported_fx_node_analysis
        diagnostic = diagnostics.Diagnostic(
            rule,
            level=diagnostic_level,
            message=rule.format_message(normalized_op_targets_map),
        )
        self.diagnostic_context.log_and_raise_if_error(diagnostic)

    def analyze(
        self, diagnostic_level: diagnostics.infra.Level
    ) -> UnsupportedFxNodesAnalysisResult:
        """Analyze the graph, emit diagnostics and return a result that contains unsupported FX nodes.

        Args:
            diagnostic_level: The diagnostic level to use when emitting diagnostics.

        Returns:
            An analysis result that contains unsupported FX nodes.

        Raises:
            RuntimeErrorWithDiagnostic: If diagnostics are emitted and the diagnostic
                level is `ERROR`.
        """
        errors: List[diagnostics.RuntimeErrorWithDiagnostic] = []
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                try:
                    # NOTE: OPSchema matcher is not in this analysis scope.
                    aten_name = self.dispatcher._get_aten_name(
                        node, self.diagnostic_context
                    )
                    self.dispatcher._get_function_overloads(
                        node, aten_name, self.diagnostic_context
                    )
                except diagnostics.RuntimeErrorWithDiagnostic as e:
                    errors.append(e)

        op_to_target_mapping: Dict[str, Set[str]] = {}

        if errors:
            for error in errors:
                node_diagnostic = error.diagnostic
                assert isinstance(
                    node_diagnostic, diagnostics.UnsupportedFxNodeDiagnostic
                )
                node = node_diagnostic.unsupported_fx_node
                assert node is not None
                op = node.op
                target = node.target
                op_to_target_mapping.setdefault(op, set()).add(str(target))

        analysis_result = UnsupportedFxNodesAnalysisResult(op_to_target_mapping)
        self._lint(analysis_result, diagnostic_level)
        return analysis_result
