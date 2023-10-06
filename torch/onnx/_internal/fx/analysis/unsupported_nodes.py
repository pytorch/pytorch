from __future__ import annotations

import dataclasses
from typing import Dict, List

import torch
from torch.onnx._internal.fx import _pass, diagnostics


@dataclasses.dataclass
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    unsupported_op_to_target_mapping: Dict[str, Dict[str, None]]


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
            op: list(targets.keys())
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
        unsupported_nodes: List[torch.fx.Node] = []
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                try:
                    # NOTE: OPSchema matcher is not in this analysis scope.
                    self.onnxfunction_dispatcher.get_function_overloads(
                        node, self.diagnostic_context
                    )
                except diagnostics.RuntimeErrorWithDiagnostic as e:
                    unsupported_nodes.append(node)

        op_to_target_mapping: Dict[str, Dict[str, None]] = {}

        for node in unsupported_nodes:
            op = node.op
            target = node.target
            op_to_target_mapping.setdefault(op, {}).setdefault(str(target), None)

        analysis_result = UnsupportedFxNodesAnalysisResult(op_to_target_mapping)
        self._lint(analysis_result, diagnostic_level)
        return analysis_result
