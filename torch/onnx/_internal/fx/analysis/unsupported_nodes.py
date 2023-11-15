from __future__ import annotations

import dataclasses
from typing import Dict

from torch.onnx._internal.fx import _pass, diagnostics


@dataclasses.dataclass
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    unsupported_op_to_target_mapping: Dict[str, Dict[str, str]]


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
            # Make error message more precise by adding dtype information.
            # e.g.  Unsupported FX nodes: {'call_function': ['aten.mul.Tensor(complex)']}
            op: [f"{node}({dtype})" for node, dtype in targets.items()]
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

        op_to_target_mapping: Dict[str, Dict[str, str]] = {}
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                try:
                    # NOTE: OPSchema matcher is not in this analysis scope.
                    self.onnxfunction_dispatcher.get_function_overloads(
                        node, self.diagnostic_context
                    )
                except diagnostics.RuntimeErrorWithDiagnostic as e:
                    # Extract the dtype information from the diagnostic message.
                    if e.diagnostic.message and e.diagnostic.message.startswith(
                        "Cannot find any COMPLEX symbolic function"
                    ):
                        op_to_target_mapping.setdefault(node.op, {}).setdefault(
                            str(node.target), "complex"
                        )
                    elif e.diagnostic.message and e.diagnostic.message.startswith(
                        "Can ONLY find COMPLEX symbolic function"
                    ):
                        op_to_target_mapping.setdefault(node.op, {}).setdefault(
                            str(node.target), "real"
                        )
                    else:
                        op_to_target_mapping.setdefault(node.op, {}).setdefault(
                            str(node.target), ""
                        )

        analysis_result = UnsupportedFxNodesAnalysisResult(op_to_target_mapping)
        self._lint(analysis_result, diagnostic_level)
        return analysis_result
