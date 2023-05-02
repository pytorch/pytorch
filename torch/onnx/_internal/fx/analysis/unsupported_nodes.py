from __future__ import annotations

from torch.onnx._internal.fx import diagnostics, function_dispatcher, _pass
from typing import List, Dict, Set
import dataclasses


@dataclasses.dataclass
class UnsupportedFxNodesAnalysisResult(_pass.AnalysisResult):
    unsupported_op_to_target_mapping: Dict[str, Set[str]]


class UnsupportedFxNodesAnalysis(_pass.Analysis):
    """An analysis that detects unsupported FX nodes in the graph."""

    def analyze(self) -> UnsupportedFxNodesAnalysisResult:
        """Analyze the graph and return a result that contains unsupported FX nodes."""
        errors: List[diagnostics.RuntimeErrorWithDiagnostic] = []
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                try:
                    function_dispatcher.get_symbolic_function(
                        self.diagnostic_context, node
                    )
                except diagnostics.RuntimeErrorWithDiagnostic as e:
                    errors.append(e)

        op_to_target_mapping = {}

        if errors:
            for e in errors:
                node_diagnostic = e.diagnostic
                assert isinstance(
                    node_diagnostic, diagnostics.UnsupportedFxNodeDiagnostic
                )
                node = node_diagnostic.unsupported_fx_node
                assert node is not None
                op = node.op
                target = node.target
                op_to_target_mapping.setdefault(op, set()).add(target)

        return UnsupportedFxNodesAnalysisResult(op_to_target_mapping)

    def lint(self) -> None:
        """Lint the graph and raise an error if unsupported FX nodes are found."""
        analysis_result = self.analyze()

        if not analysis_result.unsupported_op_to_target_mapping:
            return

        normalized_op_targets_map = {
            op: [str(target) for target in targets]
            for op, targets in analysis_result.unsupported_op_to_target_mapping.items()
        }

        rule = diagnostics.rules.unsupported_fx_node_analysis
        diagnostic = diagnostics.Diagnostic(
            rule,
            level=diagnostics.levels.ERROR,
            message=rule.format_message(normalized_op_targets_map),
        )
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
