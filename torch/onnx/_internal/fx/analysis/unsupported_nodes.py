# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
from typing import Dict

from torch.onnx._internal.fx import _pass, diagnostics, registration


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

        op_to_target_mapping: Dict[str, Dict[str, None]] = {}
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                # NOTE: OPSchema matcher is not in this analysis scope.
                internal_opname: registration.OpName = (
                    self.onnxfunction_dispatcher._get_aten_name(
                        node=node, diagnostic_context=self.diagnostic_context
                    )
                )
                overload_registration = (
                    self.onnxfunction_dispatcher.onnx_registry.is_registered_op(
                        namespace=internal_opname.namespace,
                        op_name=internal_opname.op_name,
                        overload=internal_opname.overload,
                    )
                )
                # NOTE: Fall back to default overload if the ONNX registry doesn't have the overload.
                default_registration = (
                    self.onnxfunction_dispatcher.onnx_registry.is_registered_op(
                        namespace=internal_opname.namespace,
                        op_name=internal_opname.op_name,
                        overload=None,
                    )
                )
                if not overload_registration and not default_registration:
                    op_to_target_mapping.setdefault(node.op, {}).setdefault(
                        str(node.target), None
                    )

        analysis_result = UnsupportedFxNodesAnalysisResult(op_to_target_mapping)
        self._lint(analysis_result, diagnostic_level)
        return analysis_result
