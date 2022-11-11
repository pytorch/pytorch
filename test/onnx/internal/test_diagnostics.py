# Owner(s): ["module: onnx"]

import contextlib
import dataclasses
import io
import typing
import unittest
from typing import AbstractSet, Tuple

import torch
from torch.onnx import errors
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.testing._internal import common_utils


def _assert_has_diagnostics(
    engine: infra.DiagnosticEngine,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    sarif_log = engine.sarif_log()
    unseen_pairs = {(rule.id, level.value) for rule, level in rule_level_pairs}
    actual_results = []
    for run in sarif_log.runs:
        if run.results is None:
            continue
        for result in run.results:
            id_level_pair = (result.rule_id, result.level)
            unseen_pairs.discard(id_level_pair)
            actual_results.append(id_level_pair)

    if unseen_pairs:
        raise AssertionError(
            f"Expected diagnostic results of rule id and level pair {unseen_pairs} not found. "
            f"Actual diagnostic results: {actual_results}"
        )


@contextlib.contextmanager
def assert_all_diagnostics(
    test_suite: unittest.TestCase,
    engine: infra.DiagnosticEngine,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    """Context manager to assert that all diagnostics are emitted.

    Usage:
        with assert_all_diagnostics(
            self,
            diagnostics.engine,
            {(rule, infra.Level.Error)},
        ):
            torch.onnx.export(...)

    Args:
        test_suite: The test suite instance.
        engine: The diagnostic engine.
        rule_level_pairs: A set of rule and level pairs to assert.

    Returns:
        A context manager.

    Raises:
        AssertionError: If not all diagnostics are emitted.
    """

    try:
        yield
    except errors.OnnxExporterError:
        test_suite.assertIn(infra.Level.ERROR, {level for _, level in rule_level_pairs})
    finally:
        _assert_has_diagnostics(engine, rule_level_pairs)


def assert_diagnostic(
    test_suite: unittest.TestCase,
    engine: infra.DiagnosticEngine,
    rule: infra.Rule,
    level: infra.Level,
):
    """Context manager to assert that a diagnostic is emitted.

    Usage:
        with assert_diagnostic(
            self,
            diagnostics.engine,
            rule,
            infra.Level.Error,
        ):
            torch.onnx.export(...)

    Args:
        test_suite: The test suite instance.
        engine: The diagnostic engine.
        rule: The rule to assert.
        level: The level to assert.

    Returns:
        A context manager.

    Raises:
        AssertionError: If the diagnostic is not emitted.
    """

    return assert_all_diagnostics(test_suite, engine, {(rule, level)})


class TestOnnxDiagnostics(common_utils.TestCase):
    """Test cases for diagnostics emitted by the ONNX export code."""

    def setUp(self):
        engine = diagnostics.engine
        engine.clear()
        self._sample_rule = diagnostics.rules.missing_custom_symbolic_function
        super().setUp()

    def _trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp(
        self,
    ) -> diagnostics.ExportDiagnostic:
        class CustomAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def symbolic(g, x, y):
                return g.op("custom::CustomAdd", x, y)

        class M(torch.nn.Module):
            def forward(self, x):
                return CustomAdd.apply(x, x)

        # trigger warning for missing shape inference.
        rule = diagnostics.rules.node_missing_onnx_shape_inference
        torch.onnx.export(M(), torch.randn(3, 4), io.BytesIO())

        context = diagnostics.engine.contexts[-1]
        for diagnostic in context.diagnostics:
            if (
                diagnostic.rule == rule
                and diagnostic.level == diagnostics.levels.WARNING
            ):
                return typing.cast(diagnostics.ExportDiagnostic, diagnostic)
        raise AssertionError("No diagnostic found.")

    def test_assert_diagnostic_raises_when_diagnostic_not_found(self):
        with self.assertRaises(AssertionError):
            with assert_diagnostic(
                self,
                diagnostics.engine,
                diagnostics.rules.node_missing_onnx_shape_inference,
                diagnostics.levels.WARNING,
            ):
                pass

    def test_cpp_diagnose_emits_warning(self):
        with assert_diagnostic(
            self,
            diagnostics.engine,
            diagnostics.rules.node_missing_onnx_shape_inference,
            diagnostics.levels.WARNING,
        ):
            # trigger warning for missing shape inference.
            self._trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp()

    def test_py_diagnose_emits_error(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x)

        with assert_diagnostic(
            self,
            diagnostics.engine,
            diagnostics.rules.operator_supported_in_newer_opset_version,
            diagnostics.levels.ERROR,
        ):
            # trigger error for operator unsupported until newer opset version.
            torch.onnx.export(
                M(),
                torch.randn(3, 4),
                io.BytesIO(),
                opset_version=9,
            )

    def test_diagnostics_engine_records_diagnosis_reported_outside_of_export(
        self,
    ):
        sample_level = diagnostics.levels.ERROR
        with assert_diagnostic(
            self,
            diagnostics.engine,
            self._sample_rule,
            sample_level,
        ):
            diagnostics.context.diagnose(self._sample_rule, sample_level)

    def test_diagnostics_records_python_call_stack(self):
        diagnostic = diagnostics.ExportDiagnostic(
            self._sample_rule, diagnostics.levels.NOTE
        )
        stack = diagnostic.python_call_stack
        assert stack is not None  # for mypy
        self.assertGreater(len(stack.frames), 0)
        frame = stack.frames[0]
        assert frame.location.snippet is not None  # for mypy
        self.assertIn("self._sample_rule", frame.location.snippet)
        assert frame.location.uri is not None  # for mypy
        self.assertIn("test_diagnostics.py", frame.location.uri)

    def test_diagnostics_records_cpp_call_stack(self):
        diagnostic = (
            self._trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp()
        )
        stack = diagnostic.cpp_call_stack
        assert stack is not None  # for mypy
        self.assertGreater(len(stack.frames), 0)
        frame_messages = [frame.location.message for frame in stack.frames]
        self.assertTrue(
            any(
                isinstance(message, str)
                and "torch::jit::ONNXShapeTypeInference" in message
                for message in frame_messages
            )
        )


@dataclasses.dataclass
class _RuleCollectionForTest(infra.RuleCollection):
    rule_without_message_args: infra.Rule = dataclasses.field(
        default=infra.Rule(
            "1",
            "rule-without-message-args",
            message_default_template="rule message",
        )
    )


class TestDiagnosticsInfra(common_utils.TestCase):
    """Test cases for diagnostics infra."""

    def setUp(self):
        self.engine = infra.DiagnosticEngine()
        self.rules = _RuleCollectionForTest()
        with contextlib.ExitStack() as stack:
            self.context = stack.enter_context(
                self.engine.create_diagnostic_context("test", "1.0.0")
            )
            self.addCleanup(stack.pop_all().close)
        return super().setUp()

    def test_diagnostics_engine_records_diagnosis_reported_in_nested_contexts(
        self,
    ):
        with self.engine.create_diagnostic_context("inner_test", "1.0.1") as context:
            context.diagnose(self.rules.rule_without_message_args, infra.Level.WARNING)
            sarif_log = self.engine.sarif_log()
            self.assertEqual(len(sarif_log.runs), 2)
            self.assertEqual(len(sarif_log.runs[0].results), 0)
            self.assertEqual(len(sarif_log.runs[1].results), 1)
        self.context.diagnose(self.rules.rule_without_message_args, infra.Level.ERROR)
        sarif_log = self.engine.sarif_log()
        self.assertEqual(len(sarif_log.runs), 2)
        self.assertEqual(len(sarif_log.runs[0].results), 1)
        self.assertEqual(len(sarif_log.runs[1].results), 1)

    def test_diagnostics_engine_records_diagnosis_with_custom_rules(self):
        custom_rules = infra.RuleCollection.custom_collection_from_list(
            "CustomRuleCollection",
            [
                infra.Rule(
                    "1",
                    "custom-rule",
                    message_default_template="custom rule message",
                ),
                infra.Rule(
                    "2",
                    "custom-rule-2",
                    message_default_template="custom rule message 2",
                ),
            ],
        )

        with self.engine.create_diagnostic_context(
            "custom_rules", "1.0"
        ) as diagnostic_context:
            with assert_all_diagnostics(
                self,
                self.engine,
                {
                    (custom_rules.custom_rule, infra.Level.WARNING),  # type: ignore[attr-defined]
                    (custom_rules.custom_rule_2, infra.Level.ERROR),  # type: ignore[attr-defined]
                },
            ):
                diagnostic_context.diagnose(
                    custom_rules.custom_rule, infra.Level.WARNING  # type: ignore[attr-defined]
                )
                diagnostic_context.diagnose(
                    custom_rules.custom_rule_2, infra.Level.ERROR  # type: ignore[attr-defined]
                )


if __name__ == "__main__":
    common_utils.run_tests()
