# Owner(s): ["module: onnx"]
from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import typing
import unittest
from typing import AbstractSet, Protocol, Tuple

import torch
from torch.onnx import errors
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import sarif
from torch.testing._internal import common_utils


class _SarifLogBuilder(Protocol):
    def sarif_log(self) -> sarif.SarifLog:
        ...


def _assert_has_diagnostics(
    sarif_log_builder: _SarifLogBuilder,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    sarif_log = sarif_log_builder.sarif_log()
    unseen_pairs = {(rule.id, level.name.lower()) for rule, level in rule_level_pairs}
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
    sarif_log_builder: _SarifLogBuilder,
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
        sarif_log_builder: The SARIF log builder.
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
        _assert_has_diagnostics(sarif_log_builder, rule_level_pairs)


def assert_diagnostic(
    test_suite: unittest.TestCase,
    sarif_log_builder: _SarifLogBuilder,
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
        sarif_log_builder: The SARIF log builder.
        rule: The rule to assert.
        level: The level to assert.

    Returns:
        A context manager.

    Raises:
        AssertionError: If the diagnostic is not emitted.
    """

    return assert_all_diagnostics(test_suite, sarif_log_builder, {(rule, level)})


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
            diagnostic = infra.Diagnostic(self._sample_rule, sample_level)
            diagnostics.export_context().log(diagnostic)

    def test_diagnostics_records_python_call_stack(self):
        diagnostic = diagnostics.ExportDiagnostic(self._sample_rule, diagnostics.levels.NOTE)  # fmt: skip
        # Do not break the above line, otherwise it will not work with Python-3.8+
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
        # node missing onnx shape inference warning only comes from ToONNX (_jit_pass_onnx)
        # after node-level shape type inference and processed symbolic_fn output type
        self.assertTrue(
            any(
                isinstance(message, str) and "torch::jit::NodeToONNX" in message
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
        self.rules = _RuleCollectionForTest()
        with contextlib.ExitStack() as stack:
            self.context = stack.enter_context(infra.DiagnosticContext("test", "1.0.0"))
            self.addCleanup(stack.pop_all().close)
        return super().setUp()

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

        with assert_all_diagnostics(
            self,
            self.context,
            {
                (custom_rules.custom_rule, infra.Level.WARNING),  # type: ignore[attr-defined]
                (custom_rules.custom_rule_2, infra.Level.ERROR),  # type: ignore[attr-defined]
            },
        ):
            diagnostic1 = infra.Diagnostic(
                custom_rules.custom_rule, infra.Level.WARNING  # type: ignore[attr-defined]
            )
            self.context.log(diagnostic1)

            diagnostic2 = infra.Diagnostic(
                custom_rules.custom_rule_2, infra.Level.ERROR  # type: ignore[attr-defined]
            )
            self.context.log(diagnostic2)

    def test_diagnostic_context_logs_with_correct_logger_level_based_on_diagnostic_level(
        self,
    ):
        diagnostic_logging_level_pairs = [
            (infra.Level.NONE, logging.DEBUG),
            (infra.Level.NOTE, logging.INFO),
            (infra.Level.WARNING, logging.WARNING),
            (infra.Level.ERROR, logging.ERROR),
        ]

        for diagnostic_level, expected_logger_level in diagnostic_logging_level_pairs:
            with self.assertLogs(
                self.context.logger, level=expected_logger_level
            ) as assert_log_context:
                self.context.log(
                    infra.Diagnostic(
                        self.rules.rule_without_message_args, diagnostic_level
                    )
                )
                for record in assert_log_context.records:
                    self.assertEqual(record.levelno, expected_logger_level)

    def test_diagnostic_context_raises_if_diagnostic_is_error(self):
        with self.assertRaises(infra.RuntimeErrorWithDiagnostic):
            self.context.log_and_raise_if_error(
                infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.ERROR
                )
            )

    def test_diagnostic_context_raises_original_exception_from_diagnostic_created_from_it(
        self,
    ):
        with self.assertRaises(ValueError):
            try:
                raise ValueError("original exception")
            except ValueError as e:
                diagnostic = infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.ERROR
                )
                diagnostic = diagnostic.with_source_exception(e)
                self.context.log_and_raise_if_error(diagnostic)

    def test_diagnostic_context_raises_if_diagnostic_is_warning_and_warnings_as_errors_is_true(
        self,
    ):
        with self.assertRaises(infra.RuntimeErrorWithDiagnostic):
            self.context.options.warnings_as_errors = True
            self.context.log_and_raise_if_error(
                infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.WARNING
                )
            )


if __name__ == "__main__":
    common_utils.run_tests()
