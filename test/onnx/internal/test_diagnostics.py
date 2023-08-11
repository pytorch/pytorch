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
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.fx import diagnostics as fx_diagnostics
from torch.testing._internal import common_utils, logging_utils


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


@dataclasses.dataclass
class _RuleCollectionForTest(infra.RuleCollection):
    rule_without_message_args: infra.Rule = dataclasses.field(
        default=infra.Rule(
            "1",
            "rule-without-message-args",
            message_default_template="rule message",
        )
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


class TestDynamoOnnxDiagnostics(common_utils.TestCase):
    """Test cases for diagnostics emitted by the Dynamo ONNX export code."""

    def setUp(self):
        self.diagnostic_context = fx_diagnostics.DiagnosticContext("dynamo_export", "")
        self.rules = _RuleCollectionForTest()
        return super().setUp()

    def test_log_is_recorded_in_sarif_additional_messages_according_to_diagnostic_options_verbosity_level(
        self,
    ):
        logging_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ]
        for verbosity_level in logging_levels:
            self.diagnostic_context.options.verbosity_level = verbosity_level
            with self.diagnostic_context:
                diagnostic = fx_diagnostics.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.NONE
                )
                additional_messages_count = len(diagnostic.additional_messages)
                for log_level in logging_levels:
                    diagnostic.log(level=log_level, message="log message")
                    if log_level >= verbosity_level:
                        self.assertGreater(
                            len(diagnostic.additional_messages),
                            additional_messages_count,
                            f"Additional message should be recorded when log level is {log_level} "
                            f"and verbosity level is {verbosity_level}",
                        )
                    else:
                        self.assertEqual(
                            len(diagnostic.additional_messages),
                            additional_messages_count,
                            f"Additional message should not be recorded when log level is "
                            f"{log_level} and verbosity level is {verbosity_level}",
                        )

    def test_torch_logs_environment_variable_precedes_diagnostic_options_verbosity_level(
        self,
    ):
        self.diagnostic_context.options.verbosity_level = logging.ERROR
        with logging_utils.log_settings("onnx_diagnostics"), self.diagnostic_context:
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )
            additional_messages_count = len(diagnostic.additional_messages)
            diagnostic.debug("message")
            self.assertGreater(
                len(diagnostic.additional_messages), additional_messages_count
            )

    def test_log_is_not_emitted_to_terminal_when_log_artifact_is_not_enabled(self):
        self.diagnostic_context.options.verbosity_level = logging.INFO
        with self.diagnostic_context:
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )

            with self.assertLogs(
                diagnostic.logger, level=logging.INFO
            ) as assert_log_context:
                diagnostic.info("message")
                # NOTE: self.assertNoLogs only exist >= Python 3.10
                # Add this dummy log such that we can pass self.assertLogs, and inspect
                # assert_log_context.records to check if the log we don't want is not emitted.
                diagnostic.logger.log(logging.ERROR, "dummy message")

            self.assertEqual(len(assert_log_context.records), 1)

    def test_log_is_emitted_to_terminal_when_log_artifact_is_enabled(self):
        self.diagnostic_context.options.verbosity_level = logging.INFO

        with logging_utils.log_settings("onnx_diagnostics"), self.diagnostic_context:
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )

            with self.assertLogs(diagnostic.logger, level=logging.INFO):
                diagnostic.info("message")


class TestTorchScriptOnnxDiagnostics(common_utils.TestCase):
    """Test cases for diagnostics emitted by the TorchScript ONNX export code."""

    def setUp(self):
        engine = diagnostics.engine
        engine.clear()
        self._sample_rule = diagnostics.rules.missing_custom_symbolic_function
        super().setUp()

    def _trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp(
        self,
    ) -> diagnostics.TorchScriptOnnxExportDiagnostic:
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
                return typing.cast(
                    diagnostics.TorchScriptOnnxExportDiagnostic, diagnostic
                )
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
        diagnostic = diagnostics.TorchScriptOnnxExportDiagnostic(self._sample_rule, diagnostics.levels.NOTE)  # fmt: skip
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


@common_utils.instantiate_parametrized_tests
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

    def test_diagnostic_log_is_not_emitted_when_level_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            with self.assertLogs(
                diagnostic.logger, level=verbosity_level
            ) as assert_log_context:
                diagnostic.log(logging.DEBUG, "debug message")
                # NOTE: self.assertNoLogs only exist >= Python 3.10
                # Add this dummy log such that we can pass self.assertLogs, and inspect
                # assert_log_context.records to check if the log level is correct.
                diagnostic.log(logging.INFO, "info message")

        for record in assert_log_context.records:
            self.assertGreaterEqual(record.levelno, logging.INFO)
        self.assertFalse(
            any(
                message.find("debug message") >= 0
                for message in diagnostic.additional_messages
            )
        )

    def test_diagnostic_log_is_emitted_when_level_not_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            level_message_pairs = [
                (logging.INFO, "info message"),
                (logging.WARNING, "warning message"),
                (logging.ERROR, "error message"),
            ]

            for level, message in level_message_pairs:
                with self.assertLogs(diagnostic.logger, level=verbosity_level):
                    diagnostic.log(level, message)

            self.assertTrue(
                any(
                    message.find(message) >= 0
                    for message in diagnostic.additional_messages
                )
            )

    @common_utils.parametrize(
        "log_api, log_level",
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
        ],
    )
    def test_diagnostic_log_is_emitted_according_to_api_level_and_diagnostic_options_verbosity_level(
        self, log_api: str, log_level: int
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            message = "log message"
            with self.assertLogs(
                diagnostic.logger, level=verbosity_level
            ) as assert_log_context:
                getattr(diagnostic, log_api)(message)
                # NOTE: self.assertNoLogs only exist >= Python 3.10
                # Add this dummy log such that we can pass self.assertLogs, and inspect
                # assert_log_context.records to check if the log level is correct.
                diagnostic.log(logging.ERROR, "dummy message")

            for record in assert_log_context.records:
                self.assertGreaterEqual(record.levelno, logging.INFO)

            if log_level >= verbosity_level:
                self.assertIn(message, diagnostic.additional_messages)
            else:
                self.assertNotIn(message, diagnostic.additional_messages)

    def test_diagnostic_log_lazy_string_is_not_evaluated_when_level_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            reference_val = 0

            def expensive_formatting_function() -> str:
                # Modify the reference_val to reflect this function is evaluated
                nonlocal reference_val
                reference_val += 1
                return f"expensive formatting {reference_val}"

            # `expensive_formatting_function` should NOT be evaluated.
            diagnostic.debug("%s", formatter.LazyString(expensive_formatting_function))
            self.assertEqual(
                reference_val,
                0,
                "expensive_formatting_function should not be evaluated after being wrapped under LazyString",
            )

    def test_diagnostic_log_lazy_string_is_evaluated_once_when_level_not_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            reference_val = 0

            def expensive_formatting_function() -> str:
                # Modify the reference_val to reflect this function is evaluated
                nonlocal reference_val
                reference_val += 1
                return f"expensive formatting {reference_val}"

            # `expensive_formatting_function` should NOT be evaluated.
            diagnostic.info("%s", formatter.LazyString(expensive_formatting_function))
            self.assertEqual(
                reference_val,
                1,
                "expensive_formatting_function should only be evaluated once after being wrapped under LazyString",
            )

    def test_diagnostic_nested_log_section_emits_messages_with_correct_section_title_indentation(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            with diagnostic.log_section(logging.INFO, "My Section"):
                diagnostic.log(logging.INFO, "My Message")
                with diagnostic.log_section(logging.INFO, "My Subsection"):
                    diagnostic.log(logging.INFO, "My Submessage")

            with diagnostic.log_section(logging.INFO, "My Section 2"):
                diagnostic.log(logging.INFO, "My Message 2")

            self.assertIn("## My Section", diagnostic.additional_messages)
            self.assertIn("### My Subsection", diagnostic.additional_messages)
            self.assertIn("## My Section 2", diagnostic.additional_messages)

    def test_diagnostic_log_source_exception_emits_exception_traceback_and_error_message(
        self,
    ):
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        with self.context:
            try:
                raise ValueError("original exception")
            except ValueError as e:
                diagnostic = infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.NOTE
                )
                diagnostic.log_source_exception(logging.ERROR, e)

            diagnostic_message = "\n".join(diagnostic.additional_messages)

            self.assertIn("ValueError: original exception", diagnostic_message)
            self.assertIn("Traceback (most recent call last):", diagnostic_message)

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
                diagnostic.log_source_exception(logging.ERROR, e)
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
