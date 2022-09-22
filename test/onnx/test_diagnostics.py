# Owner(s): ["module: onnx"]

import contextlib
import dataclasses
import io
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
    sarif_log = engine.sarif_log
    unseen_pairs = {(rule.id, level.value) for rule, level in rule_level_pairs}
    actual_results = []
    for run in sarif_log.runs:
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
def assertAllDiagnostics(
    test_suite: common_utils.TestCase,
    engine: infra.DiagnosticEngine,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    """
    Context manager to assert that all diagnostics are emitted.

    Usage:
        with assertAllDiagnostics(
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


def assertDiagnostic(
    test_suite: common_utils.TestCase,
    engine: infra.DiagnosticEngine,
    rule: infra.Rule,
    level: infra.Level,
):
    """
    Context manager to assert that a diagnostic is emitted.

    Usage:
        with assertDiagnostic(
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

    return assertAllDiagnostics(test_suite, engine, {(rule, level)})


class TestONNXDiagnostics(common_utils.TestCase):
    """Test cases for diagnostics emitted by the ONNX export code."""

    def setUp(self):
        engine = diagnostics.engine
        engine.reset()
        return super().setUp()

    def test_assert_diagnostic_raises_when_diagnostic_not_found(self):
        with self.assertRaises(AssertionError):
            with assertDiagnostic(
                self,
                diagnostics.engine,
                diagnostics.rules.node_missing_onnx_shape_inference,
                diagnostics.levels.WARNING,
            ):
                pass

    def test_cpp_diagnose_emits_warning(self):
        class CustomAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x + y

            @staticmethod
            def symbolic(g, x, y):
                return g.op("custom::CustomAdd", x, y)

        class M(torch.nn.Module):
            def forward(self, x):
                return CustomAdd.apply(x, x)

        with assertDiagnostic(
            self,
            diagnostics.engine,
            diagnostics.rules.node_missing_onnx_shape_inference,
            diagnostics.levels.WARNING,
        ):
            # trigger warning for missing shape inference.
            torch.onnx.export(M(), torch.randn(3, 4), io.BytesIO())

    def test_py_diagnose_emits_error(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x)

        with assertDiagnostic(
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


@dataclasses.dataclass()
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
        self.diagnostic_tool = infra.DiagnosticTool("test_tool", "1.0.0", self.rules)
        with contextlib.ExitStack() as stack:
            self.default_run_ctx = stack.enter_context(
                self.engine.start_new_run(self.diagnostic_tool)
            )
            self.addCleanup(stack.pop_all().close)
        return super().setUp()

    def test_diagnose_raises_value_error_with_nonexistent_rule(self):
        rule_id = "0"
        rule_name = "nonexistent-rule"
        with self.assertRaisesRegex(
            ValueError,
            f"Rule '{rule_id}:{rule_name}' is not supported by this tool "
            f"'{self.diagnostic_tool.name} {self.diagnostic_tool.version}'.",
        ):
            self.engine.diagnose(
                infra.Rule(id=rule_id, name=rule_name, message_default_template=""),
                infra.Level.WARNING,
            )

    def test_diagnostic_records_nested_runs(self):
        with self.engine.start_new_run(self.diagnostic_tool):
            self.engine.diagnose(
                self.rules.rule_without_message_args, infra.Level.WARNING
            )
            sarif_log = self.engine.sarif_log
            self.assertEqual(len(sarif_log.runs), 2)
            self.assertEqual(len(sarif_log.runs[0].results), 0)
            self.assertEqual(len(sarif_log.runs[1].results), 1)
        self.engine.diagnose(self.rules.rule_without_message_args, infra.Level.ERROR)
        sarif_log = self.engine.sarif_log
        self.assertEqual(len(sarif_log.runs), 2)
        self.assertEqual(len(sarif_log.runs[0].results), 1)
        self.assertEqual(len(sarif_log.runs[1].results), 1)

    def test_diagnose_raises_runtime_error_when_outside_of_run(self):
        self.engine.end_current_run()
        with self.assertRaisesRegex(
            RuntimeError,
            "No run is currently active.",
        ):
            self.engine.diagnose(
                self.rules.rule_without_message_args, infra.Level.WARNING
            )
