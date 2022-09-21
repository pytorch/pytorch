# Owner(s): ["module: onnx"]

import contextlib
import io
from typing import Set, Tuple

import torch
from torch.onnx import errors
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.testing._internal import common_utils


def _assert_has_diagnostics(
    engine: infra.DiagnosticEngine,
    rule_level_pairs: Set[Tuple[infra.Rule, infra.Level]],
):
    sarif_log = engine.sarif_log
    unseen_pairs = set({(rule.id, level.value) for rule, level in rule_level_pairs})
    actual_results = []
    for run in sarif_log.runs:
        for result in run.results:
            if (result.rule_id, result.level) in unseen_pairs:
                unseen_pairs.remove((result.rule_id, result.level))
            actual_results.append((result.rule_id, result.level))

    if unseen_pairs:
        raise AssertionError(
            f"Expected diagnostic results of rule id and level pair {unseen_pairs} not found. "
            f"Actual diagnostic results: {actual_results}"
        )


@contextlib.contextmanager
def assertAllDiagnostics(
    test_suite: common_utils.TestCase,
    engine: infra.DiagnosticEngine,
    rule_level_pairs: Set[Tuple[infra.Rule, infra.Level]],
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

    def test_assert_diagnostic_raise_when_diagnostic_not_found(self):
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


class TestDiagnosticsInfra(common_utils.TestCase):
    """Test cases for diagnostics infra."""

    def setUp(self):
        self.engine = infra.DiagnosticEngine()
        self.rules = [
            infra.Rule(
                "1",
                "rule-none",
                message_default_template="rule none",
            ),
            infra.Rule(
                "2",
                "rule-note",
                message_default_template="rule note",
            ),
            infra.Rule(
                "3",
                "rule-warning",
                message_default_template="rule warning",
            ),
            infra.Rule(
                "4",
                "rule-error",
                message_default_template="rule error",
            ),
        ]
        self.diagnostic_tool = infra.DiagnosticTool("test_tool", "1.0.0", self.rules)
        with contextlib.ExitStack() as stack:
            self.default_run_ctx = stack.enter_context(
                self.engine.start_new_run(self.diagnostic_tool)
            )
            self.addCleanup(stack.pop_all().close)
        return super().setUp()

    def test_diagnose_with_nonexistent_rule(self):
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

    def test_diagnostic_run(self):
        with assertDiagnostic(
            self,
            self.engine,
            self.rules[0],
            infra.Level.WARNING,
        ):
            self.engine.diagnose(self.rules[0], infra.Level.WARNING)
        self.assertEqual(len(self.engine._runs), 1)

    def test_diagnostic_nested_run(self):
        with self.engine.start_new_run(self.diagnostic_tool):
            with assertDiagnostic(
                self, self.engine, self.rules[0], infra.Level.WARNING
            ):
                self.engine.diagnose(self.rules[0], infra.Level.WARNING)
            self.assertEqual(len(self.engine._runs), 2)

    def test_diagnose_outside_run_raise_error(self):
        self.engine.end_current_run()
        with self.assertRaisesRegex(
            RuntimeError,
            "No run is currently active.",
        ):
            self.engine.diagnose(self.rules[0], infra.Level.WARNING)
