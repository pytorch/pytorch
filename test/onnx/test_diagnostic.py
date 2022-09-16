# Owner(s): ["module: onnx"]

import functools
import io

import torch
from torch.onnx import diagnostic, errors, sarif_om
from torch.testing._internal import common_utils


def _assert_has_result(sarif_log: sarif_om.SarifLog, rule_id: str, level: str):
    for run in sarif_log.runs:
        for result in run.results:
            if result.rule_id == rule_id and result.level == level:
                return
    raise AssertionError(
        f"Expected diagnostic result with rule_id={rule_id} and level={level} not found."
    )


def _assert_has_diagnostic(
    engine: diagnostic.DiagnosticEngine,
    rule: diagnostic.Rule,
    level: diagnostic.Level,
):
    _assert_has_result(engine.sarif_log, rule.id, level.value)


def assert_diagnostic(rule: diagnostic.Rule, level: diagnostic.Level):
    # TODO: This is a decorator for test cases to assert that a diagnostic is emitted.
    #       We might want to stick to the 'assertDiagnostic' context below instead.
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                fn(self, *args, **kwargs)
            except errors.OnnxExporterError:
                self.assertTrue(level == diagnostic.Level.ERROR)
            _assert_has_diagnostic(diagnostic.engine, rule, level)

        return wrapper

    return decorator


class _AssertDiagnosticContext:
    def __init__(
        self,
        test_suite: common_utils.TestCase,
        rule: diagnostic.Rule,
        level: diagnostic.Level,
    ):
        self.test_suite = test_suite
        self.rule = rule
        self.level = level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            if isinstance(exc_value, errors.OnnxExporterError):
                self.test_suite.assertTrue(self.level == diagnostic.Level.ERROR)
                return True
            else:
                return False
        _assert_has_diagnostic(diagnostic.engine, self.rule, self.level)

        return True


def assertDiagnostic(
    test_suite: common_utils.TestCase,
    rule: diagnostic.Rule,
    level: diagnostic.Level,
):
    """
    Context manager to assert that a diagnostic is emitted.

    Usage:
        with assertDiagnostic(
            self,
            diagnostic.Rules.OperatorSupportedInNewerOpsetVersion,
            diagnostic.Level.ERROR
        ):
            torch.onnx.export(...)

    Args:
        test_suite: The test suite instance.
        rule: The diagnostic rule.
        level: The diagnostic level.

    Returns:
        A context manager.

    Raises:
        AssertionError: If the diagnostic is not emitted.
    """
    return _AssertDiagnosticContext(test_suite, rule, level)


class TestONNXDiagnostic(common_utils.TestCase):
    def setUp(self):
        engine = diagnostic.engine
        engine.reset()
        return super().setUp()

    def test_assert_diagnostic_raise_when_diagnostic_not_found(self):
        with self.assertRaises(AssertionError):
            with assertDiagnostic(
                self,
                diagnostic.rules.ONNXShapeInferenceIsMissingForNode,
                diagnostic.Level.WARNING,
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
            diagnostic.rules.ONNXShapeInferenceIsMissingForNode,
            diagnostic.Level.WARNING,
        ):
            # trigger warning for missing shape inference.
            torch.onnx.export(M(), torch.randn(3, 4), io.BytesIO())

    def test_py_diagnose_emits_error(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x)

        with assertDiagnostic(
            self,
            diagnostic.rules.OperatorSupportedInNewerOpsetVersion,
            diagnostic.Level.ERROR,
        ):
            # trigger error for operator unsupported until newer opset version.
            torch.onnx.export(
                M(),
                torch.randn(3, 4),
                io.BytesIO(),
                opset_version=9,
            )

    def test_diagnostic_runs(self):
        engine = diagnostic.engine
        # TODO: Test coverage for sarif run properties.
        pass
