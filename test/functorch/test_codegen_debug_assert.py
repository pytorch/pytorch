# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing DebugAssertWrapper.

The codegen'd assertion function emits checks only for the specific arg
indices where requires_grad was False at compile time. Positions where
requires_grad=True are safe and generate no runtime check, replacing
the closure that iterated over all args.

Enabled via torch._functorch.config.debug_assert = True.

Tests verify that a "debug_assert_wrapper" artifact is emitted via
trace_structured.
"""

from _codegen_test_utils import CodegenArtifactMixin

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCodegenDebugAssert(CodegenArtifactMixin, TestCase):
    @torch._functorch.config.patch(debug_assert=True)
    def test_mixed_requires_grad(self):
        """
        With debug_assert=True, the wrapper should codegen assertions for
        inputs compiled without requires_grad, and skip those with
        requires_grad=True.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)

        self.assertEqual(out, x + y)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected debug_assert_wrapper codegen artifact to be emitted",
        )

    @torch._functorch.config.patch(debug_assert=True)
    def test_all_requires_grad(self):
        """
        All inputs with requires_grad=True. Codegen should emit no
        assertions (all positions are safe), but the artifact should
        still be emitted.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y, z):
                return x + y + z

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4, requires_grad=True)
            z = torch.randn(4, requires_grad=True)
            out = f(x, y, z)

        self.assertEqual(out, x + y + z)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected debug_assert_wrapper codegen artifact to be emitted",
        )

    @torch._functorch.config.patch(debug_assert=True)
    def test_some_no_grad_inputs(self):
        """
        Mix of requires_grad and non-requires_grad inputs going through
        the training path. Codegen should emit assertions only for the
        non-grad positions.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y, z):
                return x * y + z

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            z = torch.randn(4)
            out = f(x, y, z)

        self.assertEqual(out, x * y + z)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected debug_assert_wrapper codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
