# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the output alias regeneration in
_create_runtime_wrapper.

The codegen'd output alias handler inlines each handler type's logic per
output as straight-line code: NoopAliasHandler becomes a direct fw_outs[i]
reference, IsInputHandler becomes orig_inputs[base_idx], and
AliasOfInput/IntermediateHandler become inline gen_alias_from_base calls
with baked-in indices and metadata.

Tests verify that an "output_alias_wrapper" artifact is emitted via
trace_structured.
"""

from _codegen_test_utils import CodegenArtifactMixin

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCodegenOutputAlias(CodegenArtifactMixin, TestCase):
    def test_output_is_view_of_input(self):
        """
        Output that is a view of an input (alias_of_input). Codegen should
        emit gen_alias_from_base(orig_inputs[i], ...) inline.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1)

            x = torch.randn(2, 3)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_is_input(self):
        """
        Output that IS the input (is_input). Codegen should emit a direct
        reference to orig_inputs[i].
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.mul_(2)
                return x

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertIs(out, x)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_mixed_alias_and_non_alias_outputs(self):
        """
        Multiple outputs: one aliased, one not. Codegen should emit
        gen_alias_from_base for the alias and a noop for the non-alias.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_alias_with_mutation(self):
        """
        Input is mutated AND output is a view of the input. Codegen should
        handle both mutation epilogue and alias regeneration.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.view(-1)

            x = torch.randn(2, 3)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
