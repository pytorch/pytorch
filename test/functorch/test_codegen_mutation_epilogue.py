# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the mutation epilogue in _create_runtime_wrapper.

The codegen'd mutation epilogue emits one of set_(), as_strided_(), copy_(),
or detach().copy_() per mutated input, with the branch resolved at codegen
time from each input's mutation metadata (mutates_storage_metadata,
mutates_metadata, mutates_data, is_leaf).

Note: for inference (no requires_grad), mutations are kept inside the graph
via keep_input_mutations, so the runtime epilogue is not used. These tests
use requires_grad inputs to trigger the training path where the epilogue
runs.

Tests verify that a "mutation_epilogue" artifact is emitted via
trace_structured.
"""

from _codegen_test_utils import CodegenArtifactMixin

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCodegenMutationEpilogue(CodegenArtifactMixin, TestCase):
    def test_single_data_mutation(self):
        """
        Single input data mutation via mul_. Codegen should emit a direct
        copy_() for this input.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                x.mul_(2)
                return x + y

            x = torch.randn(4, requires_grad=True).clone()
            x.retain_grad()
            y = torch.randn(4)
            x_ref = x.detach().clone()
            y_ref = y.clone()
            out = f(x, y)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + y_ref)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )

    def test_multiple_data_mutations(self):
        """
        Multiple inputs mutated. Codegen should emit a copy_() per mutated
        input, with non-mutated inputs skipped entirely.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b, c):
                a.mul_(2)
                c.add_(1)
                return a + b + c

            a = torch.randn(4, requires_grad=True).clone()
            a.retain_grad()
            b = torch.randn(4)
            c = torch.randn(4, requires_grad=True).clone()
            c.retain_grad()
            a_ref, c_ref = a.detach().clone(), c.detach().clone()
            out = f(a, b, c)

        self.assertEqual(a.detach(), a_ref * 2)
        self.assertEqual(c.detach(), c_ref + 1)
        self.assertEqual(out, a_ref * 2 + b + c_ref + 1)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )

    def test_leaf_mutation_under_no_grad(self):
        """
        Leaf tensor mutated under no_grad (e.g. via detach().mul_()).
        Codegen should emit detach().copy_() for this case.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.detach().mul_(2)
                return x + 1

            x = torch.randn(4, requires_grad=True)
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + 1)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
