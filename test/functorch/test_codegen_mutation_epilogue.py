# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the mutation epilogue in _create_runtime_wrapper.

The codegen'd mutation epilogue emits one of as_strided_(), copy_(),
or detach().copy_() per mutated input, with the branch resolved at codegen
time from each input's mutation metadata (mutates_metadata, mutates_data,
is_leaf).

Tests that exercise data-only mutations use torch.compile (dynamo handles
metadata mutations in-graph, so only data mutations reach the epilogue).

Tests that exercise metadata mutations (metadata-only, data+metadata)
use aot_function directly so metadata mutations flow through the epilogue.

Tests inspect the generated "mutation_epilogue" source section.
"""

from common_utils import capture_codegen_source

import torch
import torch._functorch.config
from functorch.compile import nop
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestCodegenMutationEpilogue(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _capture_codegen_source(self, artifact_name):
        return capture_codegen_source(artifact_name)

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

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("copy_", captured[0])

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

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("copy_", captured[0])

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

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("detach().copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_metadata_only_mutation(self):
        """
        Metadata-only mutation via transpose_(). Codegen should emit
        as_strided_() without copy_(). Uses aot_function directly because
        dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(a, b):
                a.transpose_(1, 0)
                return a + b

            a = torch.randn(3, 4, requires_grad=True).add(0)
            b = torch.randn(4, 3)
            compiled_f = aot_function(f, nop)
            out = compiled_f(a, b)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("as_strided_", captured[0])
        self.assertNotIn("copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_data_and_metadata_mutation(self):
        """
        Both data and metadata mutated (transpose_ then mul_). Codegen
        should emit as_strided_() followed by copy_(). Uses aot_function
        directly because dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(a):
                a.transpose_(1, 0)
                a.mul_(2)
                return a + 1

            a = torch.randn(3, 4, requires_grad=True).add(0)
            a_ref = a.detach().clone()
            compiled_f = aot_function(f, nop)
            out = compiled_f(a)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(a.detach(), a_ref.transpose(1, 0) * 2)
        self.assertEqual(out, a_ref.transpose(1, 0) * 2 + 1)

        self.assertEqual(len(captured), 1)
        self.assertIn("as_strided_", captured[0])
        self.assertIn("copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_data_mutation_under_inference_mode(self):
        """
        Inference tensor input mutated under inference_mode. Inference
        tensors can only be mutated under inference_mode, so codegen should
        emit its copy_() inside torch.inference_mode().
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(x, cache):
                with torch.inference_mode():
                    cache.add_(x)
                return x + 1

            with torch.inference_mode():
                cache = torch.zeros(4)
            x = torch.randn(4)
            compiled_f = aot_function(f, nop, keep_inference_input_mutations=False)
            out = compiled_f(x, cache)

        self.assertEqual(cache, x)
        self.assertEqual(out, x + 1)

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def _apply_mutations(orig_inputs, updated_inputs):
    with torch.inference_mode():
        if orig_inputs[1].requires_grad: orig_inputs[1].detach().copy_(updated_inputs[0])
        else: orig_inputs[1].copy_(updated_inputs[0])""",
        )

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_coalesced_data_mutations_under_inference_mode(self):
        """
        Adjacent inference-mode data mutations share a single
        torch.inference_mode() block, which closes before a non-inference-mode
        mutation and re-opens for subsequent inference-mode mutations.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(x, cache_k, cache_v, buf, cache_o):
                with torch.inference_mode():
                    cache_k.add_(x)
                    cache_v.add_(x * 2)
                buf.add_(x)
                with torch.inference_mode():
                    cache_o.add_(x * 3)
                return x + 1

            with torch.inference_mode():
                cache_k = torch.zeros(4)
                cache_v = torch.zeros(4)
                cache_o = torch.zeros(4)
            buf = torch.zeros(4)
            x = torch.randn(4)
            compiled_f = aot_function(f, nop, keep_inference_input_mutations=False)
            out = compiled_f(x, cache_k, cache_v, buf, cache_o)

        self.assertEqual(cache_k, x)
        self.assertEqual(cache_v, x * 2)
        self.assertEqual(buf, x)
        self.assertEqual(cache_o, x * 3)
        self.assertEqual(out, x + 1)

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def _apply_mutations(orig_inputs, updated_inputs):
    with torch.inference_mode():
        if orig_inputs[1].requires_grad: orig_inputs[1].detach().copy_(updated_inputs[0])
        else: orig_inputs[1].copy_(updated_inputs[0])
        if orig_inputs[2].requires_grad: orig_inputs[2].detach().copy_(updated_inputs[1])
        else: orig_inputs[2].copy_(updated_inputs[1])
    if orig_inputs[3].requires_grad: orig_inputs[3].detach().copy_(updated_inputs[2])
    else: orig_inputs[3].copy_(updated_inputs[2])
    with torch.inference_mode():
        if orig_inputs[4].requires_grad: orig_inputs[4].detach().copy_(updated_inputs[3])
        else: orig_inputs[4].copy_(updated_inputs[3])""",
        )

    def test_no_mutation_no_epilogue(self):
        """
        No mutations at all. No mutation_epilogue artifact should be
        emitted.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(len(captured), 0)


if __name__ == "__main__":
    run_tests()
