# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer and pad-as-cat optimizations."""

import torch
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# required so that metrics.num_bytes_accessed is populated
torch._logging.set_logs(inductor_metrics=True)


class TestCatMultiConsumer(TestCase):
    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_cat_to_fp16(self):
        """Multi-consumer cat avoids duplicate computation."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])

        # Without the optimization x would be read twice (once by cat, once
        # by to_fp16). With the optimization ConcatKernel shares x so it is
        # read only once.
        z = ref[0]
        y = ref[1]
        x_bytes = x.nelement() * x.element_size()
        z_bytes = z.nelement() * z.element_size()
        y_bytes = y.nelement() * y.element_size()
        unoptimized_bytes = 2 * x_bytes + z_bytes + y_bytes
        self.assertLess(
            metrics.num_bytes_accessed,
            unoptimized_bytes,
            "Optimization should avoid reading x twice.",
        )

    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        """Single-consumer cat unchanged."""

        def fn(x):
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result, ref)

        # Single-consumer cat should use pointwise_cat which fuses the
        # zeros fill into the cat kernel.  Total bytes: read x + write z.
        x_bytes = x.nelement() * x.element_size()
        z_bytes = ref.nelement() * ref.element_size()
        expected_bytes = x_bytes + z_bytes
        self.assertEqual(
            metrics.num_bytes_accessed,
            expected_bytes,
            lambda msg: f"{msg}\nExpected {expected_bytes} bytes, got {metrics.num_bytes_accessed}.",
        )


def _high_arity_inputs(batch, width=3):
    return [
        torch.randn(batch, width + index % 5, device=GPU_TYPE) for index in range(18)
    ]


def _tanh_cat(*inputs):
    return torch.cat([x.tanh() for x in inputs], dim=1)


class TestChunkedPointwiseCat(TestCase):
    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_high_arity_simple_cat(self):
        compiled = torch.compile(_tanh_cat, dynamic=True)
        inputs = _high_arity_inputs(32)
        metrics.reset()
        result = compiled(*inputs)

        self.assertEqual(result, _tanh_cat(*inputs))
        self.assertEqual(metrics.generated_kernel_count, 3)

        dynamic_inputs = _high_arity_inputs(47)
        self.assertEqual(compiled(*dynamic_inputs), _tanh_cat(*dynamic_inputs))

    def _kernel_count(self, fn, width=3, **config_patches):
        inputs = _high_arity_inputs(32, width)
        torch._dynamo.reset()
        metrics.reset()
        with torch._inductor.config.patch(fx_graph_cache=False, **config_patches):
            self.assertEqual(torch.compile(fn)(*inputs), fn(*inputs))
        return metrics.generated_kernel_count

    @requires_gpu()
    def test_pointwise_cat_chunk_size(self):
        self.assertEqual(self._kernel_count(_tanh_cat, pointwise_cat_chunk_size=6), 3)
        self.assertEqual(self._kernel_count(_tanh_cat, pointwise_cat_chunk_size=18), 1)
        self.assertEqual(
            self._kernel_count(_tanh_cat, pointwise_cat_chunk_size=1),
            self._kernel_count(_tanh_cat, max_pointwise_cat_inputs=1),
        )

    @requires_gpu()
    def test_high_arity_realized_inputs_keep_concat_kernel(self):
        def fn(*inputs):
            return torch.cat([inputs[0].tanh(), *inputs[1:]], dim=1)

        self.assertEqual(
            self._kernel_count(fn),
            self._kernel_count(fn, pointwise_cat_chunk_size=1),
        )

    @requires_gpu()
    def test_high_arity_reduction_inputs_keep_concat_kernel(self):
        def fn(*inputs):
            return torch.cat([x.sum(1, keepdim=True).tanh() for x in inputs], dim=1)

        # Width 64 keeps each sum a real reduction instead of an unrolled pointwise op.
        self.assertEqual(
            self._kernel_count(fn, width=64),
            self._kernel_count(fn, width=64, pointwise_cat_chunk_size=1),
        )


class TestPadAsCat(TestCase):
    @requires_gpu()
    def test_mul_pad_addmm(self):
        """Multi-consumer F.pad uses ConcatKernel zero-copy."""
        counters.clear()

        def fn(x, scale, bias, weight):
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1], atol=1e-2, rtol=1e-2)
        self.assertIn("reinterpret_tensor", code)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)

    @requires_gpu()
    def test_single_consumer_pad(self):
        """Single-consumer F.pad is decomposed into cat, which fuses via pointwise_cat."""
        counters.clear()

        def fn(x, scale):
            return torch.nn.functional.pad(x * scale, [0, 192])

        x = torch.randn(128, 2880, device=GPU_TYPE)
        scale = torch.randn(128, 2880, device=GPU_TYPE)

        compiled = torch.compile(fn)
        result = compiled(x, scale)
        ref = fn(x, scale)

        self.assertEqual(result, ref)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
