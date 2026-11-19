# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer and pad-as-cat optimizations."""

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# required so that metrics.num_bytes_accessed is populated
torch._logging.set_logs(inductor_metrics=True)


class TestCatMultiConsumer(TestCase):
    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_other_cat_input_is_not_external_consumer(self):
        """A use feeding another input of the same cat is internal."""

        def fn(x):
            rotated = torch.sin(x)
            concatenated = torch.cat((rotated, -rotated), dim=-1)
            return torch.argmax(concatenated, dim=-1)

        x = torch.randn(128, 64, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)

        self.assertEqual(result, fn(x))
        self.assertNotIn("reinterpret_tensor(", code)

    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_external_consumer_still_uses_concat_kernel(self):
        """A consumer outside the cat inputs remains a multi-consumer."""

        def fn(x):
            rotated = torch.sin(x)
            concatenated = torch.cat((rotated, -rotated), dim=-1)
            return torch.argmax(concatenated, dim=-1), torch.cos(rotated)

        x = torch.randn(128, 64, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])
        self.assertIn("reinterpret_tensor(", code)

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


class TestGroupedPointwiseCat(TestCase):
    @requires_gpu()
    def test_pad_cat_grouped_exact_kernels_and_single_alloc(self):
        def fn_pad(*inputs):
            return torch.cat([F.pad(x, (0, 32)) for x in inputs], dim=0)

        for shape in ((32, 128), (512, 1024)):
            inputs = tuple(torch.randn(*shape, device=GPU_TYPE) for _ in range(16))
            ref = fn_pad(*inputs)

            with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
                act, (code_grouped,) = run_and_get_code(
                    torch.compile(fn_pad, fullgraph=True), *inputs
                )
            self.assertEqual(act, ref)
            self.assertEqual(code_grouped.count(".run("), 2)
            self.assertEqual(
                code_grouped.split("def call(")[-1].count(" = empty_strided_"), 1
            )

            with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=False):
                act_unfused, (code_unfused,) = run_and_get_code(
                    torch.compile(fn_pad, fullgraph=True), *inputs
                )
            self.assertEqual(act_unfused, ref)
            self.assertEqual(code_unfused.count(".run("), 16)

            with (
                fresh_cache(),
                inductor_config.patch(grouped_pointwise_cat=True, combo_kernels=True),
            ):
                act_combo, (code_combo,) = run_and_get_code(
                    torch.compile(fn_pad, fullgraph=True), *inputs
                )
            self.assertEqual(act_combo, ref)
            self.assertEqual(
                code_combo.split("def call(")[-1].count(" = empty_strided_"), 1
            )

    @requires_gpu()
    def test_pad_cat_with_following_pointwise(self):
        def fn(*inputs):
            return torch.cat([F.pad(x.t(), (0, 16)) for x in inputs], dim=0) * 2.0 + 1.0

        inputs = tuple(torch.randn(64, 32, device=GPU_TYPE) for _ in range(16))
        ref = fn(*inputs)
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act, (code_grouped,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), *inputs
            )
        self.assertEqual(act, ref)
        self.assertEqual(code_grouped.count(".run("), 3)

        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=False):
            _, (code_unfused,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), *inputs
            )
        self.assertEqual(code_unfused.count(".run("), 17)

    @requires_gpu()
    def test_compute_budget_scales_width_and_guards_large_compute(self):
        # 1. Small compute (16 x 64 x 128 -> ~0.25 MiB per 8-input group <= 4 MiB) -> 2 groups of 8.
        def fn_sin_add(*inputs):
            return torch.cat([x.sin() + 1.0 for x in inputs], dim=0)

        small_inps = tuple(torch.randn(64, 128, device=GPU_TYPE) for _ in range(16))
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_small, (code_small,) = run_and_get_code(
                torch.compile(fn_sin_add, fullgraph=True), *small_inps
            )
        self.assertEqual(act_small, fn_sin_add(*small_inps))
        self.assertEqual(code_small.count(".run("), 2)

        # 2. Medium compute (16 x 512 x 512 fp32 = 1 MiB/input -> 4 MiB budget caps width at 4) -> 4 kernels.
        def fn_gelu(*inputs):
            return torch.cat([F.gelu(x) for x in inputs], dim=0)

        med_inps = tuple(
            torch.randn(512, 512, device=GPU_TYPE, dtype=torch.float32)
            for _ in range(16)
        )
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_med, (code_med,) = run_and_get_code(
                torch.compile(fn_gelu, fullgraph=True), *med_inps
            )
        self.assertEqual(act_med, fn_gelu(*med_inps))
        self.assertEqual(code_med.count(".run("), 4)

        # 3. Large compute (16 x 1024 x 2048 fp32 = 8 MiB/input > 4 MiB budget) -> stays ungrouped (16 kernels).
        def fn_sin(*inputs):
            return torch.cat([x.sin() for x in inputs], dim=0)

        large_inps = tuple(
            torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.float32)
            for _ in range(16)
        )
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_large, (code_large,) = run_and_get_code(
                torch.compile(fn_sin, fullgraph=True), *large_inps
            )
        self.assertEqual(act_large, fn_sin(*large_inps))
        self.assertEqual(code_large.count(".run("), 16)

        # 4. Mixed large pad (8 MiB) followed by small compute (0.03 MiB) must not contaminate the large pad group.
        def fn_mixed_size(big_a, big_b, small_c, small_d, *more_pads):
            return torch.cat(
                [
                    F.pad(big_a, (0, 32)),
                    F.pad(big_b, (0, 32)),
                    small_c.sin(),
                    small_d.sin(),
                ]
                + [F.pad(p, (0, 32)) for p in more_pads],
                dim=0,
            )

        big_a = torch.randn(512, 4064, device=GPU_TYPE, dtype=torch.float32)
        big_b = torch.randn(512, 4064, device=GPU_TYPE, dtype=torch.float32)
        small_c = torch.randn(2, 4096, device=GPU_TYPE, dtype=torch.float32)
        small_d = torch.randn(2, 4096, device=GPU_TYPE, dtype=torch.float32)
        more_pads = tuple(
            torch.randn(4, 4064, device=GPU_TYPE, dtype=torch.float32) for _ in range(8)
        )
        all_args = (big_a, big_b, small_c, small_d, *more_pads)
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_mix, (code_mix,) = run_and_get_code(
                torch.compile(fn_mixed_size, fullgraph=True), *all_args
            )
        self.assertEqual(act_mix, fn_mixed_size(*all_args))
        self.assertEqual(code_mix.count(".run("), 3)

    @requires_gpu()
    def test_duplicate_inputs_and_skipped_empty_1d(self):
        def fn(empty_1d, *xs):
            pads = [F.pad(x, (0, 16)) for x in xs]
            return torch.cat([*pads[:5], empty_1d, *pads[5:], pads[0]], dim=0)

        empty_1d = torch.empty(0, device=GPU_TYPE)
        xs = tuple(torch.randn(64, device=GPU_TYPE) for _ in range(9))
        ref = fn(empty_1d, *xs)
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), empty_1d, *xs
            )
        self.assertEqual(act, ref)
        self.assertEqual(code.count(".run("), 2)

    @requires_gpu()
    def test_foreach_raw_inputs_and_mixed_guards(self):
        def fn_raw(*raws):
            return torch.cat(raws, dim=0)

        raws = tuple(torch.randn(1, 32, 32, device=GPU_TYPE) for _ in range(16))
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_raw, (code_raw,) = run_and_get_code(
                torch.compile(fn_raw, fullgraph=True), *raws
            )
        self.assertEqual(act_raw, fn_raw(*raws))
        self.assertEqual(code_raw.count(".run("), 1)
        self.assertIn("triton_for_fused", code_raw)

        def fn_mixed(x, *rest):
            ext = F.pad(rest[0], (0, 16))
            red = x.sum(dim=-1, keepdim=True).expand_as(ext)
            cat_out = torch.cat(
                [red, ext] + [F.pad(t, (0, 16)) for t in rest[1:]], dim=0
            )
            return cat_out, ext + 3.0

        mix_x = torch.randn(4, 32, device=GPU_TYPE)
        mix_rest = (torch.randn(4, 16, device=GPU_TYPE),) + tuple(
            torch.randn(4, 16, device=GPU_TYPE) for _ in range(9)
        )
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_mix = torch.compile(fn_mixed, fullgraph=True)(mix_x, *mix_rest)
        ref_mix = fn_mixed(mix_x, *mix_rest)
        self.assertEqual(act_mix[0], ref_mix[0])
        self.assertEqual(act_mix[1], ref_mix[1])

        def fn_nine(*xs):
            return torch.cat([F.pad(x, (0, 32)) for x in xs], dim=0)

        nine = tuple(torch.randn(8, 64, device=GPU_TYPE) for _ in range(9))
        with (
            fresh_cache(),
            inductor_config.patch(
                grouped_pointwise_cat=True,
                max_pointwise_cat_inputs=4,
                max_complex_pointwise_cat_inputs=4,
            ),
        ):
            act_nine, (code_nine,) = run_and_get_code(
                torch.compile(fn_nine, fullgraph=True), *nine
            )
        self.assertEqual(act_nine, fn_nine(*nine))
        self.assertEqual(code_nine.count(".run("), 3)

    @requires_gpu()
    def test_dynamic_shape_compute_stays_safe_and_pad_groups(self):
        def fn_pad(*xs):
            return torch.cat([F.pad(x, (0, 16)) for x in xs], dim=0)

        def fn_sin(*xs):
            return torch.cat([x.sin() for x in xs], dim=0)

        xs = tuple(torch.randn(16, 32, device=GPU_TYPE) for _ in range(10))
        with fresh_cache(), inductor_config.patch(grouped_pointwise_cat=True):
            act_pad, (code_pad,) = run_and_get_code(
                torch.compile(fn_pad, fullgraph=True, dynamic=True), *xs
            )
            act_sin, (code_sin,) = run_and_get_code(
                torch.compile(fn_sin, fullgraph=True, dynamic=True), *xs
            )
        self.assertEqual(act_pad, fn_pad(*xs))
        self.assertEqual(code_pad.count(".run("), 2)
        self.assertEqual(act_sin, fn_sin(*xs))
        self.assertEqual(code_sin.count(".run("), 10)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
