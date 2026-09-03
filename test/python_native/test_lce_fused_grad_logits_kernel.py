# Owner(s): ["module: dsl-native-ops"]
#
# Numerics for the CuTeDSL kernel that computes the row statistics and the
# softmax-gradient transform together, used by the `fused` variant of the
# chunked linear_cross_entropy overrides. Checked on its own, against a torch
# expression of the same contract, before anything routes through it.

import unittest

import torch
from torch._native import cutedsl_utils as cu
from torch.testing._internal.common_cuda import SM80OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _reference(logits, row_scale, target, out_dtype):
    # The kernel widens every element it loads, so the reference reads the
    # same values in fp32: what is compared is the kernel, not the buffer's
    # rounding.
    logits = logits.float()
    row_max = logits.amax(dim=1, keepdim=True)
    e = (logits - row_max).exp()
    row_sum = e.sum(dim=1)
    g = e * (row_scale / row_sum).unsqueeze(1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    g[rows, target] -= row_scale
    return (
        g.to(out_dtype),
        row_max.squeeze(1) + row_sum.log(),
        logits[rows, target],
    )


def _outputs(num_rows, V, dtype, device="cuda", row_stride=None):
    if row_stride is None:
        g = torch.empty(num_rows, V, device=device, dtype=dtype)
    else:
        # A wider row stride than V: the kernel indexes through the tensor's
        # own strides, so this must work as well as the packed case.
        g = torch.empty(num_rows, row_stride, device=device, dtype=dtype)[:, :V]
    lse = torch.empty(num_rows, device=device, dtype=torch.float32)
    target_logit = torch.empty(num_rows, device=device, dtype=torch.float32)
    return g, lse, target_logit


def _inputs(num_rows, V, device="cuda", logits_dtype=torch.float32):
    logits = torch.randn(num_rows, V, device=device, dtype=torch.float32).to(
        logits_dtype
    )
    row_scale = torch.rand(num_rows, device=device, dtype=torch.float32) + 0.5
    target = torch.randint(0, V, (num_rows,), device=device)
    return logits, row_scale, target


# `skipIfNoCuteDSL` and `runtime_available()` only say the package is
# installed. These tests call the kernel entry points DIRECTLY, with no `cond`
# in front to decline an unsupported device, so they need the DSL's executable
# floor -- documented as Ampere and later. (The kernel's own eligibility is a
# validated-set membership instead, since capability is not an ordering.)
@unittest.skipIf(not TEST_CUDA, "CuTeDSL kernels are CUDA-only")
@unittest.skipIf(not SM80OrLater, "the CuTeDSL runtime requires sm_80 or later")
class TestFusedGradLogitsKernel(TestCase):
    def setUp(self):
        super().setUp()
        if not cu.runtime_available() or cu.check_native_jit_disabled():
            self.skipTest("CuTeDSL runtime unavailable or native DSL disabled")
        from torch._native.ops.linear_cross_entropy import fused_grad_logits_kernel

        self.kernel = fused_grad_logits_kernel

    def _run(self, logits, row_scale, target, dtype, row_stride=None):
        g, lse, target_logit = _outputs(*logits.shape, dtype, row_stride=row_stride)
        self.kernel.fused_grad_logits_into(
            g, lse, target_logit, logits, row_scale, target
        )
        return g, lse, target_logit

    def _check(self, logits, row_scale, target, dtype, row_stride=None):
        g, lse, target_logit = self._run(
            logits, row_scale, target, dtype, row_stride=row_stride
        )
        want_g, want_lse, want_target_logit = _reference(
            logits, row_scale, target, dtype
        )
        if dtype is torch.float32:
            # The kernel's exponential is the hardware approximation and its
            # row sum reduces in a different order than torch's.
            self.assertEqual(g, want_g, atol=1e-6, rtol=1e-5)
        else:
            self.assertEqual(g, want_g)
        self.assertEqual(lse, want_lse, atol=1e-4, rtol=1e-5)
        # A copy, not a computation.
        self.assertEqual(target_logit, want_target_logit)

    @parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @parametrize(
        "num_rows, V",
        [
            (1, 7),  # V far below the block width: most threads see nothing
            (4, 512),  # exactly the block width
            (4, 513),  # one column past it
            (37, 4097),  # neither dimension a multiple of the block width
            (128, 32000),  # a realistic chunk
        ],
    )
    def test_matches_reference(self, dtype, num_rows, V):
        logits, row_scale, target = _inputs(num_rows, V)
        self._check(logits, row_scale, target, dtype)

    @parametrize("logits_dtype", [torch.bfloat16, torch.float16])
    @parametrize("num_rows, V", [(4, 513), (128, 32000)])
    def test_low_precision_logits_buffer(self, logits_dtype, num_rows, V):
        """The buffer dtype is a compile-key axis: the kernel widens each
        element on load, so the result must match the same values read in
        fp32."""
        logits, row_scale, target = _inputs(num_rows, V, logits_dtype=logits_dtype)
        self._check(logits, row_scale, target, torch.bfloat16)

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "num_rows, V",
        [
            (1, 7),  # below one tile
            (4, 513),  # crosses the block width
            (8, 2049),  # one column past a full staging group
            (37, 4097),
            (128, 32000),  # a realistic chunk, many staging groups
        ],
    )
    def test_aliased_g_shares_the_logits_storage(self, dtype, num_rows, V):
        """`g` written into the logits' own bytes, which is what makes a chunk
        cost one buffer instead of two. `g[n, j]` occupies `z[n, j // 2]`, so
        the kernel must order its writes against its reads; a stale read shows
        up as a wrong value here. An element left unwritten would hold
        reinterpreted fp32 bytes, so matching the reference everywhere also
        proves full coverage -- the aliased buffer is its own sentinel.

        The kernel takes `g`'s layout from the caller and orders its writes
        either way, so this and the separate-buffer tests above exercise one
        compiled mode."""
        logits, row_scale, target = _inputs(num_rows, V)
        source = logits.clone()
        g, lse, target_logit = _outputs(num_rows, V, dtype)
        g = logits.view(dtype).narrow(1, 0, V)
        self.kernel.fused_grad_logits_into(
            g, lse, target_logit, logits, row_scale, target
        )
        want_g, want_lse, want_target_logit = _reference(
            source, row_scale, target, dtype
        )
        self.assertEqual(g, want_g)
        self.assertEqual(lse, want_lse, atol=1e-4, rtol=1e-5)
        # Read before any write could occupy its bytes.
        self.assertEqual(target_logit, want_target_logit)

    def test_monotonic_rows_rescale_every_element(self):
        """Worst case for the online statistics: each thread walks its columns
        in increasing order, so every element raises its running maximum and
        rescales its running sum."""
        num_rows, V = 8, 4096
        logits = (
            torch.arange(V, device="cuda", dtype=torch.float32)
            .div(64.0)
            .expand(num_rows, V)
            .contiguous()
        )
        _, row_scale, target = _inputs(num_rows, V)
        self._check(logits, row_scale, target, torch.float32)

    def test_large_magnitudes_do_not_overflow(self):
        """The row maximum is what keeps the exponentials finite; without the
        shift these logits would give inf."""
        num_rows, V = 8, 1024
        logits, row_scale, target = _inputs(num_rows, V)
        logits = logits + 10000.0
        g, lse, _ = self._run(logits, row_scale, target, torch.float32)
        self.assertTrue(torch.isfinite(g).all())
        self.assertEqual(lse, _reference(logits, row_scale, target, torch.float32)[1])

    def test_uniform_row_sums_to_the_class_count(self):
        num_rows, V = 4, 2048
        logits = torch.full((num_rows, V), 3.5, device="cuda", dtype=torch.float32)
        _, row_scale, target = _inputs(num_rows, V)
        _, lse, target_logit = self._run(logits, row_scale, target, torch.float32)
        want = torch.full_like(lse, 3.5 + torch.tensor(float(V)).log().item())
        self.assertEqual(lse, want, atol=1e-4, rtol=1e-6)
        self.assertEqual(target_logit, torch.full_like(target_logit, 3.5))

    def test_target_column_is_the_shifted_probability(self):
        """The one-hot subtract is what replaces eager's index_add_, so the
        target column is checked explicitly rather than only in aggregate."""
        num_rows, V = 16, 512
        logits, row_scale, target = _inputs(num_rows, V)
        g, lse, _ = self._run(logits, row_scale, target, torch.float32)
        rows = torch.arange(num_rows, device=g.device)
        p_target = (logits[rows, target] - lse).exp()
        self.assertEqual(
            g[rows, target], (p_target - 1.0) * row_scale, atol=1e-6, rtol=1e-5
        )
        # Rows sum to ~0 for unit scale: sum_v p = 1 and one 1.0 is removed.
        self.assertEqual(
            (g / row_scale.unsqueeze(1)).sum(dim=1),
            torch.zeros(num_rows, device=g.device),
            atol=1e-4,
            rtol=0,
        )

    def test_zero_row_scale_gives_a_zero_gradient(self):
        """An ignored row carries scale 0, and its whole gradient row -- target
        column included -- must be exactly zero."""
        num_rows, V = 6, 300
        logits, row_scale, target = _inputs(num_rows, V)
        row_scale = torch.zeros_like(row_scale)
        g, _, _ = self._run(logits, row_scale, target, torch.bfloat16)
        self.assertEqual(g, torch.zeros_like(g))

    def test_wider_row_stride(self):
        logits, row_scale, target = _inputs(24, 100)
        self._check(logits, row_scale, target, torch.bfloat16, row_stride=128)

    @parametrize(
        "meta",
        [
            {},
            {"threads_per_block": 128},
            {"tiles_per_stage": 1},
            {"threads_per_block": 1024, "tiles_per_stage": 8},
        ],
    )
    def test_shape_knobs_do_not_change_the_result(self, meta):
        """`threads_per_block` and `tiles_per_stage` pick the block width and how
        many column tiles pass 2 stages per barrier. They exist for tuning, so
        every legal combination has to compute the same thing -- including the
        write ordering, whose safety argument holds for any staging depth."""
        logits, row_scale, target = _inputs(24, 4097)
        g, lse, target_logit = _outputs(24, 4097, torch.bfloat16)
        self.kernel.fused_grad_logits_into(
            g, lse, target_logit, logits, row_scale, target, **meta
        )
        want_g, want_lse, _ = _reference(logits, row_scale, target, torch.bfloat16)
        self.assertEqual(g, want_g)
        self.assertEqual(lse, want_lse, atol=1e-4, rtol=1e-5)

    @parametrize(
        "meta, message",
        [
            ({"threads_per_block": 100}, "multiple of 32"),
            ({"threads_per_block": 2048}, "multiple of 32"),
            ({"tiles_per_stage": 0}, "at least 1"),
            ({"tiles_pre_stage": 4}, "unknown meta parameters"),
        ],
    )
    def test_illegal_shape_knobs_raise(self, meta, message):
        """A typo or an out-of-range value must fail loudly rather than compile
        something that silently drops per-warp partials (above 32 warps) or
        never runs (a zero-tile stage)."""
        logits, row_scale, target = _inputs(4, 64)
        g, lse, target_logit = _outputs(4, 64, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, message):
            self.kernel.fused_grad_logits_into(
                g, lse, target_logit, logits, row_scale, target, **meta
            )

    def test_logits_are_not_modified(self):
        """The buffer is read twice and never shifted in place, which is what
        lets the caller skip eager's `sub_` and `exp_`."""
        logits, row_scale, target = _inputs(16, 777)
        before = logits.clone()
        self._run(logits, row_scale, target, torch.bfloat16)
        self.assertEqual(logits, before)


instantiate_parametrized_tests(TestFusedGradLogitsKernel)

if __name__ == "__main__":
    run_tests()
