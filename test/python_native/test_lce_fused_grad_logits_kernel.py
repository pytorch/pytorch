# Owner(s): ["module: dsl-native-ops"]
#
# Numerics of the CuTeDSL fused_grad_logits kernel, checked directly against a
# torch expression of its contract.

import unittest

import torch
from torch._native import cutedsl_utils as cu
from torch.testing._internal.common_cuda import (
    has_device_side_assert,
    SM80OrLater,
    TEST_CUDA,
)
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
    # Both statistics shifted by the row max, as in the kernel.
    return (
        g.to(out_dtype),
        row_sum.log(),
        logits[rows, target] - row_max.squeeze(1),
    )


def _outputs(num_rows, V, dtype, device="cuda", row_stride=None):
    if row_stride is None:
        g = torch.empty(num_rows, V, device=device, dtype=dtype)
    else:
        # A wider row stride than V: the kernel indexes through the tensor's
        # own strides, so this must work as well as the packed case.
        g = torch.empty(num_rows, row_stride, device=device, dtype=dtype)[:, :V]
    term = torch.empty(num_rows, device=device, dtype=torch.float32)
    return g, term


def _inputs(num_rows, V, device="cuda", logits_dtype=torch.float32):
    logits = torch.randn(num_rows, V, device=device, dtype=torch.float32).to(
        logits_dtype
    )
    row_scale = torch.rand(num_rows, device=device, dtype=torch.float32) + 0.5
    target = torch.randint(0, V, (num_rows,), device=device)
    return logits, row_scale, target


# These tests call the kernel directly, with no `cond` to decline an
# unsupported device, so they need the CuTeDSL runtime's floor, sm_80.
@unittest.skipIf(not TEST_CUDA, "CuTeDSL kernels are CUDA-only")
@unittest.skipIf(not SM80OrLater, "the CuTeDSL runtime requires sm_80 or later")
class TestFusedGradLogitsKernel(TestCase):
    def setUp(self):
        super().setUp()
        if not cu.runtime_available() or cu.check_native_jit_disabled():
            self.skipTest("CuTeDSL runtime unavailable or native DSL disabled")
        from torch._native.ops.linear_cross_entropy import fused_grad_logits_kernel

        self.kernel = fused_grad_logits_kernel

    def _run(
        self, logits, row_scale, target, dtype, row_stride=None, aliased=False, **meta
    ):
        g, term = _outputs(*logits.shape, dtype, row_stride=row_stride)
        if aliased:
            g = logits.view(dtype).narrow(1, 0, logits.shape[1])
        self.kernel.fused_grad_logits_into(g, term, logits, row_scale, target, **meta)
        return g, term

    def _check(
        self, logits, row_scale, target, dtype, row_stride=None, aliased=False, **meta
    ):
        # The reference comes first: an aliased run overwrites the logits.
        want_g, want_log_row_sum, want_shifted = _reference(
            logits, row_scale, target, dtype
        )
        g, term = self._run(
            logits, row_scale, target, dtype, row_stride, aliased, **meta
        )
        if dtype is torch.float32:
            # The kernel's exponential is the hardware approximation and its
            # row sum reduces in a different order than torch's.
            self.assertEqual(g, want_g, atol=1e-6, rtol=1e-5)
        else:
            self.assertEqual(g, want_g)
        # The row max cancels out of `term` and `g`, so this sees only its effect
        # on the row sum; `test_a_lone_high_column_pins_the_row_max` pins the rest.
        self.assertEqual(
            term, row_scale * (want_log_row_sum - want_shifted), atol=1e-4, rtol=1e-5
        )

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
            (8, 2049),  # partial tiles inside one staging group
            (37, 4097),  # one column past a full staging group
            (128, 32000),  # a realistic chunk, many staging groups
        ],
    )
    def test_aliased_g_shares_the_logits_storage(self, dtype, num_rows, V):
        """`g` written into the logits' own bytes: `g[n, j]` occupies
        `z[n, j // 2]`. An element left unwritten would hold reinterpreted fp32
        bytes, so matching the reference everywhere also proves full coverage.
        The target logit is read before any write could occupy its bytes,
        which the loss term would expose: it is the only place that logit
        reaches an output. The write ordering is not what this checks: at the
        default staging depth a barrier-less kernel still passes.
        `compute-sanitizer --tool racecheck` checks it."""
        logits, row_scale, target = _inputs(num_rows, V)
        self._check(logits, row_scale, target, dtype, aliased=True)

    @parametrize("num_rows, V", [(4, 4097), (8, 12289)])
    def test_aliased_g_at_the_same_width_as_the_logits(self, num_rows, V):
        """The fp16 layout: an fp16 buffer aliased by an fp16 `g`, so `g[n, j]`
        lands exactly on `z[n, j]` -- a compile key of its own. At this width
        each thread overwrites only what it read, so this pins the values, not
        the write ordering."""
        logits, row_scale, target = _inputs(num_rows, V, logits_dtype=torch.float16)
        self._check(logits, row_scale, target, torch.float16, aliased=True)

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
        self._check(logits + 10000.0, row_scale, target, torch.float32)

    def test_a_lone_high_column_pins_the_row_max(self):
        """The row max cancels out of both outputs, so its one load-bearing job
        is keeping `exp` in range. Here it sits alone in the last thread's
        column, so the range depends on one partial surviving both reduction
        stages."""
        num_rows, V = 4, 4096
        logits = torch.zeros((num_rows, V), device="cuda", dtype=torch.float32)
        # Lane 31 of the last warp: `block_reduce` publishes only from lane 0, so
        # this value must survive the warp butterfly and the cross-warp combine;
        # exp(120) is inf in fp32.
        logits[:, self.kernel._DEFAULT_THREADS_PER_BLOCK - 1] = 120.0
        _, row_scale, target = _inputs(num_rows, V)
        self._check(logits, row_scale, target, torch.float32)

    def test_uniform_logits_give_the_log_class_count(self):
        num_rows, V = 4, 2048
        logits = torch.full((num_rows, V), 3.5, device="cuda", dtype=torch.float32)
        _, row_scale, target = _inputs(num_rows, V)
        _, term = self._run(logits, row_scale, target, torch.float32)
        # Uniform logits: lse = z + log(V) and the target logit is z, so the
        # row's whole loss term is scale * log(V) whatever z is.
        want = row_scale * torch.tensor(float(V), device=term.device).log()
        self.assertEqual(term, want, atol=1e-4, rtol=1e-6)

    def test_target_column_is_the_shifted_probability(self):
        """The one-hot subtract is what replaces eager's index_add_, so the
        target column is checked explicitly rather than only in aggregate."""
        num_rows, V = 16, 512
        logits, row_scale, target = _inputs(num_rows, V)
        g, term = self._run(logits, row_scale, target, torch.float32)
        rows = torch.arange(num_rows, device=g.device)
        # term = scale * (lse - z_target), so lse comes back out of it.
        lse = term / row_scale + logits[rows, target]
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

    def test_masked_class_does_not_poison_the_row(self):
        """A -inf logit is a masked class: its probability is 0 and the rest of
        the row is an ordinary softmax. The online pass sees it with m = -inf
        when it is the first column a thread touches, where `exp(z - m)` would
        be `exp(nan)`."""
        num_rows, V = 8, 1024
        logits, row_scale, target = _inputs(num_rows, V)
        # Column 0 is thread 0's first column, and the block is wider than one
        # warp, so this is exactly the first-touch case.
        logits[:, 0] = float("-inf")
        self._check(logits, row_scale, target, torch.float32)

    def test_every_class_masked_is_nan_like_eager(self):
        """A fully masked row has no valid class: eager's shifted softmax is
        `-inf - -inf`, so the row is NaN there, and must be NaN here too."""
        num_rows, V = 4, 512
        logits, row_scale, target = _inputs(num_rows, V)
        logits[1] = float("-inf")
        g, term = self._run(logits, row_scale, target, torch.float32)
        self.assertTrue(torch.isnan(g[1]).all())
        self.assertTrue(torch.isnan(term[1]))
        self.assertTrue(torch.isfinite(g[0]).all())

    def test_nan_logit_poisons_its_row(self):
        """NaN must reach the output: the online pass's -inf guard lets a NaN
        take the accumulate branch."""
        num_rows, V = 4, 512
        logits, row_scale, target = _inputs(num_rows, V)
        logits[2, 7] = float("nan")
        g, term = self._run(logits, row_scale, target, torch.float32)
        self.assertTrue(torch.isnan(g[2]).all())
        self.assertTrue(torch.isnan(term[2]))
        self.assertTrue(torch.isfinite(g[0]).all())

    # `1 << 32` is the one that needs int64 to catch: its low 32 bits are zero,
    # so a check made after narrowing to int32 reads it as class 0 and passes.
    @parametrize(
        "bad_target, message",
        [
            (-1, "target < 0"),
            (1 << 20, "target >= num_classes"),
            (1 << 32, "target >= num_classes"),
            ((1 << 32) + 1, "target >= num_classes"),
        ],
    )
    def test_out_of_range_target_traps(self, bad_target, message):
        """Like eager's index assert. The trap takes the CUDA context with it,
        so it runs in a child process."""
        stdout, stderr = self.run_process_no_exception(f"""
import torch
from torch._native.ops.linear_cross_entropy import fused_grad_logits_kernel as k
logits = torch.randn(6, 256, device="cuda")
target = torch.randint(0, 256, (6,), device="cuda")
target[3] = {bad_target}
term = torch.empty(6, device="cuda")
k.fused_grad_logits_into(
    torch.empty_like(logits), term, logits, torch.ones(6, device="cuda"), target
)
torch.cuda.synchronize()
""")
        self.assertTrue(has_device_side_assert(stderr.decode()))
        self.assertIn(f"linear_cross_entropy: {message}", stdout.decode())

    def test_zero_row_scale_gives_a_zero_gradient(self):
        """An ignored row carries scale 0, so its whole gradient row -- target
        column included -- and its loss contribution must both be exactly zero."""
        num_rows, V = 6, 300
        logits, row_scale, target = _inputs(num_rows, V)
        row_scale = torch.zeros_like(row_scale)
        g, term = self._run(logits, row_scale, target, torch.bfloat16)
        self.assertEqual(g, torch.zeros_like(g))
        self.assertEqual(term, torch.zeros_like(term))

    def test_wider_row_stride(self):
        logits, row_scale, target = _inputs(24, 100)
        self._check(logits, row_scale, target, torch.bfloat16, row_stride=128)

    @parametrize(
        "meta",
        [
            {},
            {"threads_per_block": 128, "tiles_per_stage": 4},
            {"tiles_per_stage": 1},
            {"threads_per_block": 1024, "tiles_per_stage": 8},
        ],
    )
    def test_shape_knobs_do_not_change_the_result(self, meta):
        """Every legal (threads_per_block, tiles_per_stage) gives the same result."""
        logits, row_scale, target = _inputs(24, 4097)
        self._check(logits, row_scale, target, torch.bfloat16, **meta)

    @parametrize(
        "meta, message",
        [
            ({"threads_per_block": 100}, "multiple of 32"),
            # A multiple of 32 on both sides of the range, so each one can only
            # be reported by the bound it crosses.
            ({"threads_per_block": 2048}, r"in \[32, 1024\]"),
            ({"threads_per_block": 0}, r"in \[32, 1024\]"),
            ({"tiles_per_stage": 0}, "at least 1"),
            ({"tiles_pre_stage": 4}, "unknown meta parameters"),
        ],
    )
    def test_illegal_shape_knobs_raise(self, meta, message):
        """A typo or an out-of-range value must fail loudly rather than compile
        something that silently drops per-warp partials (above 32 warps) or
        never runs (a zero-tile stage)."""
        logits, row_scale, target = _inputs(4, 64)
        with self.assertRaisesRegex(ValueError, message):
            self._run(logits, row_scale, target, torch.bfloat16, **meta)

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
