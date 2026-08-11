# Owner(s): ["module: dsl-native-ops"]

import os
import unittest
from unittest.mock import patch

import torch
import torch.autograd.forward_ad as fwAD
import torch.backends.python_native as pn
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


EPS = 1e-5
DISPATCH_M = 8192
DISPATCH_N = 4096
DISPATCH_DTYPE = torch.float16
BACKWARD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
KERNEL_PATH_NS = (4096, 12288, 24576, 114688, 4097, 4103, 8193, 16385, 32769, 98305)


def _unsupported_environment_reason() -> str | None:
    """Why this machine cannot exercise the override, or None if it can.

    Only environmental facts belong here. Whether the override actually
    registered on a machine that satisfies all of them is a property of the
    gate and the registry, so it is asserted by a test rather than folded in
    here -- a regression there must fail, not quietly skip this whole class.
    """
    if not TEST_CUDA or torch.version.hip is None:
        return "ROCm required"

    from torch._native import flydsl_utils as fu

    if fu.check_native_jit_disabled():
        return "native DSL overrides are disabled via TORCH_DISABLE_NATIVE_JIT"
    if not fu.runtime_available():
        return "FlyDSL runtime is not installed"
    if not fu._version_is_ok():
        return f"FlyDSL {fu.runtime_version()} is outside the supported release"

    from torch._native.ops.norm.flydsl_rmsnorm_impl import _is_supported_arch

    if not _is_supported_arch(torch.cuda.current_device()):
        return "FlyDSL RMSNorm override requires gfx950"
    return None


_UNSUPPORTED_REASON = _unsupported_environment_reason()


class TestFlyDSLRMSNormArch(TestCase):
    @parametrize(
        "arch,expected",
        (
            ("gfx950", True),
            # _resolve_rocm_arch returns HSA_OVERRIDE_GFX_VERSION verbatim, so
            # the gate has to tolerate feature flags rather than compare the
            # whole string.
            ("gfx950:sramecc+", True),
            ("gfx950:sramecc+:xnack-", True),
            ("gfx942", False),
            ("gfx942:sramecc+", False),
            (None, False),
        ),
    )
    def test_arch_gate_allows_only_gfx950(self, arch, expected):
        import torch._native.ops.norm.flydsl_rmsnorm_impl as flydsl_rmsnorm_impl

        with patch.object(
            flydsl_rmsnorm_impl.fu, "_resolve_rocm_arch", return_value=arch
        ):
            self.assertEqual(flydsl_rmsnorm_impl._is_supported_arch(0), expected)

    def test_arch_gate_follows_env_changes(self):
        # The gate must not cache: _resolve_rocm_arch re-reads FLYDSL_GPU_ARCH
        # on every call and rmsnorm_fwd compiles for whatever it returns, so a
        # remembered verdict would admit work for one arch and build for another.
        import torch._native.ops.norm.flydsl_rmsnorm_impl as flydsl_rmsnorm_impl

        with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx950"}):
            self.assertTrue(flydsl_rmsnorm_impl._is_supported_arch(0))
        with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx942"}):
            self.assertFalse(flydsl_rmsnorm_impl._is_supported_arch(0))


class TestFlyDSLRMSNormHelpers(TestCase):
    """Helpers that touch neither flydsl nor a device.

    These run everywhere, including the machines where the kernel tests below
    are skipped for lack of a gfx950 device.
    """

    @parametrize(
        "normalized_shape,expected",
        (
            (128, 128),
            ([128], 128),
            ((128,), 128),
            # More than one normalized dimension: the kernel flattens only the
            # last one, so these decline rather than guess.
            ([2, 64], None),
            ([], None),
            # Not a sequence, and a one-element sequence that is not an int.
            (None, None),
            (["n"], None),
        ),
    )
    def test_normalized_shape_1d(self, normalized_shape, expected):
        from torch._native.ops.norm.flydsl_rmsnorm_utils import normalized_shape_1d

        self.assertEqual(normalized_shape_1d(normalized_shape), expected)

    @parametrize(
        "rows_m,n,itemsize,expected",
        (
            (2048, 114688, 4, True),
            (16383, 65536, 4, True),
            (16384, 65536, 4, False),
            (16385, 65536, 4, False),
            (1 << 31, 1, 1, False),
            (1, 1 << 31, 1, False),
        ),
    )
    def test_flydsl_buffer_span(self, rows_m, n, itemsize, expected):
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _fits_int32_buffer_span

        self.assertEqual(_fits_int32_buffer_span(rows_m, n, itemsize), expected)

    def test_impl_without_weight_raises(self):
        # The predicate declines weight=None, so reaching the impl means a
        # caller bypassed it. The error has to name the missing argument
        # instead of surfacing from somewhere inside the kernel wrapper.
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _fused_rms_norm_impl

        x = make_tensor((8, 128), device="cpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires an explicit weight"):
            _fused_rms_norm_impl(x, [128], None, EPS)


@unittest.skipIf(_UNSUPPORTED_REASON is not None, str(_UNSUPPORTED_REASON))
class TestFlyDSLRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import clear_rmsnorm_caches

        clear_rmsnorm_caches()
        torch.manual_seed(0)

    def _make_inputs(self, m, n, dtype, *, requires_grad=False):
        x = make_tensor((m, n), device="cuda", dtype=dtype, requires_grad=requires_grad)
        weight = make_tensor(
            (n,), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        return x, weight

    def _make_dispatch_inputs(self):
        return self._make_inputs(DISPATCH_M, DISPATCH_N, DISPATCH_DTYPE)

    def _assert_no_flydsl_compiles(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 0)

    def _assert_nonfinite_matches(self, actual, expected):
        self.assertEqual(torch.isnan(actual), torch.isnan(expected))
        self.assertEqual(torch.isinf(actual), torch.isinf(expected))
        finite = torch.isfinite(expected)
        self.assertEqual(actual[finite], expected[finite])

    def test_override_is_registered(self):
        # This class only runs where every precondition the gate checks is
        # already satisfied, so a missing registration means the gate or the
        # registry regressed. The rest of the class would then silently
        # compare aten against aten, which is why this is asserted rather
        # than made a skip condition.
        self.assertIn("_fused_rms_norm", pn.get_dsl_operations("flydsl"))

    @parametrize("dtype", BACKWARD_DTYPES)
    def test_backward_through_override_matches_aten(self, dtype):
        # The override sits at the CUDA key, below Autograd, so ATen's
        # _fused_rms_norm_backward_cuda consumes the rstd this kernel writes:
        # fp32, shaped (*batch, 1). A drift in that contract shows up as wrong
        # gradients rather than an error, and OpInfo's test_backward only runs
        # fp32, so fp16/bf16 -- where the fp32 rstd meets a lower-precision
        # input -- would otherwise go unchecked.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        clear_rmsnorm_caches()
        x, weight = self._make_inputs(DISPATCH_M, DISPATCH_N, dtype, requires_grad=True)
        grad_out = make_tensor(x.shape, device=x.device, dtype=x.dtype)

        def grads():
            out = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            return torch.autograd.grad(out, (x, weight), grad_out)

        with pn.flydsl.disabled():
            ref_dx, ref_dw = grads()
        self._assert_no_flydsl_compiles()

        got_dx, got_dw = grads()
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got_dx, ref_dx)
        self.assertEqual(got_dw, ref_dw)

    def test_forward_ad_through_override_matches_aten(self):
        # The OpInfo variant advertises forward AD, but the gradient suites
        # filter to float64/complex128, which it does not list, so nothing there
        # ever reaches this kernel. Assert the JVP here instead, and that the
        # override really ran while producing it.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        x, weight = self._make_inputs(DISPATCH_M, DISPATCH_N, torch.float32)
        tangent_x = make_tensor(x.shape, device=x.device, dtype=x.dtype)
        tangent_w = make_tensor(weight.shape, device=x.device, dtype=x.dtype)

        def jvp():
            with fwAD.dual_level():
                dual_x = fwAD.make_dual(x, tangent_x)
                dual_w = fwAD.make_dual(weight, tangent_w)
                out = torch.rms_norm(dual_x, (DISPATCH_N,), dual_w, EPS)
                primal, tangent = fwAD.unpack_dual(out)
                return primal.clone(), tangent.clone()

        clear_rmsnorm_caches()
        with pn.flydsl.disabled():
            ref_primal, ref_tangent = jvp()
        self._assert_no_flydsl_compiles()

        got_primal, got_tangent = jvp()
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got_primal, ref_primal)
        self.assertEqual(got_tangent, ref_tangent)

    def test_double_backward_through_override_matches_aten(self):
        # Same gap as the forward-AD case: fwgrad_bwgrad is advertised but the
        # dtype filter keeps the OpInfo suites off this kernel entirely.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        x, weight = self._make_inputs(
            DISPATCH_M, DISPATCH_N, torch.float32, requires_grad=True
        )
        grad_out = make_tensor(x.shape, device=x.device, dtype=x.dtype)

        def gradgrad():
            out = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            dx, dw = torch.autograd.grad(out, (x, weight), grad_out, create_graph=True)
            return torch.autograd.grad(dx.sum() + dw.sum(), (x, weight))

        clear_rmsnorm_caches()
        with pn.flydsl.disabled():
            ref_ddx, ref_ddw = gradgrad()
        self._assert_no_flydsl_compiles()

        got_ddx, got_ddw = gradgrad()
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)
        self.assertEqual(got_ddx, ref_ddx)
        self.assertEqual(got_ddw, ref_ddw)

    @parametrize("eps_kind", ("cancelling", "negative", "nan"))
    def test_nonpositive_eps_falls_back_to_aten(self, eps_kind):
        # A negative eps can drive mean(x^2) + eps to exactly zero under one
        # reduction order and to a tiny positive value under another. Measured
        # on this shape with eps = -mean(x^2): the kernel returned rstd=inf
        # where aten returned 2896.31, and every element of that row differed.
        # The predicate declines instead, so aten's answer is the only answer.
        m, n = 2048, 16384
        x, weight = self._make_inputs(m, n, torch.float32)
        if eps_kind == "cancelling":
            eps = -(x.double() ** 2).mean(dim=-1)[0].item()
        elif eps_kind == "negative":
            eps = -1.0
        else:
            eps = float("nan")

        out, rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, eps)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, eps)

        self._assert_no_flydsl_compiles()
        self._assert_nonfinite_matches(out, ref)
        self._assert_nonfinite_matches(rstd, ref_rstd)

    def test_direct_forward_matches_aten_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        x, weight = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, _ = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)

        out, rstd = rmsnorm_fwd(x, [128], weight, EPS)
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertEqual(rstd.shape, (16, 1))
        self.assertEqual(rstd.device, x.device)
        self.assertEqual(out, ref_out)

        # A 3-D input with the same N/dtype/device must hit the same dynamic-M
        # specialization instead of compiling a second kernel.
        x3 = make_tensor((2, 16, 128), device="cuda", dtype=x.dtype)
        out3, _ = rmsnorm_fwd(x3, [128], weight, EPS)
        self.assertEqual(out3.shape, x3.shape)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_nondefault_stream_reuses_specialization(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            rmsnorm_cache_info,
            rmsnorm_fwd,
        )

        x, weight = self._make_inputs(16, 128, torch.float16)
        with pn.flydsl.disabled():
            ref_out, ref_rstd = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)

        rmsnorm_fwd(x, [128], weight, EPS)
        stream = torch.cuda.Stream(device=x.device)
        with torch.cuda.stream(stream):
            out, rstd = rmsnorm_fwd(x, [128], weight, EPS)
        stream.synchronize()

        self.assertEqual(out, ref_out)
        self.assertEqual(rstd, ref_rstd)
        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    @parametrize("dtype", BACKWARD_DTYPES)
    @parametrize("n", KERNEL_PATH_NS)
    def test_numerics_across_kernel_paths(self, dtype, n):
        # OpInfo covers the shapes the dispatcher accepts in the common case;
        # this walks the kernel's internal branches instead, which OpInfo has
        # no way to target. Each N is chosen for a specific one:
        #
        #   4096   fast path, one vector per thread
        #   12288  fast path, multiple vectors per thread
        #   114688 fast path, upper end of the dispatcher's N range
        #
        # The rest are odd, so N % vec_width != 0 for both vector widths (8 for
        # fp16/bf16, 4 for fp32) and they take the generic path's scalar tail.
        # 4097/8193/16385/32769 are the shapes the PR reports the largest wins
        # on, so the tail is the branch those numbers actually come from. Those
        # all leave exactly one element in the tail; 4103 leaves seven (three
        # for fp32) so the tail's own bounds check is exercised too.
        #
        # Rows are kept small and rmsnorm_fwd is called directly: this is about
        # the reduction being right, not about dispatch.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        rows = 8
        x, weight = self._make_inputs(rows, n, dtype)
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [n], weight, EPS)
        got, got_rstd = rmsnorm_fwd(x, [n], weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(got_rstd.dtype, torch.float32)
        self.assertEqual(got_rstd.shape, (rows, 1))
        self.assertEqual(got_rstd, ref_rstd)

    @parametrize("dtype", BACKWARD_DTYPES)
    def test_nonfinite_input_matches_aten(self, dtype):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(2, 128, dtype)
        x[0, 0] = float("nan")
        x[1, 0] = float("inf")
        with pn.flydsl.disabled():
            ref, ref_rstd = torch.ops.aten._fused_rms_norm(x, [128], weight, EPS)
        got, got_rstd = rmsnorm_fwd(x, [128], weight, EPS)

        self._assert_nonfinite_matches(got, ref)
        self._assert_nonfinite_matches(got_rstd, ref_rstd)

    def test_public_rms_norm_dispatches_to_flydsl(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    @parametrize("m,n", ((8192, 4096), (4096, 8192), (2048, 16384)))
    def test_band_minimum_dispatches(self, m, n):
        # The smallest (rows, N) each band in _fused_rms_norm_fwd_perf_wins
        # accepts. The OpInfo variant deliberately carries only one shape this
        # large, so the other bands are covered here instead of in the general
        # op_db where every TestCommon test would pay for them.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_inputs(m, n, DISPATCH_DTYPE)
        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (n,), weight, EPS)
        got = torch.rms_norm(x, (n,), weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_runtime_eps_dispatches_and_reuses_cache(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        x, weight = self._make_dispatch_inputs()
        for eps in (EPS, 1e-6):
            with pn.flydsl.disabled():
                ref = torch.rms_norm(x, (DISPATCH_N,), weight, eps)
            got = torch.rms_norm(x, (DISPATCH_N,), weight, eps)
            self.assertEqual(got, ref)

        info = rmsnorm_cache_info()["fwd"]
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.currsize, 1)

    def test_nn_rmsnorm_default_eps_dispatches(self):
        # nn.RMSNorm leaves eps at None and passes it straight through, so this
        # is the shape of call a real model makes. Declining eps=None in the
        # predicate would make the override unreachable from nn.RMSNorm while
        # every explicit-eps test here still passed.
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        module = torch.nn.RMSNorm(DISPATCH_N, device="cuda", dtype=DISPATCH_DTYPE)
        self.assertIsNone(module.eps)
        with torch.no_grad():
            module.weight.copy_(
                make_tensor((DISPATCH_N,), device="cuda", dtype=DISPATCH_DTYPE)
            )
        x = make_tensor((DISPATCH_M, DISPATCH_N), device="cuda", dtype=DISPATCH_DTYPE)

        with pn.flydsl.disabled():
            ref = module(x)
        got = module(x)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_multi_dim_normalized_shape_raises(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(8, 128, torch.float16)
        with self.assertRaisesRegex(ValueError, "one normalized dimension"):
            rmsnorm_fwd(x.view(8, 8, 16), [8, 16], weight, EPS)

    def test_unsupported_dtype_raises(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_fwd

        x, weight = self._make_inputs(8, 128, torch.float64)
        with self.assertRaisesRegex(TypeError, "unsupported RMSNorm dtype"):
            rmsnorm_fwd(x, [128], weight, EPS)

    def test_unresolvable_arch_raises(self):
        # The arch selects the wave size baked into the reduction, so a missing
        # answer has to be an error rather than a guess.
        from torch._native.ops.norm import flydsl_rmsnorm_fwd

        x, weight = self._make_inputs(16, 128, torch.float16)
        with patch.object(flydsl_rmsnorm_fwd, "_resolve_rocm_arch", return_value=None):
            with self.assertRaisesRegex(
                RuntimeError, "Could not determine the ROCm arch"
            ):
                flydsl_rmsnorm_fwd.rmsnorm_fwd(x, [128], weight, EPS)

    def test_n_above_upper_bound_falls_back_without_compiling(self):
        # 114688 is the largest N the dispatcher accepts.
        n = 131072
        x = make_tensor((2048, n), device="cuda", dtype=DISPATCH_DTYPE)
        weight = make_tensor((n,), device="cuda", dtype=DISPATCH_DTYPE)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (n,), weight, EPS)
        got = torch.rms_norm(x, (n,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()

    def test_fused_aten_noncontiguous_input_falls_back_without_compiling(self):
        base = make_tensor(
            (DISPATCH_N, DISPATCH_M), device="cuda", dtype=DISPATCH_DTYPE
        )
        x = base.transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        weight = make_tensor((DISPATCH_N,), device="cuda", dtype=x.dtype)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()

    @parametrize("dtype", (torch.float16, torch.float32))
    def test_misaligned_base_dispatches_and_matches_aten(self, dtype):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import (
            clear_rmsnorm_caches,
            rmsnorm_cache_info,
        )

        # Both dtypes issue 128-bit copies (fp32 4x32, fp16 8x16), so both hit
        # the kernel with a base address that is not 16-byte aligned.
        clear_rmsnorm_caches()
        storage = make_tensor(
            (DISPATCH_M * DISPATCH_N + 1,), device="cuda", dtype=dtype
        )
        weight_storage = make_tensor((DISPATCH_N + 1,), device="cuda", dtype=dtype)
        x = storage[1:].view(DISPATCH_M, DISPATCH_N)
        weight = weight_storage[1:]
        self.assertTrue(x.is_contiguous())
        self.assertNotEqual(x.data_ptr() % 16, 0)
        self.assertNotEqual(weight.data_ptr() % 16, 0)

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)

    def test_cow_inputs_fall_back_without_materializing(self):
        # The predicate declines copy-on-write inputs because flattening an N-D
        # input calls reshape, which would materialize them. Both halves matter:
        # that the shape is otherwise accepted, and that the tensors are still
        # COW once the call has gone through aten instead.
        from torch._native.ops.norm.flydsl_rmsnorm_impl import _fused_rms_norm_cond

        x, weight = self._make_dispatch_inputs()
        # Same shape without COW dispatches, so COW is the only thing declining
        # the cases below. Checked before any _lazy_clone call: cloning marks
        # the source COW as well, so x stops being a clean control afterwards.
        self.assertTrue(_fused_rms_norm_cond(x, [DISPATCH_N], weight, EPS))

        x_cow = x._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(x_cow))
        self.assertFalse(_fused_rms_norm_cond(x_cow, [DISPATCH_N], weight, EPS))

        plain_x, plain_weight = self._make_dispatch_inputs()
        weight_cow = plain_weight._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(weight_cow))
        self.assertFalse(_fused_rms_norm_cond(plain_x, [DISPATCH_N], weight_cow, EPS))

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        got = torch.rms_norm(x_cow, (DISPATCH_N,), weight, EPS)

        self.assertEqual(got, ref)
        self._assert_no_flydsl_compiles()
        self.assertTrue(torch._C._is_cow_tensor(x_cow))
        self.assertTrue(torch._C._is_cow_tensor(weight_cow))

    def test_user_disable_falls_back_and_restores(self):
        from torch._native.ops.norm.flydsl_rmsnorm_fwd import rmsnorm_cache_info

        # A shape the dispatcher would otherwise take, so being disabled is the
        # only reason nothing compiles inside the block.
        x, weight = self._make_dispatch_inputs()

        with pn.flydsl.disabled():
            ref = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
            self._assert_no_flydsl_compiles()

        # Leaving the block must re-enable the override, so this call is the
        # FlyDSL kernel and the comparison against ref is aten vs FlyDSL.
        got = torch.rms_norm(x, (DISPATCH_N,), weight, EPS)
        self.assertEqual(got, ref)
        self.assertEqual(rmsnorm_cache_info()["fwd"].misses, 1)


instantiate_parametrized_tests(TestFlyDSLRMSNormArch)
instantiate_parametrized_tests(TestFlyDSLRMSNormHelpers)
instantiate_parametrized_tests(TestFlyDSLRMSNorm)


if __name__ == "__main__":
    run_tests()
