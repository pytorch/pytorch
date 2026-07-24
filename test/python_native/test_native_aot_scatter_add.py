# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for the native-AOT scatter_add TMA kernel @ CUDA.

scatter_add stress-tests manifest features the earlier ops don't: only
ONE of the op's two JIT paths is AOT-embedded (TMA; vec-scatter shapes
must keep their JIT eligibility), covered_axes reuses the JIT cond's
own eligibility helpers (TI layout analysis) instead of a hand-written
projection, the C++ prelude runs a TensorIterator analysis, and the
in-place variant registers under the trailing-underscore symbol
("scatter_add_") which must share the base declaration's coverage.

Routing is three-layer: AOT (TMA-eligible calls), JIT (vec-scatter-only
shapes, int32 indices), stock aten (everything else). JIT and AOT
compile the same kernel body, so profiler names can't separate those
two layers; layer preference is proven by the JIT compile cache staying
cold while the DSL kernel appears in the profile.

The kernel accumulates via cp.reduce.async.bulk in nondeterministic
order, so value checks use tolerances rather than bit-exactness.

Tests that require the AOT kernel library skip unless it was loaded at
import; correctness tests run everywhere.
"""

import unittest

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _aot_lib_loaded() -> bool:
    from torch._native import _native_aot_embedded

    return _native_aot_embedded()


def skipIfNoAotLib(fn):
    return unittest.skipUnless(
        _aot_lib_loaded(), "AOT kernels not embedded in this build"
    )(fn)


def _ran_dsl_kernel(fn) -> bool:
    """True if the CuTeDSL TMA kernel ran (either JIT or AOT layer);
    False when stock aten served (its kernels are named
    at::native::tma_scatter_add_kernel / _cuda_scatter_gather...)."""
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return any(
        e.name.startswith("kernel_cutlass_")
        for e in prof.events()
        if e.device_type.name == "CUDA"
    )


def _jit_cache_misses() -> int:
    from torch._native.ops.scatter_add.tma_kernel import _compile_tma_scatter

    return _compile_tma_scatter.cache_info().misses


def _reference(self_t, dim, index, src):
    with torch.backends.python_native.cutedsl.disabled():
        return torch.scatter_add(self_t, dim, index, src)


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "CUDA sm_90+ required")
@skipIfNoCuteDSL
class TestNativeAotScatterAdd(TestCase):
    def _inputs(self, m=8192, n=512, rows=4096, dtype=torch.float32, seed=0):
        torch.manual_seed(seed)
        src = torch.randn(m, n, device="cuda", dtype=dtype)
        index = torch.randint(0, rows, (m,), device="cuda")
        index = index.unsqueeze(-1).expand(m, n)
        self_t = torch.randn(rows, n, device="cuda", dtype=dtype)
        return self_t, index, src

    def _tol(self, dtype):
        # Accumulation order differs between kernels; halves accumulate
        # in-dtype (both routes), so order changes cost up to an ulp of
        # the partial-sum magnitude (~1e-1 at bf16 magnitude 4-8).
        return {"rtol": 5e-2, "atol": 1e-1} if dtype != torch.float32 else {}

    @skipIfNoAotLib
    def test_covered_call_routes_to_aot(self):
        self_t, index, src = self._inputs()
        misses_before = _jit_cache_misses()
        self.assertTrue(
            _ran_dsl_kernel(lambda: torch.scatter_add(self_t, 0, index, src))
        )
        # The DSL kernel ran but the JIT layer never compiled: AOT served.
        self.assertEqual(_jit_cache_misses(), misses_before)
        out = torch.scatter_add(self_t, 0, index, src)
        ref = _reference(self_t, 0, index, src)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_all_grid_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self_t, index, src = self._inputs(dtype=dtype, seed=1)
            misses_before = _jit_cache_misses()
            out = torch.scatter_add(self_t, 0, index, src)
            self.assertEqual(_jit_cache_misses(), misses_before, str(dtype))
            ref = _reference(self_t, 0, index, src)
            self.assertEqual(out, ref, **self._tol(dtype))

    @skipIfNoAotLib
    def test_inplace_and_out_variants(self):
        self_t, index, src = self._inputs(seed=2)
        ref = _reference(self_t, 0, index, src)
        x = self_t.clone()
        self.assertTrue(_ran_dsl_kernel(lambda: x.scatter_add_(0, index, src)))
        y = self_t.clone()
        y.scatter_add_(0, index, src)
        self.assertEqual(y, ref, rtol=1e-4, atol=1e-4)
        out = torch.empty_like(self_t)
        torch.scatter_add(self_t, 0, index, src, out=out)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_vec_scatter_shapes_keep_jit_eligibility(self):
        # Only the TMA path is AOT-embedded. A dst whose row stride is
        # 4B- but not 16B-aligned (here 516 floats = 2064 B... use 517)
        # fails TMA's 16B stride contract but satisfies vec-scatter's 4B
        # one, so the call must be UNCOVERED (keeps JIT eligibility) and
        # the vec-scatter JIT kernel must serve it.
        from torch._native import aot_manifest
        from torch._native.ops.scatter_add.vec_scatter_kernel import (
            _compile_vec_scatter,
        )

        m, rows, n = 4096, 512, 512
        base = torch.randn(rows, n + 5, device="cuda")
        self_t = base[:, :n]  # row stride 517 floats = 2068 B: %16 != 0
        src = torch.randn(m, n, device="cuda")
        index = torch.randint(0, rows, (m,), device="cuda")
        index = index.unsqueeze(-1).expand(m, n)
        self.assertFalse(
            aot_manifest.covers("scatter_add", "CUDA", (self_t, 0, index, src), {})
        )
        misses_before = _compile_vec_scatter.cache_info().misses
        out = torch.scatter_add(self_t, 0, index, src)
        served_by_jit_vec = _compile_vec_scatter.cache_info().misses > misses_before
        self.assertTrue(served_by_jit_vec or _compile_vec_scatter.cache_info().currsize)
        ref = _reference(self_t, 0, index, src)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_uncovered_calls_avoid_aot(self):
        from torch._native import aot_manifest

        self_t, index, src = self._inputs(seed=3)
        # dim=1 needs index values < self_t.size(1).
        idx_d1 = torch.randint(0, self_t.size(1), (256, 512), device="cuda")
        cases = {
            "fp64": (self_t.double(), 0, index, src.double()),
            "int32 index": (self_t, 0, index.int(), src),
            "dim=1": (self_t[:256], 1, idx_d1, src[:256, :512]),
        }
        for label, args in cases.items():
            self.assertFalse(
                aot_manifest.covers("scatter_add", "CUDA", args, {}),
                f"{label} must not be covered",
            )
            out = torch.scatter_add(*args)
            with torch.backends.python_native.cutedsl.disabled():
                ref = torch.scatter_add(*args)
            self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_deterministic_mode_avoids_dsl(self):
        self_t, index, src = self._inputs(seed=4)
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            self.assertFalse(
                _ran_dsl_kernel(lambda: torch.scatter_add(self_t, 0, index, src))
            )
        finally:
            torch.use_deterministic_algorithms(prior)

    @skipIfNoAotLib
    def test_empty_index_early_return(self):
        self_t = torch.randn(128, 32, device="cuda")
        index = torch.empty(0, 32, dtype=torch.long, device="cuda")
        src = torch.empty(0, 32, device="cuda")
        out = torch.scatter_add(self_t, 0, index, src)
        self.assertEqual(out, self_t)

    @skipIfNoAotLib
    def test_disabled_context_masks_aot(self):
        self_t, index, src = self._inputs(seed=5)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertFalse(
                _ran_dsl_kernel(lambda: torch.scatter_add(self_t, 0, index, src))
            )
        self.assertTrue(
            _ran_dsl_kernel(lambda: torch.scatter_add(self_t, 0, index, src))
        )

    @skipIfNoAotLib
    def test_cpp_covers_agrees_with_python(self):
        # The AOT lib registers torch.ops._native_aot.covers_scatter_add
        # (from the declaration's cpp_covers); the router prefers it. It
        # must decide the same covered set as the Python covered_axes
        # matching on every case this suite exercises.
        from torch._native import aot_manifest

        c = aot_manifest.get_coverage("scatter_add", "CUDA")
        self.assertIsNotNone(c._resolve_cpp_covers())
        py = aot_manifest._Coverage("scatter_add", c._covered_axes, c._grid)
        py._cpp_probed = True  # pin to the Python path

        self_t, index, src = self._inputs(seed=7)
        base = torch.randn(4096, 517, device="cuda")
        cases = {
            "covered": ((self_t, 0, index, src), {}),
            "neg dim": ((self_t, -2, index, src), {}),
            "out kwarg": ((self_t, 0, index, src), {"out": torch.empty_like(self_t)}),
            "int32 index": ((self_t, 0, index.int(), src), {}),
            "fp64": ((self_t.double(), 0, index, src.double()), {}),
            "unaligned stride": ((base[:, :512], 0, index[:4096], src[:4096]), {}),
        }
        for label, (args, kwargs) in cases.items():
            self.assertEqual(
                c.covers(args, kwargs), py.covers(args, kwargs), f"drift on {label}"
            )

    @skipIfNoAotLib
    def test_oob_index_traps(self):
        # The kernels bounds-check indices with a predicated PTX trap
        # (_ptx.trap_if_oob, replacing --enable-assertions): an OOB
        # index must surface as a CUDA error, never silent corruption.
        # Subprocess: the trap poisons the CUDA context.
        import subprocess
        import sys

        code = (
            "import torch\n"
            "M, N = 512, 512\n"
            "src = torch.randn(M, N, device='cuda')\n"
            "bad = torch.randint(0, M // 2, (M,), device='cuda')\n"
            "bad[7] = M\n"
            "index = bad.unsqueeze(-1).expand(M, N)\n"
            "self_t = torch.randn(M // 2, N, device='cuda')\n"
            "torch.scatter_add(self_t, 0, index, src)\n"
            "torch.cuda.synchronize()\n"
            "print('NO-ERROR')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertNotIn("NO-ERROR", proc.stdout, "OOB index completed silently")
        self.assertNotEqual(proc.returncode, 0)

    def test_covered_call_correct_regardless_of_routing(self):
        self_t, index, src = self._inputs(seed=6)
        out = torch.scatter_add(self_t, 0, index, src)
        ref = _reference(self_t, 0, index, src)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    run_tests()
