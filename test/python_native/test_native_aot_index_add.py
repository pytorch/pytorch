# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for the native-AOT index_add @ CUDA.

index_add re-exports the scatter_add TMA kernel (index_add(x, 0, i, s)
with alpha=1 IS scatter_add on the expanded index; the TMA ABI takes a
1D index, so the mapping is direct) -- one kernel body serving two aten
ops is the feature under test. It also exercises: a precomputed dim
(negative schema dims arrive wrapped), a prelude reject on a Scalar's
VALUE (alpha != 1 declines; the kernel bakes alpha=1), and an
early-return path (empty index). There is no JIT override for
index_add, so routing is two-layer: AOT or stock aten.

Duplicate indices accumulate via cp.reduce.async.bulk in
nondeterministic order, so value checks use tolerances (matching aten's
own nondeterministic CUDA index_add) rather than bit-exactness.

Tests that require the AOT kernel library skip unless it was loaded at
import; correctness tests run everywhere.
"""

import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _aot_lib_loaded() -> bool:
    from torch._native import _native_aot_embedded

    return _native_aot_embedded()


def skipIfNoAotLib(fn):
    return unittest.skipUnless(
        _aot_lib_loaded(), "AOT kernels not embedded in this build"
    )(fn)


def _ran_aot(fn) -> bool:
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return any(
        e.name.startswith("kernel_cutlass_")
        for e in prof.events()
        if e.device_type.name == "CUDA"
    )


def _load_covered_axes():
    # The declaration module is stdlib-only and loaded by file path (it
    # is not an importable member of the torch package).
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(torch.__file__), "_native", "ops", "index_add", "aot.py"
    )
    spec = importlib.util.spec_from_file_location("index_add_aot_t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.covered_axes


def _reference(self_t, dim, index, source, alpha=1):
    with torch.backends.python_native.cutedsl.disabled():
        return torch.index_add(self_t, dim, index, source, alpha=alpha)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestNativeAotIndexAdd(TestCase):
    def _inputs(self, rows=1000, cols=256, n_idx=5000, seed=0):
        # cols * 4B is 16B-aligned (TMA contract); fresh allocations are
        # 16B-aligned bases.
        torch.manual_seed(seed)
        self_t = torch.randn(rows, cols, device="cuda")
        index = torch.randint(0, rows, (n_idx,), device="cuda")
        source = torch.randn(n_idx, cols, device="cuda")
        return self_t, index, source

    @skipIfNoAotLib
    def test_covered_call_routes_to_aot(self):
        self_t, index, source = self._inputs()
        self.assertTrue(_ran_aot(lambda: torch.index_add(self_t, 0, index, source)))
        out = torch.index_add(self_t, 0, index, source)
        ref = _reference(self_t, 0, index, source)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_same_kernel_as_scatter_add(self):
        # The two ops must launch the IDENTICAL kernel symbol for the
        # equivalent call -- the point of the shared-kernel design.
        from torch.profiler import profile, ProfilerActivity

        self_t, index, source = self._inputs(seed=10)
        expanded = index.unsqueeze(-1).expand_as(source)

        def kernels(fn):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            return {
                e.name.split("tensorptr")[0]
                for e in prof.events()
                if e.device_type.name == "CUDA" and e.name.startswith("kernel_cutlass_")
            }

        ia = kernels(lambda: torch.index_add(self_t, 0, index, source))
        sa = kernels(lambda: torch.scatter_add(self_t, 0, expanded, source))
        self.assertTrue(ia, "index_add did not run the DSL kernel")
        self.assertEqual(ia, sa)
        out = torch.index_add(self_t, 0, index, source)
        ref = torch.scatter_add(self_t, 0, expanded, source)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_half_dtypes_route_to_aot(self):
        for dtype in (torch.float16, torch.bfloat16):
            self_t = torch.randn(1000, 256, device="cuda", dtype=dtype)
            index = torch.randint(0, 1000, (5000,), device="cuda")
            source = torch.randn(5000, 256, device="cuda", dtype=dtype)
            self.assertTrue(
                _ran_aot(lambda: torch.index_add(self_t, 0, index, source)), dtype
            )
            out = torch.index_add(self_t, 0, index, source)
            ref = _reference(self_t, 0, index, source)
            # Halves accumulate in-dtype on both routes; duplicate
            # indices reorder the partial sums (same tolerance rationale
            # as the scatter_add suite's _tol).
            self.assertEqual(out, ref, rtol=5e-2, atol=1e-1)

    @skipIfNoAotLib
    def test_alpha_declines_to_aten(self):
        # The kernel bakes alpha=1; a non-default alpha must decline (a
        # prelude reject on the Scalar's VALUE) and still be correct.
        self_t, index, source = self._inputs(seed=1)
        self.assertFalse(
            _ran_aot(lambda: torch.index_add(self_t, 0, index, source, alpha=2.5))
        )
        out = torch.index_add(self_t, 0, index, source, alpha=2.5)
        ref = _reference(self_t, 0, index, source, alpha=2.5)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_inplace_variant_routes_to_aot(self):
        self_t, index, source = self._inputs(seed=2)
        x = self_t.clone()
        self.assertTrue(_ran_aot(lambda: x.index_add_(0, index, source)))
        ref = _reference(self_t, 0, index, source)
        # x has accumulated twice (once inside _ran_aot); recompute clean.
        y = self_t.clone()
        y.index_add_(0, index, source)
        self.assertEqual(y, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_negative_dim_arrives_precomputed(self):
        # Schema dim=-2 on a 2D tensor wraps to 0 in the structured
        # precompute, so the AOT cond (dim != 0) accepts it.
        self_t, index, source = self._inputs(seed=3)
        self.assertTrue(_ran_aot(lambda: torch.index_add(self_t, -2, index, source)))
        out = torch.index_add(self_t, -2, index, source)
        ref = _reference(self_t, -2, index, source)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfNoAotLib
    def test_uncovered_calls_avoid_aot(self):
        self_t, index, source = self._inputs(seed=4)
        cases = {
            "dim=1": lambda: torch.index_add(
                self_t,
                1,
                torch.randint(0, 256, (64,), device="cuda"),
                torch.randn(1000, 64, device="cuda"),
            ),
            "fp64": lambda: torch.index_add(self_t.double(), 0, index, source.double()),
            "int32 index": lambda: torch.index_add(self_t, 0, index.int(), source),
            # The sidecar ABI assumes a dense index (stride 1); strided
            # index views must decline.
            "noncontig index": lambda: torch.index_add(
                self_t, 0, index.repeat_interleave(2)[::2], source
            ),
            # TMA 16B contract: a 1-col fp32 row (4B) cannot use the
            # bulk-reduce path.
            "row < 16B": lambda: torch.index_add(
                torch.randn(1000, 1, device="cuda"),
                0,
                index,
                torch.randn(5000, 1, device="cuda"),
            ),
        }
        for label, fn in cases.items():
            self.assertFalse(_ran_aot(fn), f"{label} must not route to AOT")

    @skipIfNoAotLib
    def test_deterministic_mode_avoids_aot(self):
        self_t, index, source = self._inputs(seed=5)
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            self.assertFalse(
                _ran_aot(lambda: torch.index_add(self_t, 0, index, source))
            )
        finally:
            torch.use_deterministic_algorithms(prior)

    @skipIfNoAotLib
    def test_empty_index_early_return(self):
        # The cond's early-return path: out = self, no kernel launch.
        self_t = torch.randn(100, 32, device="cuda")
        out = torch.index_add(
            self_t,
            0,
            torch.empty(0, dtype=torch.long, device="cuda"),
            torch.empty(0, 32, device="cuda"),
        )
        self.assertEqual(out, self_t)

    @skipIfNoAotLib
    def test_oob_index_traps(self):
        # The shared TMA kernel bounds-checks indices with a predicated
        # PTX trap (_ptx.trap_if_oob): an OOB index must surface as a
        # CUDA error, never silent corruption (stock aten device-asserts
        # on the same input). Subprocess: the trap poisons the context.
        code = (
            "import torch\n"
            "x = torch.zeros(100, 32, device='cuda')\n"
            "src = torch.ones(4, 32, device='cuda')\n"
            "bad = torch.tensor([1, 2, {oob}, 3], device='cuda')\n"
            "torch.index_add(x, 0, bad, src)\n"
            "torch.cuda.synchronize()\n"
            "print('NO-ERROR')\n"
        )
        for label, oob in (("too large", 100), ("negative", -1)):
            proc = subprocess.run(
                [sys.executable, "-c", code.format(oob=oob)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertNotIn(
                "NO-ERROR", proc.stdout, f"OOB index ({label}) completed silently"
            )
            self.assertNotEqual(proc.returncode, 0, label)

    @skipIfNoAotLib
    def test_disabled_context_masks_aot(self):
        self_t, index, source = self._inputs(seed=6)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertFalse(
                _ran_aot(lambda: torch.index_add(self_t, 0, index, source))
            )
        self.assertTrue(_ran_aot(lambda: torch.index_add(self_t, 0, index, source)))

    def test_covered_axes_function_directly(self):
        # Raw negative dim normalizes inside the eligibility check --
        # the Python side sees the schema call, unlike the C++ cond's
        # precomputed dim. alpha != 1 kills coverage.
        bind = _load_covered_axes()

        if not torch.cuda.is_available():
            self.skipTest("CUDA required (TMA eligibility queries the device)")
        x = torch.empty(100, 32, device="cuda")
        idx = torch.zeros(8, dtype=torch.long, device="cuda")
        src = torch.empty(8, 32, device="cuda")
        sm90 = torch.cuda.get_device_capability()[0] >= 9
        self.assertEqual(bind(x, -2, idx, src)["tma"], sm90)
        self.assertEqual(bind(x, 0, idx, src)["tma"], sm90)
        self.assertFalse(bind(x, 1, idx, src)["tma"])
        self.assertFalse(bind(x, 0, idx, src, alpha=2)["tma"])

    def test_covered_call_correct_regardless_of_routing(self):
        # With or without the AOT library, results must be correct.
        self_t, index, source = self._inputs(seed=7)
        out = torch.index_add(self_t, 0, index, source)
        ref = _reference(self_t, 0, index, source)
        self.assertEqual(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    run_tests()
