# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for native-AOT kernels on aten::topk @ CUDA.

Three-layer routing under test (declaration: torch/_native/ops/topk/aot.py):

  * covered calls -> the AOT kernel in the structured wrapper, because the JIT
    layer's conds subtract AOT coverage
  * JIT-only exact-N register calls and covered calls without AOT -> the JIT override
  * everything else -> stock aten

Layers are isolated with the process-level switches TORCH_DISABLE_NATIVE_JIT and
TORCH_DISABLE_NATIVE_AOT in subprocesses: with the JIT layer off, a top-k DSL kernel
in a profile can only come from the AOT hook. Values are checked against a sort-based
reference, which topk routing cannot affect.

Tests needing the AOT kernels skip unless this build embedded them; the
correctness tests run everywhere, since covered calls must be correct through stock
aten when the artifacts are absent.
"""

import json
import os
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


def skipIfNoJitTopk(fn):
    """Both top-k routes require Hopper or newer."""
    capability = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    )
    return unittest.skipUnless(capability[0] >= 9, "JIT topk override needs sm_90+")(fn)


# Representative points in the original measured grid.
GRID_N = (2048, 4096, 8192, 16384)
GRID_K = (64, 128, 256)
# Enough rows to pass the full-wave perf gate on any current GPU.
M = 256
JIT_ONLY_REGISTER = ((16, 2048), (32, 256))

# Subprocess probe with the JIT layer disabled and the AOT hooks live. Both layers
# launch the same CuTeDSL kernel, so the name in the profile says a DSL kernel ran
# while the environment is what makes it the AOT one; aten shows mbtopk instead.
_PROBE = r"""
import json, torch
from torch.profiler import profile, ProfilerActivity

results = []
for case in json.loads({cases!r}):
    dtype = getattr(torch, case["dtype"])
    kwargs = case.get("kwargs", {{}})
    torch.manual_seed(case["n"] * 31 + case["k"])
    x = torch.randn({m}, case["n"], device="cuda", dtype=dtype)
    if case.get("det"):
        torch.use_deterministic_algorithms(True)
    out = None
    if case.get("out_variant"):
        out = (
            torch.empty({m}, case["k"], device="cuda", dtype=dtype),
            torch.empty({m}, case["k"], device="cuda", dtype=torch.int64),
        )
        kwargs = dict(kwargs, out=out)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        v, i = torch.topk(x, case["k"], dim=-1, **kwargs)
        torch.cuda.synchronize()
    ran_dsl = any(
        "RadixSelectTopK" in e.name or "RegisterTopK" in e.name
        for e in prof.events()
        if e.device_type.name == "CUDA"
    )
    ref_v = torch.sort(x, dim=-1, descending=True).values[..., : case["k"]]
    values_ok = bool(torch.equal(v, ref_v))
    gather_ok = bool(torch.equal(x.gather(-1, i), v))
    if case.get("det"):
        torch.use_deterministic_algorithms(False)
    results.append(
        {{"ran_dsl": ran_dsl, "values_ok": values_ok, "gather_ok": gather_ok,
          "index_dtype": str(i.dtype)}}
    )
print("PROBE_RESULTS=" + json.dumps(results))
"""


def _run_probe(cases, extra_env):
    env = dict(os.environ, **extra_env)
    src = _PROBE.format(cases=json.dumps(cases), m=M)
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        env=env,
        timeout=1200,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])
    line = next(
        line for line in proc.stdout.splitlines() if line.startswith("PROBE_RESULTS=")
    )
    return json.loads(line[len("PROBE_RESULTS=") :])


class TestNativeAotTopKDeclaration(TestCase):
    def test_dispatch_alignment_is_radix_only(self):
        from torch._native.ops.topk import aot

        prelude = aot.cpp_dispatch_prelude()
        self.assertNotIn("N % 4", prelude)
        self.assertNotIn("getCurrentDeviceProperties", prelude)
        self.assertIn("_naot_props->major", prelude)
        self.assertIn("_naot_props->multiProcessorCount", prelude)

        register = aot.cpp_dispatch(
            {
                "kernel": "register",
                "dtype": "float32",
                "K": 16,
                "N_rung": "64_128_256_512_1024",
            }
        )
        radix = aot.cpp_dispatch(
            {
                "kernel": "radix",
                "dtype": "float32",
                "K": 64,
                "deterministic": False,
                "fixed_vec_iters": None,
            }
        )
        self.assertNotIn("N % 4", register)
        self.assertIn("N % 4 == 0", radix)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestNativeAotTopK(TestCase):
    @skipIfNoAotLib
    def test_covered_grid_routes_to_aot(self):
        cases = [
            {"dtype": dtype, "n": n, "k": k}
            for dtype in ("float32", "bfloat16")
            for n in GRID_N
            for k in GRID_K
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertTrue(r["ran_dsl"], f"AOT kernel did not fire for {case}")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")
            self.assertEqual(r["index_dtype"], "torch.int64")

    @skipIfNoAotLib
    def test_register_grid_routes_to_aot(self):
        cases = [
            {"dtype": "float32", "n": n, "k": 16} for n in (64, 128, 256, 512, 1024)
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertTrue(
                r["ran_dsl"], f"AOT register kernel did not fire for {case}"
            )
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")
            self.assertEqual(r["index_dtype"], "torch.int64")

    @skipIfNoAotLib
    def test_dynamic_n_and_work_rungs_route_to_aot(self):
        shapes = (
            (64, 3072),
            (64, 4100),
            (64, 4356),
            (64, 4612),
            (64, 4868),
            (512, 4096),
            (512, 5120),
            (512, 6140),
            (512, 6144),
            (1024, 32772),
            (1024, 34820),
            (1024, 36860),
        )
        cases = [
            {"dtype": dtype, "n": n, "k": k, "det": deterministic}
            for dtype in ("float32", "bfloat16")
            for deterministic in (False, True)
            for k, n in shapes
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertTrue(r["ran_dsl"], f"AOT kernel did not fire for {case}")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")

    @skipIfNoAotLib
    def test_deterministic_mode_routes_to_aot_bit_exact(self):
        # Det mode is on the grid, so its kernel must fire and match aten bit-exactly.
        # Probe values are torch.randn; ties are exercised in the next test.
        cases = [
            {"dtype": "float32", "n": 4096, "k": 64, "det": True},
            {"dtype": "bfloat16", "n": 2048, "k": 128, "det": True},
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertTrue(r["ran_dsl"], f"AOT det kernel did not fire for {case}")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")

    @skipIfNoAotLib
    def test_deterministic_ties_bit_exact(self):
        # Tie-heavy input, so det-mode indices must match aten's exactly. The
        # reference runs under disabled(), or it would come from the route under test.
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(11)
                x = torch.randint(0, 50, (M * 8, 4096), device="cuda").to(dtype)
                v, i = torch.topk(x, 64)
                with torch.backends.python_native.cutedsl.disabled():
                    rv, ri = torch.topk(x, 64)
                self.assertTrue(torch.equal(v, rv), f"values differ ({dtype})")
                self.assertTrue(torch.equal(i, ri), f"indices differ ({dtype})")
        finally:
            torch.use_deterministic_algorithms(prior)

    @skipIfNoAotLib
    def test_out_variant_routes_to_aot(self):
        results = _run_probe(
            [{"dtype": "float32", "n": 4096, "k": 64, "out_variant": True}],
            {"TORCH_DISABLE_NATIVE_JIT": "1"},
        )
        self.assertTrue(results[0]["ran_dsl"])
        self.assertTrue(results[0]["values_ok"])

    @skipIfNoAotLib
    def test_uncovered_calls_avoid_aot(self):
        cases = [
            *({"dtype": "float32", "n": n, "k": k} for k, n in JIT_ONLY_REGISTER),
            {"dtype": "float32", "n": 124, "k": 64},  # below every radix range
            {"dtype": "float32", "n": 4101, "k": 64},  # not vector-aligned
            {"dtype": "float32", "n": 4096, "k": 100},  # off-grid K
            {"dtype": "float32", "n": 512, "k": 32},  # off-grid register N
            {"dtype": "bfloat16", "n": 256, "k": 16},  # register is fp32-only
            {"dtype": "float16", "n": 4096, "k": 64},
            {"dtype": "float64", "n": 4096, "k": 64},
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertFalse(r["ran_dsl"], f"{case} must not route to AOT")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")

    @skipIfNoJitTopk
    def test_aot_coverage_is_subset_of_jit_eligibility(self):
        from torch._native import aot_manifest
        from torch._native.ops.topk import cutedsl_impl
        from torch._native.ops.topk.aot import _radix_min_n

        major = torch.cuda.get_device_capability()[0]
        cases = [
            (torch.float32, 16, 64),
            (torch.float32, 16, 2048),
            (torch.float32, 32, 256),
            (torch.float32, 32, 512),
            (torch.bfloat16, 16, 256),
        ]
        for dtype_name, dtype in (
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
        ):
            for k in (64, 128, 256, 512, 1024):
                min_n = _radix_min_n(dtype_name, k, major)
                cases.extend(
                    (
                        (dtype, k, min_n - 4),
                        (dtype, k, min_n),
                        (dtype, k, min_n + 4),
                    )
                )
        for dtype, k, n in cases:
            with self.subTest(dtype=dtype, k=k, n=n):
                x = torch.empty(M, n, dtype=dtype, device="cuda")
                jit_eligible = cutedsl_impl._eligible(x, k, -1, True, True)
                aot_covered = aot_manifest.covers("topk", "CUDA", (x, k), {})
                if dtype == torch.float32 and (k, n) in JIT_ONLY_REGISTER:
                    self.assertTrue(jit_eligible)
                    self.assertFalse(aot_covered)
                else:
                    self.assertEqual(aot_covered, jit_eligible)

    @skipIfNoJitTopk
    def test_exact_n_register_cases_route_to_jit(self):
        cases = [{"dtype": "float32", "n": n, "k": k} for k, n in JIT_ONLY_REGISTER]
        results = _run_probe(cases, {})
        for case, r in zip(cases, results):
            self.assertTrue(r["ran_dsl"], f"JIT kernel did not fire for {case}")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")

    @skipIfNoJitTopk
    def test_shared_policy_cases_route_to_jit_without_aot(self):
        cases = [
            *({"dtype": "float32", "n": n, "k": 16} for n in (64, 128, 256, 512, 1024)),
            {"dtype": "float32", "n": 4100, "k": 64},
            {"dtype": "float32", "n": 5120, "k": 512, "det": True},
            {"dtype": "bfloat16", "n": 2048, "k": 64},
            {
                "dtype": "bfloat16",
                "n": 2048,
                "k": 64,
                "out_variant": True,
            },
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_AOT": "1"})
        for case, r in zip(cases, results):
            self.assertTrue(r["ran_dsl"], f"JIT kernel did not fire for {case}")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")
            self.assertTrue(r["gather_ok"], f"gather mismatch for {case}")

    @skipIfNoAotLib
    def test_disabled_context_masks_aot_in_process(self):
        # cutedsl.disabled() flips the native-AOT Context switch as well as the JIT
        # layer, so no DSL kernel may run inside the block.
        from torch import _native
        from torch.profiler import profile, ProfilerActivity

        def ran_dsl():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                torch.topk(x, 64, dim=-1)
                torch.cuda.synchronize()
            return any(
                "RadixSelectTopK" in e.name or "RegisterTopK" in e.name
                for e in prof.events()
                if e.device_type.name == "CUDA"
            )

        x = torch.randn(M, 4096, device="cuda")
        pn = torch.backends.python_native
        self.assertTrue(_native.aot_enabled())
        with pn.cutedsl.disabled():
            self.assertFalse(_native.aot_enabled())
            self.assertFalse(ran_dsl())
        self.assertTrue(_native.aot_enabled())
        self.assertTrue(ran_dsl())

    def test_covered_call_correct_regardless_of_routing(self):
        # Correct whichever layer serves it, including stock aten with no AOT lib.
        torch.manual_seed(2)
        x = torch.randn(M, 4096, device="cuda")
        v, i = torch.topk(x, 64, dim=-1)
        ref_v = torch.sort(x, dim=-1, descending=True).values[..., :64]
        self.assertEqual(v, ref_v, atol=0, rtol=0)
        self.assertEqual(x.gather(-1, i), v, atol=0, rtol=0)

    def test_covered_call_correct_without_aot_lib(self):
        # Null-hook degradation, forced: both layers off in a subprocess.
        results = _run_probe(
            [{"dtype": "float32", "n": 4096, "k": 64}],
            {"TORCH_DISABLE_NATIVE_JIT": "1", "TORCH_DISABLE_NATIVE_AOT": "1"},
        )
        self.assertFalse(results[0]["ran_dsl"])
        self.assertTrue(results[0]["values_ok"])

    def test_covered_axes_function_directly(self):
        # covered_axes() is plain Python, loaded by file path since the module is
        # stdlib-only at import.
        import importlib.util
        import os

        path = os.path.join(
            os.path.dirname(torch.__file__), "_native", "ops", "topk", "aot.py"
        )
        spec = importlib.util.spec_from_file_location("topk_aot_t", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        x = torch.empty(M, 4100, device="cuda")
        v = mod.covered_axes(x, 64)
        self.assertEqual(v["kernel"], "radix")
        self.assertEqual(v["N"], None)
        self.assertEqual(v["K"], 64)
        self.assertEqual(v["dtype"], torch.float32)
        self.assertEqual(v["scalar_tail_iters"], None)
        self.assertEqual(v["fixed_vec_iters"], None)
        self.assertTrue(v["eligible"])
        register = mod.covered_axes(torch.empty(M, 256, device="cuda"), 16)
        self.assertEqual(register["kernel"], "register")
        self.assertEqual(register["N"], None)
        self.assertEqual(register["N_rung"], "64_128_256_512_1024")
        self.assertTrue(register["eligible"])
        base = torch.empty(M, 4096, device="cuda")
        cow = base._lazy_clone()
        data_ptr = cow.const_data_ptr()
        uncovered = mod.covered_axes(cow, 100)
        self.assertFalse(uncovered["eligible"])
        self.assertTrue(torch._C._is_cow_tensor(cow))
        self.assertEqual(cow.const_data_ptr(), data_ptr)
        # Schema defaults come from the function signature itself.
        self.assertEqual(
            mod.covered_axes(x, 64), mod.covered_axes(x, 64, -1, True, True)
        )

    def test_covered_axes_selects_work_rungs(self):
        import importlib.util
        import os

        path = os.path.join(
            os.path.dirname(torch.__file__), "_native", "ops", "topk", "aot.py"
        )
        spec = importlib.util.spec_from_file_location("topk_aot_work_t", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for n, k, expected in (
                (4100, 64, (4, None)),
                (5120, 512, (4, 2)),
                (6144, 512, (4, None)),
                (36860, 1024, (4, None)),
            ):
                x = torch.empty(M, n, device="cuda")
                axes = mod.covered_axes(x, k)
                self.assertEqual(
                    (axes["scalar_tail_iters"], axes["fixed_vec_iters"]), expected
                )
                self.assertTrue(axes["eligible"])
        finally:
            torch.use_deterministic_algorithms(prior)

    def test_flags_the_stub_declines_are_uncovered(self):
        # The stub takes only dim=last, largest and sorted; coverage must agree, or such
        # a call declines on both routes and lands on stock aten instead of the JIT one.
        from torch._native import aot_manifest

        x = torch.empty(M, 4096, dtype=torch.float32, device="cuda")
        self.assertTrue(aot_manifest.covers("topk", "CUDA", (x, 64), {}))
        for kwargs in ({"dim": 0}, {"largest": False}, {"sorted": False}):
            with self.subTest(**kwargs):
                self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 64), kwargs))
        # dim spelled as the last axis is the covered case, however it is written.
        self.assertTrue(aot_manifest.covers("topk", "CUDA", (x, 64), {"dim": 1}))

    def test_manifest_covers_matches_grid(self):
        # CUDA tensors, because coverage includes the prelude's full-wave M gate, so
        # a CPU probe is always uncovered.
        from torch._native import aot_manifest

        for n in GRID_N:
            for k in GRID_K:
                x = torch.empty(M, n, dtype=torch.float32, device="cuda")
                self.assertTrue(aot_manifest.covers("topk", "CUDA", (x, k), {}))
        for k, n in (
            (16, 64),
            (64, 3072),
            (64, 4100),
            (512, 4096),
            (512, 6140),
            (1024, 32772),
            (1024, 36860),
        ):
            x = torch.empty(M, n, dtype=torch.float32, device="cuda")
            self.assertTrue(aot_manifest.covers("topk", "CUDA", (x, k), {}))
        for k, n in JIT_ONLY_REGISTER:
            x = torch.empty(M, n, dtype=torch.float32, device="cuda")
            self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, k), {}))
        from torch._native.ops.topk.aot import _radix_min_n

        major = torch.cuda.get_device_capability()[0]
        min_n = _radix_min_n("float32", 64, major)
        x = torch.empty(M, min_n - 4, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 64), {}))
        x = torch.empty(M, 4101, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 64), {}))
        x = torch.empty(M, 4092, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 512), {}))
        x = torch.empty(M, 32764, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 1024), {}))
        xh = torch.empty(M, 4096, dtype=torch.float16, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (xh, 64), {}))
        # Below the full-wave gate: on-grid but NOT covered (JIT keeps it).
        xs = torch.empty(4, 4096, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (xs, 64), {}))


if __name__ == "__main__":
    run_tests()
