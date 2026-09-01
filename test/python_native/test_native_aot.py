# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for native-AOT kernels on aten::topk @ CUDA.

Three-layer routing under test (declaration: torch/_native/ops/topk/aot.py):

  * covered calls (fp32/bf16 on the exported grid) -> the AOT kernel in the
    structured wrapper, because the JIT layer's conds subtract AOT coverage
  * uncovered but JIT-eligible calls (off-grid fp32) -> the JIT override
  * everything else -> stock aten

Layers are isolated with the process-level switches TORCH_DISABLE_NATIVE_JIT and
TORCH_DISABLE_NATIVE_AOT in subprocesses: with the JIT layer off, a RadixSelectTopK
kernel in a profile can only come from the AOT hook. Values are checked against a
sort-based reference, which topk routing cannot affect.

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


# The exported grid (must match the manifest specs).
GRID_N = (2048, 4096, 8192, 16384)
GRID_K = (64, 128, 256)
# Enough rows to pass the full-wave perf gate on any current GPU.
M = 256

# Subprocess probe with the JIT layer disabled and the AOT hooks live. Reports,
# per case, whether the DSL kernel ran and whether values matched the reference.
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
        "RadixSelectTopK" in e.name
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
            {"dtype": "float32", "n": 3072, "k": 64},  # off-grid N
            {"dtype": "float32", "n": 4096, "k": 100},  # off-grid K
            {"dtype": "float16", "n": 4096, "k": 64},
            {"dtype": "float64", "n": 4096, "k": 64},
        ]
        results = _run_probe(cases, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        for case, r in zip(cases, results):
            self.assertFalse(r["ran_dsl"], f"{case} must not route to AOT")
            self.assertTrue(r["values_ok"], f"values mismatch for {case}")

    @skipIfNoAotLib
    def test_uncovered_fp32_served_by_jit_layer(self):
        # JIT layer live in this process, and off-grid fp32 is uncovered, so the cond
        # is not subtracted and the JIT DSL kernel runs.
        from torch.profiler import profile, ProfilerActivity

        x = torch.randn(M, 3072, device="cuda")
        torch.topk(x, 64, dim=-1)  # trigger lazy compile outside profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.topk(x, 64, dim=-1)
            torch.cuda.synchronize()
        self.assertTrue(
            any(
                "RadixSelectTopK" in e.name
                for e in prof.events()
                if e.device_type.name == "CUDA"
            )
        )

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
                "RadixSelectTopK" in e.name
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

        x = torch.empty(4, 4096)
        v = mod.covered_axes(x, 64)
        self.assertEqual(v["N"], 4096)
        self.assertEqual(v["K"], 64)
        self.assertEqual(v["dtype"], torch.float32)
        # Schema defaults come from the function signature itself.
        self.assertEqual(
            mod.covered_axes(x, 64), mod.covered_axes(x, 64, -1, True, True)
        )

    def test_manifest_covers_matches_grid(self):
        # CUDA tensors, because coverage includes the prelude's full-wave M gate, so
        # a CPU probe is always uncovered.
        from torch._native import aot_manifest

        for n in GRID_N:
            for k in GRID_K:
                x = torch.empty(M, n, dtype=torch.float32, device="cuda")
                self.assertTrue(aot_manifest.covers("topk", "CUDA", (x, k), {}))
        x = torch.empty(M, 3072, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (x, 64), {}))
        xh = torch.empty(M, 4096, dtype=torch.float16, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (xh, 64), {}))
        # Below the full-wave gate: on-grid but NOT covered (JIT keeps it).
        xs = torch.empty(4, 4096, dtype=torch.float32, device="cuda")
        self.assertFalse(aot_manifest.covers("topk", "CUDA", (xs, 64), {}))


if __name__ == "__main__":
    run_tests()
