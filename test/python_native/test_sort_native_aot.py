# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for native-AOT kernels on aten::sort @ CUDA.

Declaration: torch/_native/ops/sort/aot.py. Covered calls (narrow out= indices,
last-dim contiguous rows of at most 4096, hot key dtypes) run the embedded in-place
kernel from sort.stable's structured impl; the router declines them. Everything
else keeps the JIT override or stock aten.

The probe runs with TORCH_DISABLE_NATIVE_JIT=1, so an InplaceSort kernel in the
profile can only come from the AOT stub.
"""

import json
import os
import subprocess
import sys

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoNativeAot,
    TestCase,
)


_PROBE = r"""
import json, torch
from torch.profiler import profile, ProfilerActivity

results = []
for case in json.loads({cases!r}):
    torch.manual_seed(case["n"])
    shape = tuple(case.get("shape", (64, case["n"])))
    key = getattr(torch, case["key"])
    x = (torch.randn(shape, device="cuda") * 100).to(key)
    if case.get("transpose"):
        x = x.transpose(0, 1)
    stable = case.get("stable", True)
    out = (torch.empty_like(x), torch.empty(x.shape, device="cuda", dtype=getattr(torch, case["index"])))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if case.get("functional"):
            v, i = torch.sort(x, stable=stable, descending=case["descending"])
        else:
            v, i = torch.sort(x, stable=stable, descending=case["descending"], out=out)
        torch.cuda.synchronize()
    ran_dsl = any("InplaceSort" in e.name for e in prof.events() if e.device_type.name == "CUDA")
    rv, ri = torch.sort(x, stable=True, descending=case["descending"])
    if stable:
        ok = torch.equal(v, rv) and torch.equal(i.long(), ri)
    else:
        ok = torch.equal(v, rv) and torch.equal(x.gather(-1, i.long()), rv)
    results.append({{"ran_dsl": ran_dsl, "ok": bool(ok), "index_dtype": str(i.dtype)}})
print("PROBE_RESULTS=" + json.dumps(results))
"""


def _run_probe(cases, extra_env):
    env = dict(os.environ, QUACK_CACHE_ENABLED="0", **extra_env)
    src = _PROBE.format(cases=json.dumps(cases))
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


COVERED = [
    {"key": key, "index": index, "n": n, "descending": descending}
    for key in ("bool", "float16", "bfloat16", "float32")
    for index, n in (("uint8", 100), ("uint8", 256), ("uint16", 1000), ("int32", 4096))
    for descending in (False, True)
]
UNCOVERED = [
    # Empty rows: the JIT cond leaves them to stock, and a launch with no rows fails.
    {
        "key": "float32",
        "index": "uint16",
        "shape": (0, 1000),
        "n": 1000,
        "descending": False,
    },
    {"key": "float32", "index": "int64", "n": 1000, "descending": False},
    {
        "key": "float32",
        "index": "int16",
        "n": 1000,
        "descending": False,
        "functional": True,
    },
    {
        "key": "float32",
        "index": "int16",
        "n": 1000,
        "descending": False,
    },  # off-grid index
    {"key": "int32", "index": "int32", "n": 1000, "descending": False},  # off-grid key
    {
        "key": "float32",
        "index": "uint8",
        "n": 32,
        "descending": False,
        "stable": False,
    },  # bitonic leaf
    {
        "key": "float32",
        "index": "int32",
        "n": 5000,
        "descending": False,
    },  # multi-kernel leaf
    {
        "key": "float32",
        "index": "int32",
        "shape": (1000, 64),
        "n": 64,
        "descending": False,
        "transpose": True,
    },
]


class TestSortNativeAot(TestCase):
    hw_classification = HardwareClassification.CUDA

    @skipIfNoNativeAot("sort")
    def test_covered_calls_route_to_aot(self):
        for case, r in zip(
            COVERED, _run_probe(COVERED, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        ):
            self.assertTrue(r["ran_dsl"], f"AOT kernel did not fire for {case}")
            self.assertTrue(r["ok"], f"mismatch for {case}")
            self.assertEqual(r["index_dtype"], f"torch.{case['index']}")

    @skipIfNoNativeAot("sort")
    def test_aliasing_outs_not_covered(self):
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        x = torch.randn(64, 1000, device="cuda")
        indices = torch.empty(x.shape, dtype=torch.uint16, device="cuda")
        kwargs = {"stable": True, "dim": -1, "descending": False}
        self.assertFalse(
            coverage.covers((x,), kwargs | {"values": x, "indices": indices})
        )
        buf = torch.randn(65 * 1000, device="cuda")
        x, values = buf[: 64 * 1000].view(64, 1000), buf[1000:].view(64, 1000)
        self.assertFalse(
            coverage.covers((x,), kwargs | {"values": values, "indices": indices})
        )
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.sort(x, stable=True, out=(values, indices))
        buf = torch.empty(2 * 64 * 1000, dtype=torch.int16, device="cuda")
        x = torch.randn(64, 1000, device="cuda").bfloat16()
        values, indices = (
            buf[: x.numel()].view(torch.bfloat16).view(x.shape),
            buf[x.numel() :].view(torch.uint16).view(x.shape),
        )
        self.assertFalse(
            coverage.covers((x,), kwargs | {"values": values, "indices": indices})
        )
        ref_values, ref_indices = torch.sort(x, stable=True)
        torch.sort(x, stable=True, out=(values, indices))
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    @skipIfNoNativeAot("sort")
    def test_uncovered_calls_avoid_aot(self):
        for case, r in zip(
            UNCOVERED, _run_probe(UNCOVERED, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        ):
            self.assertFalse(r["ran_dsl"], f"{case} must not route to AOT")
            self.assertTrue(r["ok"], f"mismatch for {case}")

    @skipIfNoNativeAot("sort")
    def test_disabled_aot_keeps_stock_results(self):
        results = _run_probe(
            COVERED[:4],
            {"TORCH_DISABLE_NATIVE_JIT": "1", "TORCH_DISABLE_NATIVE_AOT": "1"},
        )
        for case, r in zip(COVERED[:4], results):
            self.assertFalse(
                r["ran_dsl"], f"{case} ran a DSL kernel with both layers off"
            )
            self.assertTrue(r["ok"], f"mismatch for {case}")

    @skipIfNoNativeAot("sort")
    @parametrize(
        "case",
        COVERED + UNCOVERED,
        name_fn=lambda c: "_".join(str(v) for v in c.values()),
    )
    def test_cpp_covers_matches_covered_axes(self, case):
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        x = torch.empty(
            case.get("shape", (64, case["n"])),
            device="cuda",
            dtype=getattr(torch, case["key"]),
        )
        if case.get("transpose"):
            x = x.transpose(0, 1)
        indices = torch.empty(
            x.shape, device="cuda", dtype=getattr(torch, case["index"])
        )
        kwargs = {
            "stable": case.get("stable", True),
            "dim": -1,
            "descending": case["descending"],
        }
        if not case.get("functional"):
            kwargs |= {"values": torch.empty_like(x), "indices": indices}
        cpp = coverage.covers((x,), kwargs)
        coverage._cpp_covers, coverage._cpp_probed = None, True
        try:
            python = coverage.covers((x,), kwargs)
        finally:
            coverage._cpp_probed = False
        self.assertEqual(cpp, python)
        self.assertEqual(cpp, case in COVERED)


instantiate_parametrized_tests(TestSortNativeAot)


if __name__ == "__main__":
    run_tests()
