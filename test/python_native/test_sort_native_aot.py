# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for native-AOT kernels on aten::sort @ CUDA.

Declaration: torch/_native/ops/sort/aot.py. Covered calls (out= uint8, uint16 or
int32 indices, last-dim contiguous rows of at most 2048, float32/bfloat16 keys) run the
embedded register or bitonic kernel from sort.stable's structured impl; the router
declines them. Everything else keeps the JIT override or stock aten.

The probe runs with TORCH_DISABLE_NATIVE_JIT=1, so a sort DSL kernel in the
profile can only come from the AOT stub.
"""

import json
import math
import os
import re
import subprocess
import sys
from unittest import mock

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
import json, math, torch
from torch.profiler import profile, ProfilerActivity

results = []
for case in json.loads({cases!r}):
    torch.manual_seed(case["n"])
    shape = tuple(case.get("shape", (64, case["n"])))
    key = getattr(torch, case["key"])
    # Element offsets of x, values and indices into their allocations.
    ox, ov, oi = case.get("offsets", (0, 0, 0))
    numel = math.prod(shape)
    x = (torch.randn(numel + ox, device="cuda") * 100).to(key)[ox:].view(shape)
    if case.get("specials") and numel:
        specials = torch.tensor([float("nan"), -float("nan"), 0.0, -0.0, float("inf"), -float("inf")])
        x.view(-1)[: numel // 2 : 7] = specials.repeat(numel // 84 + 1)[: len(range(0, numel // 2, 7))].to(key).cuda()
    if case.get("transpose"):
        x = x.transpose(0, 1)
    if case.get("unit_dim"):
        x = x.unsqueeze(-1).transpose(-1, -2)
    stable = case.get("stable", True)
    values = torch.empty(numel + ov, device="cuda", dtype=key)[ov:].view(x.shape)
    if case.get("resize_out"):
        size, offset = case["resize_out"]
        values = torch.empty(size, device="cuda", dtype=key)[offset:]
    indices = torch.empty(numel + oi, device="cuda", dtype=getattr(torch, case["index"]))[oi:].view(x.shape)
    if case.get("strided_outs"):
        values = torch.empty(shape[0], 20, device="cuda", dtype=key)[:, ::2]
        indices = torch.empty(shape[0], 20, device="cuda", dtype=indices.dtype)[:, ::2]
    if case.get("padded") == "values":
        values = torch.empty(shape[0], 2 * shape[1], device="cuda", dtype=key)[:, : shape[1]]
    elif case.get("padded") == "indices":
        indices = torch.empty(shape[0], 2 * shape[1], device="cuda", dtype=indices.dtype)[:, : shape[1]]
    if case.get("empty_outs"):
        values, indices = torch.empty(0, device="cuda", dtype=key), torch.empty(0, device="cuda", dtype=indices.dtype)
    if case.get("alias") == "x":
        values = x
    elif case.get("alias") == "x_indices":
        indices = x.view(indices.dtype)
    elif case.get("alias") == "indices":
        buf = torch.empty(2 * numel, device="cuda", dtype=key)
        values, indices = buf[:numel].view(shape), buf[numel:].view(indices.dtype).view(shape)
    if case.get("unaligned_storage"):
        # A float32 storage whose base is 8 bytes past a 16-byte boundary,
        # resized in place.
        backing = torch.empty(numel + 4, device="cuda")
        cai = {{"shape": (numel + 2,), "typestr": "<f4", "data": (backing.data_ptr() + 8, False), "version": 3}}
        # The tensor keeps the interface object, which keeps backing.
        values = torch.as_tensor(type("Iface", (), {{"backing": backing, "__cuda_array_interface__": cai}})(), device="cuda")
    out = (values, indices)
    x0 = x.clone()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if case.get("functional"):
            v, i = torch.sort(x, dim=case.get("dim", -1), stable=stable, descending=case["descending"])
        else:
            v, i = torch.sort(x, dim=case.get("dim", -1), stable=stable, descending=case["descending"], out=out)
        torch.cuda.synchronize()
    # CuTeDSL kernel names carry the module path; cub's overlap the class names.
    kernels = sorted({{e.name for e in prof.events() if e.device_type.name == "CUDA" and "torch_nativeopssortcutedsl_" in e.name}})
    ran_dsl = bool(kernels)
    dim = case.get("dim", -1)
    rv, ri = torch.sort(x0, dim=dim, stable=True, descending=case["descending"])
    # Bit patterns: NaN keys compare unequal, and -0.0 must stay -0.0.
    bits = {{1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}}[v.element_size()]
    v, rv = v.view(bits), rv.view(bits)
    if stable:
        ok = torch.equal(v, rv) and torch.equal(i.long(), ri)
    else:
        ok = torch.equal(v, rv) and torch.equal(x0.gather(dim, i.long()).view(bits), rv)
    results.append({{"ran_dsl": ran_dsl, "kernels": kernels, "ok": bool(ok), "index_dtype": str(i.dtype)}})
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
    {"key": key, "index": index, "n": n, "descending": descending, **extra}
    for key in ("bfloat16", "float32")
    for index, n, extra in (
        ("uint8", 17, {}),  # register rung 32, width 1
        ("uint8", 32, {"specials": True}),  # rung 32, full width; NaN, -0, inf
        ("uint8", 64, {}),  # rung 64, full width
        ("uint8", 100, {}),  # rung 128, full width
        ("uint8", 256, {"specials": True}),  # rung 256: radix order
        ("uint16", 512, {}),  # rung 512, full width
        ("uint16", 1024, {}),  # rung 1024, full width
        ("int32", 2048, {}),  # rung 2048, full width
        ("uint16", 1023, {}),  # odd row: width 1
        ("uint16", 1024, {"offsets": (1, 0, 0)}),  # misaligned x: width 1
        ("uint16", 1024, {"offsets": (0, 1, 0)}),  # misaligned values: width 1
        ("uint16", 1024, {"empty_outs": True}),  # outs meta resizes: full width
        # values resized in place, 16 or 32 bytes into its storage: full width
        ("uint16", 1024, {"resize_out": (64 * 1024 + 8, 8)}),
        ("uint16", 33, {"stable": False}),  # unstable past the bitonic leaf
        ("uint8", 32, {"stable": False}),  # bitonic leaf
        ("uint16", 1024, {"stable": None}),  # stable unset: unstable
        ("uint8", 100, {"shape": (100,), "dim": 0}),  # 1-D, dim given as 0
        ("uint16", 512, {"shape": (2, 32, 512), "dim": 2}),  # 3-D rows
        ("uint16", 1024, {"unit_dim": True}),  # size-1 dim with stride 1
        ("uint8", 32, {"stable": None}),  # stable unset: bitonic leaf
        ("uint16", 1024, {"strided_outs": True}),  # meta restrides them dense
    )
    for descending in (False, True)
]
UNCOVERED = [
    # values meta reallocates or resizes in place 8 bytes into its storage:
    # width 2, JIT.
    *(
        {
            "key": "float32",
            "index": "uint16",
            "n": 1024,
            "descending": False,
            "resize_out": resize_out,
        }
        for resize_out in ((2, 2), (64 * 1024 + 2, 2))
    ),
    # Aliased outs go to stock; without the alias these are covered (n 1024).
    # Every decline below is tested at a shape that is covered without it.
    *(
        {"key": key, "index": index, "n": 1024, "descending": False, "alias": alias}
        for key, index in (("float32", "int32"), ("bfloat16", "uint16"))
        for alias in ("x", "x_indices", "indices")
    ),
    # Width 1 past rung 1024: as slow as the JIT host time, JIT only.
    {"key": "float32", "index": "uint16", "n": 2047, "descending": False},
    # Not the last dim, and padded outs (rows of 1024 are covered otherwise).
    {"key": "float32", "index": "uint16", "n": 1024, "descending": False, "dim": 0},
    *(
        {
            "key": "float32",
            "index": "uint16",
            "n": 1024,
            "descending": False,
            "padded": out,
        }
        for out in ("values", "indices")
    ),
    # values resized in place in a storage based 8 bytes off 16: width 2, JIT.
    {
        "key": "float32",
        "index": "uint16",
        "n": 1024,
        "descending": False,
        "unaligned_storage": True,
    },
    # In-place radix leaf: long enough that host overhead does not matter.
    {"key": "float32", "index": "int32", "n": 2049, "descending": False},
    {"key": "float32", "index": "int32", "n": 4096, "descending": False},
    {
        "key": "float16",
        "index": "uint16",
        "n": 1024,
        "descending": False,
    },  # off-grid key
    {
        "key": "float32",
        "index": "uint16",
        "n": 1000,
        "descending": False,
    },  # width 8: JIT
    # Indices 4 bytes off: width 2, JIT.
    {
        "key": "float32",
        "index": "uint16",
        "n": 1024,
        "descending": False,
        "offsets": (0, 0, 2),
    },
    # Empty rows: the JIT cond leaves them to stock, and a launch with no rows fails.
    {
        "key": "float32",
        "index": "uint16",
        "shape": (0, 1024),
        "n": 1024,
        "descending": False,
    },
    # Rows of length 0: N == 0 declines before M = numel / N.
    {
        "key": "float32",
        "index": "uint16",
        "shape": (64, 0),
        "n": 0,
        "descending": False,
    },
    {"key": "float32", "index": "int64", "n": 1024, "descending": False},
    {
        "key": "float32",
        "index": "int16",
        "n": 1024,
        "descending": False,
        "functional": True,
    },
    {
        "key": "float32",
        "index": "int16",
        "n": 1024,
        "descending": False,
    },  # off-grid index
    {"key": "int32", "index": "int32", "n": 1024, "descending": False},  # off-grid key
    {
        "key": "bool",
        "index": "uint16",
        "n": 1024,
        "descending": False,
    },  # partition, JIT only
    {
        "key": "float32",
        "index": "int32",
        "n": 5000,
        "descending": False,
    },  # multi-kernel leaf
    {
        "key": "float32",
        "index": "int32",
        "shape": (1024, 64),
        "n": 64,
        "descending": False,
        "transpose": True,
    },
]

# Uncovered calls meta rejects (so not probed): both deciders must still agree.
UNCOVERED_META_ERRORS = [
    # uint8 indices past rung 256.
    {"key": "float32", "index": "uint8", "n": 300, "descending": False},
]


def _operands(case):
    """x and the sort kwargs of a case, outs included, as the probe builds them."""
    shape = case.get("shape", (64, case["n"]))
    ox, ov, oi = case.get("offsets", (0, 0, 0))
    numel = math.prod(shape)
    key, index = getattr(torch, case["key"]), getattr(torch, case["index"])
    x = torch.empty(numel + ox, device="cuda", dtype=key)[ox:].view(shape)
    if case.get("transpose"):
        x = x.transpose(0, 1)
    if case.get("unit_dim"):
        x = x.unsqueeze(-1).transpose(-1, -2)
    kwargs = {
        "stable": case.get("stable", True),
        "dim": case.get("dim", -1),
        "descending": case["descending"],
    }
    if not case.get("functional"):
        values = torch.empty(numel + ov, device="cuda", dtype=key)[ov:].view(x.shape)
        if case.get("resize_out"):
            size, offset = case["resize_out"]
            values = torch.empty(size, device="cuda", dtype=key)[offset:]
        indices = torch.empty(numel + oi, device="cuda", dtype=index)[oi:].view(x.shape)
        if case.get("strided_outs"):
            values = torch.empty(shape[0], 20, device="cuda", dtype=key)[:, ::2]
            indices = torch.empty(shape[0], 20, device="cuda", dtype=index)[:, ::2]
        if case.get("padded") == "values":
            values = torch.empty(shape[0], 2 * shape[1], device="cuda", dtype=key)[
                :, : shape[1]
            ]
        elif case.get("padded") == "indices":
            indices = torch.empty(shape[0], 2 * shape[1], device="cuda", dtype=index)[
                :, : shape[1]
            ]
        kwargs |= {"values": values, "indices": indices}
        if case.get("empty_outs"):
            kwargs["values"] = torch.empty(0, device="cuda", dtype=key)
            kwargs["indices"] = torch.empty(0, device="cuda", dtype=index)
        if case.get("alias") == "x":
            kwargs["values"] = x
        elif case.get("alias") == "x_indices":
            kwargs["indices"] = x.view(index)
        elif case.get("alias") == "indices":
            buf = torch.empty(2 * numel, device="cuda", dtype=key)
            kwargs["values"] = buf[:numel].view(shape)
            kwargs["indices"] = buf[numel:].view(index).view(shape)
        if case.get("unaligned_storage"):
            backing = torch.empty(numel + 4, device="cuda")
            cai = {
                "shape": (numel + 2,),
                "typestr": "<f4",
                "data": (backing.data_ptr() + 8, False),
                "version": 3,
            }
            iface = type(
                "Iface", (), {"backing": backing, "__cuda_array_interface__": cai}
            )
            kwargs["values"] = torch.as_tensor(iface(), device="cuda")
    return x, kwargs


class TestSortNativeAot(TestCase):
    hw_classification = HardwareClassification.CUDA

    @skipIfNoNativeAot("sort")
    def test_covered_calls_route_to_aot(self):
        # The stub must launch covered_axes' grid point; register kernel names
        # carry the rung and store.
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        for case, r in zip(
            COVERED, _run_probe(COVERED, {"TORCH_DISABLE_NATIVE_JIT": "1"})
        ):
            self.assertTrue(r["ran_dsl"], f"AOT kernel did not fire for {case}")
            self.assertTrue(r["ok"], f"mismatch for {case}")
            self.assertEqual(r["index_dtype"], f"torch.{case['index']}")
            x, kwargs = _operands(case)
            point = coverage._covered_axes(x, **kwargs)
            (name,) = r["kernels"]
            self.assertEqual(
                "RegisterSort" in name, point["kernel"] == "register", f"{case}"
            )
            if point["kernel"] == "register":
                launched = re.search(
                    r"RegisterSort_rung(\d+)_store(\d+)", name
                ).groups()
                expected = (str(point["rung"]), str(point["store"]))
                self.assertEqual(launched, expected, f"{case}")

    @skipIfNoNativeAot("sort")
    def test_aot_picks_the_jit_spec(self):
        # The grid point coverage matches must be the spec the JIT route would
        # compile for the same call (spec names encode kernel, rung or sort size,
        # vector width, dtypes and direction).
        from torch._native import aot_manifest
        from torch._native.ops.sort import cutedsl_inplace, cutedsl_register

        modules = {"register": cutedsl_register, "inplace": cutedsl_inplace}
        coverage = aot_manifest.get_coverage("sort", "CUDA")
        for case in COVERED:
            x, kwargs = _operands(case)
            point = coverage._covered_axes(x, **kwargs)
            module = modules[point["kernel"]]
            specs = []

            def record(*args, make_spec=module.make_spec, **kw):
                specs.append(make_spec(*args, **kw))
                return specs[-1]

            aot_enabled = torch._native.aot_enabled()
            torch._native.set_aot_enabled(False)
            try:
                # Only the spec matters: skip compiling and launching it.
                with (
                    mock.patch.object(module, "make_spec", record),
                    mock.patch.object(
                        module, "_compile", lambda spec: lambda *args: None
                    ),
                ):
                    torch.sort(
                        x,
                        out=(kwargs["values"], kwargs["indices"]),
                        **{k: kwargs[k] for k in ("stable", "dim", "descending")},
                    )
            finally:
                torch._native.set_aot_enabled(aot_enabled)
            jit = {k: v for k, v in specs[-1].items() if k != "arch"}
            self.assertEqual(module.spec_name(jit), module.spec_name(point), f"{case}")

    @skipIfNoNativeAot("sort")
    def test_out_of_range_dim_not_covered(self):
        # Python and C++ must both decline it (C++ maybe_wrap_dim would throw).
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        x, kwargs = _operands(
            {"key": "float32", "index": "uint16", "n": 1024, "descending": False}
        )
        kwargs["dim"] = 3
        self.assertFalse(coverage.covers((x,), kwargs))
        self.assertFalse(coverage._covered_axes(x, **kwargs)["eligible"])

    @skipIfNoNativeAot("sort")
    def test_overlapping_outs_not_covered(self):
        # Full aliasing is in UNCOVERED; a partial overlap makes stock raise.
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        kwargs = {"stable": True, "dim": -1, "descending": False}
        indices = torch.empty(64, 1024, dtype=torch.uint16, device="cuda")
        buf = torch.randn(65 * 1024, device="cuda")
        x, values = buf[: 64 * 1024].view(64, 1024), buf[1024:].view(64, 1024)
        twin = torch.randn(65 * 1024, device="cuda")[1024:].view(64, 1024)
        self.assertTrue(
            coverage.covers((x,), kwargs | {"values": twin, "indices": indices})
        )
        self.assertFalse(
            coverage.covers((x,), kwargs | {"values": values, "indices": indices})
        )
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.sort(x, stable=True, out=(values, indices))

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
        COVERED + UNCOVERED + UNCOVERED_META_ERRORS,
        name_fn=lambda c: "_".join(f"{k}_{v}" for k, v in c.items()),
    )
    def test_cpp_covers_matches_covered_axes(self, case):
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage("sort", "CUDA")
        x, kwargs = _operands(case)
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
