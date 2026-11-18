# Owner(s): ["module: dsl-native-ops"]

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfNoNativeAot,
    subtest,
    TestCase,
)
from torchgen.native_aot_decl import cc_of, load_declarations
from torchgen.native_aot_spec_grid import expand_specs


_DECL = (
    Path(__file__).parents[2] / "torch" / "_native" / "ops" / "reductions" / "aot.py"
)

_AOT_PROBE = r"""
import json
import torch
from torch.profiler import profile, ProfilerActivity

results = []
for n in (1536, 131072):
    torch.manual_seed(n)
    x = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = x.sum(dim=-1, dtype=torch.float32)
        torch.cuda.synchronize()
    launches = sum(
        "torch_nativeopsreductions" in event.name
        for event in prof.events()
        if event.device_type.name == "CUDA"
    )
    expected = x.to(torch.float64).sum(dim=-1).to(torch.float32)
    correct = torch.allclose(out, expected, atol=2e-2, rtol=1e-4)
    results.append((launches, correct))
print("RESULTS=" + json.dumps(results))
"""

# Calls through the embedded stubs, functionally and with out=, checked against ATen:
# label -> (op, input, dim, keepdim, dtype=, out= dtype, PYTORCH_NATIVE_INNER_TREE).
# Prod inputs stay near 1 so that products stay finite.
_PATH_PROBE = r"""
import json
import os
import torch
from torch.profiler import profile, ProfilerActivity

f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
TOL = {f16: 1e-2, bf16: 1e-2, f32: 1e-4}


def run(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type.name == "CUDA"]
    return out, sum("torch_nativeopsreductions" in e.name for e in events)


def randn(*shape, dtype=f32):
    return torch.randn(*shape, device="cuda").to(dtype)


def near_one(*shape, dtype=f32):
    return (1 + 0.01 * torch.randn(*shape, device="cuda")).to(dtype)


def channels_last():
    return randn(2, 64, 8, 8).to(memory_format=torch.channels_last)


cases = {
    "sum_t32_b128": ("sum", lambda: randn(8, 1024), -1, False, None, None, "0"),
    "mean_fp16": ("mean", lambda: randn(8, 1024, dtype=f16), -1, False, None, None, "0"),
    "amax_t64_b128": ("amax", lambda: randn(8, 4096), -1, False, None, None, "0"),
    "sum_t128_b128": ("sum", lambda: randn(8, 7168, dtype=bf16), -1, False, None, None, "0"),
    "prod": ("prod", lambda: near_one(8, 1024), -1, False, None, None, "0"),
    "amin_xcta": ("amin", lambda: randn(4, 131072, dtype=bf16), -1, False, None, None, "0"),
    "mean_xcta": ("mean", lambda: randn(4, 131072, dtype=bf16), -1, False, None, None, "0"),
    "amax_xcta": ("amax", lambda: randn(4, 131072), -1, False, None, None, "0"),
    "mean_ragged": ("mean", lambda: randn(8, 1001, dtype=bf16), -1, False, None, None, "0"),
    "sum_ragged_t64": ("sum", lambda: randn(8, 4097, dtype=bf16), -1, False, None, None, "0"),
    "sum_pitched": ("sum", lambda: randn(8, 1040)[:, :1024], -1, False, None, None, "0"),
    "amax_pitched_ragged": ("amax", lambda: randn(8, 1032)[:, :1001], -1, False, None, None, "0"),
    "sum_two_dims": ("sum", lambda: randn(2, 4, 4096), [-2, -1], False, None, None, "0"),
    "mean_permuted": ("mean", lambda: randn(64, 1024).t(), 0, False, None, None, "0"),
    "sum_channels_last": ("sum", channels_last, 1, False, None, None, "0"),
    "amin_keepdim": ("amin", lambda: randn(8, 1024), -1, True, None, None, "0"),
    "sum_reduce_all": ("sum", lambda: randn(1024), -1, False, None, None, "0"),
    "mean_widened": ("mean", lambda: randn(8, 1024, dtype=bf16), -1, False, f32, None, "0"),
    "prod_widened": ("prod", lambda: near_one(8, 1024, dtype=f16), -1, False, f32, None, "0"),
    "sum_out_fp32": ("sum", lambda: randn(8, 1024, dtype=f16), -1, False, None, f32, "0"),
    "sum_ordered": ("sum", lambda: randn(4, 32), -1, False, None, None, "1"),
    "prod_ordered": ("prod", lambda: near_one(4, 32, dtype=f16), -1, False, None, None, "1"),
    "mean_ordered": ("mean", lambda: randn(4, 32768), -1, False, None, None, "1"),
}
results = {}
for label, (name, make, dim, keepdim, dtype, out_dtype, order) in cases.items():
    os.environ["PYTORCH_NATIVE_INNER_TREE"] = order
    torch.manual_seed(0)
    x = make()
    op = getattr(torch, name)
    kw = {} if dtype is None else {"dtype": dtype}
    out_kw = kw if out_dtype is None else {"dtype": out_dtype}
    with torch.backends.python_native.cutedsl.disabled():
        ref, out_ref = op(x, dim, keepdim, **kw), op(x, dim, keepdim, **out_kw)
    got, launches = run(lambda: op(x, dim, keepdim, **kw))
    out = torch.empty_like(out_ref)
    _, out_launches = run(lambda: op(x, dim, keepdim, **kw, out=out))
    ok = []
    for t, want in ((got, ref), (out, out_ref)):
        tol = 0 if name in ("amax", "amin") else TOL[want.dtype]
        ok.append(t.dtype == want.dtype and torch.allclose(t, want, atol=tol, rtol=tol))
    results[label] = [launches, out_launches, *ok]
print("RESULTS=" + json.dumps(results))
"""
# Row and ragged callables launch once, cross-CTA ones twice, and the stubs decline
# ordered requests.
_PATH_LAUNCHES = {
    "sum_t32_b128": 1,
    "mean_fp16": 1,
    "amax_t64_b128": 1,
    "sum_t128_b128": 1,
    "prod": 1,
    "amin_xcta": 2,
    "mean_xcta": 2,
    "amax_xcta": 2,
    "mean_ragged": 1,
    "sum_ragged_t64": 1,
    "sum_pitched": 1,
    "amax_pitched_ragged": 1,
    "sum_two_dims": 1,
    "mean_permuted": 1,
    "sum_channels_last": 1,
    "amin_keepdim": 1,
    "sum_reduce_all": 1,
    "mean_widened": 1,
    "prod_widened": 1,
    "sum_out_fp32": 1,
    "sum_ordered": 0,
    "prod_ordered": 0,
    "mean_ordered": 0,
}
_OPS = [d.ATEN_OP for d in load_declarations(str(_DECL))]


def _covers_inputs(device):
    """Views across the row classes, the ragged and cross-CTA paths, and declines."""
    flat = torch.randn(8 * 1024 + 1, device=device)
    nchw = torch.randn(2, 64, 8, 8, device=device)
    return [
        torch.randn(4, 32, device=device),  # t32_b128
        torch.randn(8, 1024, device=device),  # t32_b128
        torch.randn(8, 4096, device=device, dtype=torch.bfloat16),  # t64_b128
        torch.randn(8, 7168, device=device, dtype=torch.bfloat16),  # t128_b128
        torch.randn(2, 4, 4096, device=device),  # t256_b256
        torch.randn(256, device=device),  # t64_b64 as a reduce-all
        torch.randn(2, 32768, device=device),  # t256_b256
        torch.randn(2, 131072, device=device, dtype=torch.bfloat16),  # xcta
        torch.randn(4, 1001, device=device),  # ragged rows
        torch.randn(4, 4097, device=device, dtype=torch.bfloat16),  # t64 ragged
        torch.randn(4, 9001, device=device),  # t256 ragged
        torch.randn(1, 1001, device=device),  # a single ragged row: JIT reduce_all
        torch.randn(4, 258, device=device, dtype=torch.float16),  # even, too narrow
        torch.randn(4, 1026, device=device, dtype=torch.float16),  # even ragged
        torch.randn(4, 259, device=device, dtype=torch.float16)[:, :258],  # odd pitch
        torch.randn(8, 1040, device=device)[:, :1024],  # pitched rows
        torch.randn(8, 1032, device=device)[:, :1001],  # pitched ragged rows
        torch.randn(4, 8, 1024, device=device)[:, :4],  # kept dims that do not coalesce
        torch.randn(8, 1, 1024, device=device),  # a size-1 dim
        nchw.to(memory_format=torch.channels_last),
        torch.randn(64, 32, device=device).t(),  # dense permuted
        torch.randn(1024, device=device).expand(8, 1024),  # overlapping rows
        flat[1:].view(8, 1024),  # misaligned
        torch.randn(4096, 32, device=device),  # narrow rows, not a reduce-all
        # A row over 192 KiB that is not a multiple of 8192 elements.
        torch.randn(2, 98312, device=device, dtype=torch.bfloat16),
        torch.randn(8, 1024, device=device, dtype=torch.float16),  # fp16
    ]


class TestReductionAOT(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_archs_follow_override_majors(self):
        # The embedded stubs run even where the overrides decline, so they must cover
        # the same compute-capability majors.
        from torch._native.ops.reductions import _aot_data, overrides

        majors = {cc_of(target)[0] for target in _aot_data.ARCHS}
        self.assertEqual(majors, set(overrides._ARCH_MAJORS))

    def test_core_family_grid(self):
        declarations = load_declarations(str(_DECL))
        self.assertEqual(
            [declaration.ATEN_OP for declaration in declarations],
            ["sum.dim_IntList", "mean.dim", "amax", "amin", "prod.dim_int"],
        )
        for declaration in declarations:
            grid = expand_specs(declaration.kernel_precompile_grid())
            has_dtype = declaration.ATEN_OP in {
                "sum.dim_IntList",
                "mean.dim",
                "prod.dim_int",
            }
            # 10 points per dtype pair; dtype= adds only FP16->FP32 and BF16->FP32.
            self.assertEqual(len(grid), 50 if has_dtype else 30)
            paths = {point["path"] for point in grid}
            self.assertEqual(paths, {"row", "ragged", "xcta"})
            same = {(dtype, dtype) for dtype in ("float16", "bfloat16", "float32")}
            widened = {("float16", "float32"), ("bfloat16", "float32")}
            self.assertEqual(
                {(point["dtype"], point["out_dtype"]) for point in grid},
                same | widened if has_dtype else same,
            )
            self.assertEqual(
                {point["row_class"] for point in grid if point["path"] == "row"},
                {
                    "t32_b128",
                    "t64_b64",
                    "t64_b128",
                    "t128_b128",
                    "t256_b256",
                },
            )
            self.assertEqual(
                {point["row_class"] for point in grid if point["path"] == "ragged"},
                {"t32_b128", "t64_b128", "t128_b128", "t256_b256"},
            )


class TestReductionAOTCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize(
        "probe,expected",
        [
            subtest(
                (_AOT_PROBE, [[1, True], [2, True]]),
                name="sum",
                decorators=[skipIfNoNativeAot("sum.dim_IntList")],
            ),
            subtest(
                (
                    _PATH_PROBE,
                    {k: [n, n, True, True] for k, n in _PATH_LAUNCHES.items()},
                ),
                name="paths",
                decorators=[skipIfNoNativeAot(op) for op in _OPS],
            ),
        ],
    )
    def test_embedded_callables_own_reduction_launches(self, device, probe, expected):
        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "0"}
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=dict(os.environ, TORCH_DISABLE_NATIVE_JIT="1", **gates),
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        line = next(
            line for line in proc.stdout.splitlines() if line.startswith("RESULTS=")
        )
        self.assertEqual(json.loads(line.removeprefix("RESULTS=")), expected)

    @skipIfNoCuteDSL
    @parametrize("op", _OPS)
    @parametrize("gate", ["", "PYTORCH_NATIVE_INNER_TREE", "PYTORCH_SUM_INNER_TREE"])
    def test_cpp_covers_matches_python_coverage(self, device, op, gate):
        """The generated covers_<op> predicate decides exactly what covered_axes does."""
        from torch._native import aot_manifest

        cpp = aot_manifest.get_coverage(op, "CUDA")
        if cpp is None or not cpp.is_available(torch.device(device)):
            self.skipTest(f"no native-AOT kernels for {op} on {device}")
        self.assertIsNotNone(cpp._resolve_cpp_covers())
        decl = next(d for d in load_declarations(str(_DECL)) if d.ATEN_OP == op)
        grid = expand_specs(decl.kernel_precompile_grid())
        python = aot_manifest._Coverage(op, decl.covered_axes, grid)
        python._cpp_probed = True  # leaves _cpp_covers unset: the Python path decides

        inputs = _covers_inputs(device)
        if op == "prod.dim_int":
            dims = [-1, 0]
        elif op in ("amax", "amin"):
            dims = [[-1], [-2, -1], [], [0]]
        else:
            dims = [-1, [-1], [-2, -1], None, [0]]
        has_dtype = op not in ("amax", "amin")
        dtypes = [None, torch.float32, torch.float16] if has_dtype else [None]
        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "0"}
        if gate:
            gates[gate] = "1"
        covered = 0
        cases = itertools.product(inputs, dims, (False, True), dtypes)
        with mock.patch.dict(os.environ, gates):
            for x, dim, keepdim, dtype in cases:
                if x.dim() == 1 and dim in ([-2, -1], [0], 0):
                    continue
                args, kwargs = (x, dim, keepdim), {"dtype": dtype} if dtype else {}
                want = python.covers(args, kwargs)
                msg = f"{tuple(x.shape)} {x.dtype} {dim=} {keepdim=} {dtype=}"
                self.assertEqual(cpp.covers(args, kwargs), want, msg=msg)
                covered += want
        sum_ordered = op in ("sum.dim_IntList", "prod.dim_int")
        if gate == "PYTORCH_NATIVE_INNER_TREE" or (sum_ordered and gate):
            # There are no inner-tree callables, so ordered requests stay JIT.
            self.assertEqual(covered, 0)
        else:
            self.assertGreater(covered, 0)

    @skipIfNoCuteDSL
    @parametrize("op", _OPS)
    def test_cpp_coverage_matches_what_the_stub_serves(self, device, op):
        """covers_<op> is what drops a call's JIT route, so a call it claims but the stub
        declines loses both accelerated routes: the two must agree call for call."""
        from torch._native import aot_manifest
        from torch.profiler import profile, ProfilerActivity

        cpp = aot_manifest.get_coverage(op, "CUDA")
        if cpp is None or not cpp.is_available(torch.device(device)):
            self.skipTest(f"no native-AOT kernels for {op} on {device}")
        name, _, overload = op.partition(".")
        aten = getattr(torch.ops.aten, name)
        out_op = {"sum": "IntList_out", "prod": "int_out"}.get(name, "out")
        functional = getattr(aten, overload or "default")
        with_out = getattr(aten, out_op)
        dim = -1 if op == "prod.dim_int" else [-1]
        calls = [((x, dim, False), {}) for x in _covers_inputs(device)]
        # out= dtypes the stubs serve (the input's, or widened) and decline (narrowed),
        # and a non-contiguous out=. amax and amin outputs keep the input dtype.
        x16 = torch.randn(8, 1024, device=device, dtype=torch.float16)
        out_dtypes = (torch.float16, torch.float32, torch.bfloat16)
        for out_dtype in out_dtypes if op not in ("amax", "amin") else out_dtypes[:1]:
            out = torch.empty(8, device=device, dtype=out_dtype)
            calls.append(((x16, dim, False), {"out": out}))
        strided = torch.empty(8, 2, device=device, dtype=torch.float16)[:, 0]
        calls.append(((x16, dim, False), {"out": strided}))

        def fired(fn, args, kwargs):
            # With the JIT overrides deregistered, reduction kernels are the stubs'.
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(*args, **kwargs)
                torch.cuda.synchronize()
            events = [e for e in prof.events() if e.device_type.name == "CUDA"]
            return any("torch_nativeopsreductions" in e.name for e in events)

        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "0"}
        cutedsl = torch.backends.python_native.cutedsl
        was_enabled = cutedsl.enabled
        cutedsl.disable()
        try:
            with mock.patch.dict(os.environ, gates):
                for args, kwargs in calls:
                    claimed = cpp.covers(args, kwargs)
                    fn = with_out if "out" in kwargs else functional
                    x, out = args[0], kwargs.get("out")
                    out_dtype = None if out is None else out.dtype
                    msg = f"{tuple(x.shape)} {x.stride()} {x.dtype} {out_dtype=}"
                    self.assertEqual(claimed, fired(fn, args, kwargs), msg=msg)
        finally:
            if was_enabled:
                cutedsl.enable()

    @skipIfNoCuteDSL
    def test_dynamic_row_builder_converts_output(self, device):
        import cutlass.cute as cute

        from torch._native.ops.reductions import aot_kernel

        built = aot_kernel.build(
            {
                "path": "row",
                "trait": "sum",
                "dtype": "bfloat16",
                "out_dtype": "float32",
                "row_class": "t32_b128",
            }
        )
        compiled = cute.compile(built["fn"], *built["fake_args"])
        self.assertEqual(built["prefix"], "reduction_row_sum_bf16_f32_t32_b128")
        self.assertEqual(
            built["tensor_args"],
            [
                {
                    "name": "mX",
                    "dynamic_sizes": [0, 1],
                    "dynamic_strides": [0],
                    "read_only": True,
                },
                {"name": "mOut", "dynamic_sizes": [0]},
            ],
        )
        self.assertEqual(
            built["scalar_args"],
            [
                {"name": "nchunks", "ctype": "int32_t"},
                {"name": "nwaves", "ctype": "int32_t"},
                {"name": "n", "ctype": "int32_t"},
            ],
        )
        x = torch.randn(8, 1536, device=device, dtype=torch.bfloat16)
        out = torch.empty(8, device=device, dtype=torch.float32)
        compiled(x, out, 192, 6, 1536, torch.cuda.current_stream().cuda_stream)
        with torch.backends.python_native.cutedsl.disabled():
            want = x.sum(dim=1, dtype=torch.float32)
        self.assertEqual(out, want)

    @skipIfNoCuteDSL
    def test_xcta_builder_sequences_both_stages(self, device):
        import cutlass.cute as cute

        from torch._native.ops.reductions import aot_kernel

        built = aot_kernel.build(
            {
                "path": "xcta",
                "trait": "sum",
                "dtype": "bfloat16",
                "out_dtype": "float32",
            }
        )
        compiled = cute.compile(built["fn"], *built["fake_args"])
        m, n, subrow = 4, 131072, 8192
        partials = n // subrow
        x = torch.randn(m, n, device=device, dtype=torch.bfloat16)
        part = torch.empty(m * partials, device=device)
        out = torch.empty(m, device=device)
        compiled(
            x.view(m * partials, subrow),
            [part],
            [out],
            m * partials,
            partials,
            m,
            n,
            torch.cuda.current_stream().cuda_stream,
        )
        with torch.backends.python_native.cutedsl.disabled():
            want = x.sum(dim=1, dtype=torch.float32)
        self.assertEqual(out, want, atol=2e-2, rtol=1e-4)

    @skipIfNoCuteDSL
    def test_trailing_dims_and_ordered_coverage(self, device):
        declarations = {
            declaration.ATEN_OP: declaration
            for declaration in load_declarations(str(_DECL))
        }
        x = torch.randn(4, 3, 512, device=device, dtype=torch.bfloat16)
        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "0"}
        with mock.patch.dict(os.environ, gates):
            self.assertEqual(
                declarations["sum.dim_IntList"].covered_axes(
                    x, [-2, -1], dtype=torch.float32
                )["path"],
                "row",
            )
            self.assertEqual(
                declarations["prod.dim_int"].covered_axes(x, -1)["path"], "row"
            )
            self.assertIsNone(
                declarations["sum.dim_IntList"].covered_axes(x, [0])["path"]
            )
            wide = torch.randn(2, 131072, device=device, dtype=torch.bfloat16)
            self.assertEqual(
                declarations["sum.dim_IntList"].covered_axes(
                    wide, [-1], dtype=torch.float32
                )["path"],
                "xcta",
            )

            covers = declarations["sum.dim_IntList"].covered_axes
            x16 = torch.randn(4, 1040, device=device, dtype=torch.bfloat16)
            self.assertEqual(covers(x16[:, :512], [-1])["path"], "row")
            r = covers(x16[:, :511], [-1])
            self.assertEqual((r["path"], r["row_class"]), ("ragged", "t32_b128"))
            # A pitch that is not whole vectors takes the ragged kernel at a whole width.
            narrow = x16.view(-1)[: 4 * 1028].view(4, 1028)[:, :1024]
            self.assertEqual(covers(narrow, [-1])["path"], "ragged")

            misaligned = torch.randn(4 * 512 + 1, device=device)[1:].view(4, 512)
            self.assertIsNone(
                declarations["sum.dim_IntList"].covered_axes(misaligned, [-1])["path"]
            )

        # Ordered requests stay with the JIT overrides.
        for order in ("1", "strict"):
            gates["PYTORCH_NATIVE_INNER_TREE"] = order
            with mock.patch.dict(os.environ, gates):
                for op in ("sum.dim_IntList", "mean.dim", "amax"):
                    self.assertIsNone(declarations[op].covered_axes(x, [-1])["path"])

        # PYTORCH_SUM_INNER_TREE orders only sum and prod.
        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "1"}
        with mock.patch.dict(os.environ, gates):
            for op in ("sum.dim_IntList", "prod.dim_int"):
                self.assertIsNone(declarations[op].covered_axes(x, -1)["path"])
            mean_path = declarations["mean.dim"].covered_axes(x, -1)["path"]
            self.assertEqual(mean_path, "row")


instantiate_device_type_tests(TestReductionAOTCUDA, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
