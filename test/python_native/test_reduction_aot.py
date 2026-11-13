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

# (op, shape, dtype, PYTORCH_NATIVE_INNER_TREE): row and cross-CTA callables, each
# called functionally and with out=, and ordered requests, which the stubs decline.
_PATH_PROBE = r"""
import json
import os
import torch
from torch.profiler import profile, ProfilerActivity

def run(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type.name == "CUDA"]
    return out, sum("torch_nativeopsreductions" in e.name for e in events)

cases = (
    ("sum", (8, 1024), torch.float32, "0"),
    ("mean", (8, 1024), torch.float16, "0"),
    ("amax", (8, 4096), torch.float32, "0"),
    ("amin", (4, 131072), torch.bfloat16, "0"),
    ("mean", (4, 131072), torch.bfloat16, "0"),
    ("prod", (8, 1024), torch.float32, "0"),
    ("sum", (4, 32), torch.float32, "1"),
    ("prod", (4, 32), torch.float16, "1"),
    ("mean", (4, 32768), torch.float32, "1"),
)
results = []
for name, shape, dtype, order in cases:
    os.environ["PYTORCH_NATIVE_INNER_TREE"] = order
    torch.manual_seed(0)
    x = (1 + 0.01 * torch.randn(shape, device="cuda")).to(dtype)
    op = getattr(torch, name)
    with torch.backends.python_native.cutedsl.disabled():
        ref = op(x, -1)
    got, launches = run(lambda: op(x, -1))
    out = torch.empty_like(ref)
    _, out_launches = run(lambda: op(x, -1, out=out))
    tol = 0 if name in ("amax", "amin") else 1e-2
    ok = [torch.allclose(t, ref, atol=tol, rtol=tol) for t in (got, out)]
    results.append([launches, out_launches, *ok])
print("RESULTS=" + json.dumps(results))
"""


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
            # 6 points per dtype pair; dtype= adds only FP16->FP32 and BF16->FP32.
            self.assertEqual(len(grid), 30 if has_dtype else 18)
            self.assertEqual({point["path"] for point in grid}, {"row", "xcta"})
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


class TestReductionAOTCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize(
        "probe,env,expected",
        [
            subtest(
                (_AOT_PROBE, {}, [[1, True], [2, True]]),
                name="sum",
                decorators=[skipIfNoNativeAot("sum.dim_IntList")],
            ),
            subtest(
                (
                    _PATH_PROBE,
                    {},
                    [[k, k, True, True] for k in (1, 1, 1, 2, 2, 1, 0, 0, 0)],
                ),
                name="paths",
                decorators=[skipIfNoNativeAot("mean.dim")],
            ),
        ],
    )
    def test_embedded_callables_own_reduction_launches(
        self, device, probe, env, expected
    ):
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=dict(os.environ, TORCH_DISABLE_NATIVE_JIT="1", **env),
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        line = next(
            line for line in proc.stdout.splitlines() if line.startswith("RESULTS=")
        )
        self.assertEqual(json.loads(line.removeprefix("RESULTS=")), expected)

    @parametrize("op", [d.ATEN_OP for d in load_declarations(str(_DECL))])
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

        flat = torch.randn(8 * 1024 + 1, device=device)
        inputs = [
            torch.randn(4, 32, device=device),  # t32_b128
            torch.randn(8, 1024, device=device),  # t32_b128
            torch.randn(8, 4096, device=device, dtype=torch.bfloat16),  # t64_b128
            torch.randn(8, 7168, device=device, dtype=torch.bfloat16),  # t128_b128
            torch.randn(2, 4, 4096, device=device),  # t256_b256
            torch.randn(256, device=device),  # t64_b64 as a reduce-all
            torch.randn(2, 32768, device=device),  # t256_b256
            torch.randn(2, 131072, device=device, dtype=torch.bfloat16),  # xcta
            torch.randn(4, 1001, device=device),  # ragged rows
            flat[1:].view(8, 1024),  # misaligned
            torch.randn(4096, 32, device=device),  # narrow rows, not a reduce-all
            # A row over 192 KiB that is not a multiple of 8192 elements.
            torch.randn(2, 98312, device=device, dtype=torch.bfloat16),
            torch.randn(8, 1024, device=device, dtype=torch.float16),  # fp16
        ]
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
        self.assertEqual(out, x.sum(dim=1, dtype=torch.float32))

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
        self.assertEqual(out, x.sum(dim=1, dtype=torch.float32), atol=2e-2, rtol=1e-4)

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
