# Owner(s): ["module: dsl-native-ops"]

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfNoNativeAot,
    TestCase,
)
from torchgen.native_aot_decl import load_declarations
from torchgen.native_aot_spec_grid import expand_specs


_DECL = (
    Path(__file__).parents[2] / "torch" / "_native" / "ops" / "reductions" / "aot.py"
)

_AOT_PROBE = r"""
import json
import os
import torch
from torch.profiler import profile, ProfilerActivity

results = []
ordered = os.getenv("PYTORCH_NATIVE_INNER_TREE") == "1"
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
    if ordered:
        from torch._native.ops.reductions.ordered import sum_into
        expected = torch.empty_like(out)
        sum_into(expected, x)
        correct = torch.equal(out, expected)
    else:
        expected = x.to(torch.float64).sum(dim=-1).to(torch.float32)
        correct = torch.allclose(out, expected, atol=2e-2, rtol=1e-4)
    results.append((launches, correct))
print("RESULTS=" + json.dumps(results))
"""


class TestReductionAOT(TestCase):
    def test_core_family_grid(self):
        declarations = load_declarations(str(_DECL))
        self.assertEqual(
            [declaration.ATEN_OP for declaration in declarations],
            ["sum.dim_IntList", "mean.dim", "amax", "amin", "prod.dim_int"],
        )
        for declaration in declarations:
            grid = expand_specs(declaration.kernel_precompile_grid())
            ordered = declaration.ATEN_OP in {
                "sum.dim_IntList",
                "mean.dim",
                "prod.dim_int",
            }
            expected = 135 if ordered else 18
            self.assertEqual(len(grid), expected)
            paths = {"row", "xcta"}
            if ordered:
                paths.update({"itree_multirow", "itree_looped", "itree_split"})
            self.assertEqual({point["path"] for point in grid}, paths)
            self.assertEqual(
                {point["order"] for point in grid},
                {"linear", "inner_tree"} if ordered else {"linear"},
            )
            self.assertEqual(
                {point["dtype"] for point in grid},
                {"float16", "bfloat16", "float32"},
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
    @skipIfNoNativeAot("sum.dim_IntList")
    @parametrize("ordered", (False, True))
    def test_embedded_callable_owns_reduction_launches(self, device, ordered):
        env = dict(os.environ, TORCH_DISABLE_NATIVE_JIT="1")
        if ordered:
            env["PYTORCH_NATIVE_INNER_TREE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", _AOT_PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        line = next(
            line for line in proc.stdout.splitlines() if line.startswith("RESULTS=")
        )
        self.assertEqual(
            json.loads(line.removeprefix("RESULTS=")), [[1, True], [2, True]]
        )

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
                "order": "linear",
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
    @parametrize("n", (32, 1536, 32768))
    def test_inner_tree_builders_preserve_order(self, device, n):
        import cutlass.cute as cute

        from torch._native.ops.reductions import _aot_data, aot_kernel
        from torch._native.ops.reductions.inner_tree_plan import (
            compute_inner_tree_params,
        )
        from torch._native.ops.reductions.ordered import sum_into

        m = 5
        path, loads = _aot_data.inner_tree_spec("float32", n, m)
        built = aot_kernel.build(
            {
                "path": path,
                "trait": "sum",
                "dtype": "float32",
                "out_dtype": "float32",
                "loads": loads,
                "order": "inner_tree",
            }
        )
        compiled = cute.compile(built["fn"], *built["fake_args"])
        x = torch.randn(m, n, device=device)
        out = torch.empty(m, device=device)
        args = [x, [out]]
        plan = compute_inner_tree_params(n, m, 4)
        if path == "itree_multirow":
            args.extend((n, m))
        elif path == "itree_looped":
            args.extend((n, m, plan.num_warps, plan.num_batches))
        else:
            part = torch.empty(m * plan.num_batches, device=device)
            args = [x, [part], [out], m, plan.num_batches, n]
        compiled(*args, torch.cuda.current_stream().cuda_stream)
        expected = torch.empty_like(out)
        sum_into(expected, x)
        self.assertEqual(out, expected)

    @skipIfNoCuteDSL
    @parametrize(
        "trait_name,out_dtypes",
        (
            ("maxdim", ("float32", "int64")),
            ("varmean", ("float32", "float32")),
        ),
    )
    def test_inner_tree_split_uses_trait_state(self, device, trait_name, out_dtypes):
        import cutlass.cute as cute

        from torch._native.cutedsl.dtypes import cute2torch
        from torch._native.ops.reductions import _aot_data, aot_kernel

        m, n = 3, 32768
        path, loads = _aot_data.inner_tree_spec("float32", n, m)
        self.assertEqual(path, "itree_split")
        built = aot_kernel.build(
            {
                "path": path,
                "trait": trait_name,
                "dtype": "float32",
                "out_dtypes": out_dtypes,
                "loads": loads,
                "order": "inner_tree",
            }
        )
        compiled = cute.compile(built["fn"], *built["fake_args"])
        trait = aot_kernel._TRAITS[trait_name]()
        x = torch.randn(m, n, device=device)
        batches = n // 8192
        parts = [
            torch.empty(m * batches, device=device, dtype=cute2torch[field_dtype])
            for field_dtype in trait.fdtypes
        ]
        name_to_dtype = {"float32": torch.float32, "int64": torch.int64}
        outs = [
            torch.empty(m, device=device, dtype=name_to_dtype[name])
            for name in out_dtypes
        ]
        compiled(
            x,
            parts,
            outs,
            m,
            batches,
            n,
            torch.cuda.current_stream().cuda_stream,
        )
        expected = (
            torch.max(x, dim=-1)
            if trait_name == "maxdim"
            else torch.var_mean(x, dim=-1)
        )
        self.assertEqual(tuple(outs), expected)

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
                "order": "linear",
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
    def test_trailing_dims_and_inner_tree_coverage(self, device):
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

        gates["PYTORCH_NATIVE_INNER_TREE"] = "1"
        with mock.patch.dict(os.environ, gates):
            self.assertEqual(
                declarations["sum.dim_IntList"].covered_axes(x, [-1])["path"],
                "itree_looped",
            )
            self.assertEqual(declarations["amax"].covered_axes(x, [-1])["path"], "row")


instantiate_device_type_tests(TestReductionAOTCUDA, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
