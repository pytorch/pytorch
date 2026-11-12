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
from torchgen.native_aot_decl import cc_of, load_by_path, load_declarations
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

# (op, shape, dtype, PYTORCH_NATIVE_INNER_TREE): row, cross-CTA, multirow and split
# callables, each called functionally and with out=.
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
            ordered = declaration.ATEN_OP in {
                "sum.dim_IntList",
                "mean.dim",
                "prod.dim_int",
            }
            # 15 points per dtype pair; dtype= adds only FP16->FP32 and BF16->FP32.
            expected = 75 if ordered else 18
            self.assertEqual(len(grid), expected)
            paths = {"row", "xcta"}
            if ordered:
                paths.update({"itree_multirow", "itree_looped", "itree_split"})
            self.assertEqual({point["path"] for point in grid}, paths)
            self.assertEqual(
                {point["order"] for point in grid},
                {"linear", "inner_tree"} if ordered else {"linear"},
            )
            same = {(dtype, dtype) for dtype in ("float16", "bfloat16", "float32")}
            widened = {("float16", "float32"), ("bfloat16", "float32")}
            self.assertEqual(
                {(point["dtype"], point["out_dtype"]) for point in grid},
                same | widened if ordered else same,
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
                    _AOT_PROBE,
                    {"PYTORCH_NATIVE_INNER_TREE": "1"},
                    [[1, True], [2, True]],
                ),
                name="sum_ordered",
                decorators=[skipIfNoNativeAot("sum.dim_IntList")],
            ),
            subtest(
                (
                    _PATH_PROBE,
                    {},
                    [[k, k, True, True] for k in (1, 1, 2, 2, 1, 1, 1, 2)],
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

    def test_sum_rollout_gate_matches_the_adapter(self, device):
        """PYTORCH_SUM_INNER_TREE orders exactly the calls cutedsl_impl's adapter takes."""
        from torch._native.ops.reductions import cutedsl_impl

        aot = load_by_path("reduction_aot_gates", str(_DECL))
        x = torch.randn(8, 1024, device=device)
        xb = x.bfloat16()
        # Int32-unsafe batches without the memory: m * ceil(n / 128) exceeds 2**31.
        huge = torch.empty(1, 2**20, device=device).expand(2**21, 2**20)
        cases = (
            ("sum", x, [-1], None),
            ("prod", xb, 1, None),
            ("mean", x, [-1], None),
            ("sum", xb, [-1], torch.float32),
            ("sum", x, [-1], torch.float32),
            ("sum", x, [-2, -1], None),
            ("sum", x, None, None),
            ("sum", huge, [-1], None),
        )
        for trait, t, dim, dtype in cases:
            style = "int" if isinstance(dim, int) else "optional_list"
            n, m, _ = aot._row_geometry(t, dim, style)
            got = aot._rollout_takes(trait, dim, dtype, n, m, t.element_size())
            # The adapter registers sum and prod only.
            want = trait in ("sum", "prod") and cutedsl_impl._cond(t, dim, dtype=dtype)
            self.assertEqual(got, want, msg=f"{trait} {tuple(t.shape)} {dim=} {dtype=}")

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
            torch.randn(4, 32, device=device),  # multirow
            torch.randn(8, 1024, device=device),  # t32_b128, looped
            torch.randn(8, 4096, device=device, dtype=torch.bfloat16),  # t64_b128
            torch.randn(8, 7168, device=device, dtype=torch.bfloat16),  # t128_b128
            torch.randn(2, 4, 4096, device=device),  # t256_b256
            torch.randn(256, device=device),  # t64_b64 as a reduce-all
            torch.randn(2, 32768, device=device),  # split
            torch.randn(2, 131072, device=device, dtype=torch.bfloat16),  # xcta
            torch.randn(4, 1001, device=device),  # ragged rows
            flat[1:].view(8, 1024),  # misaligned
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
        if op in ("amax", "amin") and gate == "PYTORCH_NATIVE_INNER_TREE":
            # amax and amin export no inner-tree callables, so ordered calls stay JIT.
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
    @parametrize(
        "trait_name,dtype_name,n",
        (
            ("sum", "float32", 20),  # multirow, padded to 8 loads
            ("sum", "float32", 32),
            ("sum", "float32", 1536),
            ("mean", "bfloat16", 1536),
            ("sum", "float32", 12288),  # looped, shorter last batch
            ("prod", "float32", 20480),
            ("sum", "float32", 32768),  # split
            ("sum", "float32", 33768),  # split, shorter last batch
            ("mean", "float16", 34824),
            ("prod", "bfloat16", 34824),
        ),
        name_fn=lambda trait_name, dtype_name, n: f"{trait_name}_{dtype_name}_{n}",
    )
    def test_inner_tree_builders_match_jit_bits(
        self, device, trait_name, dtype_name, n
    ):
        """Every inner-tree callable reproduces the JIT inner-tree path bit for bit."""
        import cutlass.cute as cute

        from torch._native.ops.reductions import (
            _aot_data,
            aot_kernel,
            kernel_general as kg,
        )
        from torch._native.ops.reductions.inner_tree_plan import (
            compute_inner_tree_params,
        )

        m = 5
        dtype = getattr(torch, dtype_name)
        path, loads = _aot_data.inner_tree_spec(dtype_name, n, m)
        built = aot_kernel.build(
            {
                "path": path,
                "trait": trait_name,
                "dtype": dtype_name,
                "out_dtype": dtype_name,
                "loads": loads,
                "order": "inner_tree",
            }
        )
        compiled = cute.compile(built["fn"], *built["fake_args"])
        x = torch.randn(m, n, device=device)
        if trait_name == "prod":
            x = 1 + 0.01 * x  # keep a length-n product finite
        x = x.to(dtype)
        out = torch.empty(m, device=device, dtype=dtype)
        args = [x, [out]]
        plan = compute_inner_tree_params(n, m, 16 // x.element_size())
        if path == "itree_multirow":
            args.extend((n, m))
        elif path == "itree_looped":
            args.extend((n, m, plan.num_warps, plan.num_batches))
        else:
            part = torch.empty(m * plan.num_batches, device=device)
            args = [x, [part], [out], m, plan.num_batches, n]
        compiled(*args, torch.cuda.current_stream().cuda_stream)
        trait = aot_kernel._TRAITS[trait_name]()
        key = f"aot_bits_{trait_name}"
        expected = kg.reduce_dim(trait, key, x, [1], dtype, order="inner_tree")
        self.assertEqual(out.view(torch.uint8), expected.view(torch.uint8))

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

            misaligned = torch.randn(4 * 512 + 1, device=device)[1:].view(4, 512)
            self.assertIsNone(
                declarations["sum.dim_IntList"].covered_axes(misaligned, [-1])["path"]
            )

        for order in ("1", "strict"):
            gates["PYTORCH_NATIVE_INNER_TREE"] = order
            with mock.patch.dict(os.environ, gates):
                self.assertEqual(
                    declarations["sum.dim_IntList"].covered_axes(x, [-1])["path"],
                    "itree_looped",
                )
                # amax exports no inner-tree callables, so the JIT serves the order.
                self.assertIsNone(declarations["amax"].covered_axes(x, [-1])["path"])

        # PYTORCH_SUM_INNER_TREE orders only calls its adapter would take.
        gates = {"PYTORCH_NATIVE_INNER_TREE": "0", "PYTORCH_SUM_INNER_TREE": "1"}
        sum_axes = declarations["sum.dim_IntList"].covered_axes
        with mock.patch.dict(os.environ, gates):
            self.assertEqual(sum_axes(x, [-1])["path"], "itree_looped")
            self.assertEqual(sum_axes(x, [-1], dtype=torch.float32)["path"], "row")
            self.assertEqual(sum_axes(x, [-2, -1])["path"], "row")


instantiate_device_type_tests(TestReductionAOTCUDA, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
