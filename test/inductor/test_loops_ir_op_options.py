# Owner(s): ["module: inductor"]
"""Per-operation implementation options: typed registry/config/fusion
contracts (CPU) and isolated GPU accuracy of the generated/native codegen.
No timing; no global precision mutation except a restored freeze test."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch._inductor.codegen import loops_ir_op_options as oo
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires CUDA")

OPT_DIR = (
    Path(oo.__file__).resolve().parents[2]
    / "csrc/inductor/experimental/loops_ir/op_options"
)
IMPLEMENTED = [
    "native_aten_gemm",
    "native_aten_mm_bias_epilogue",
    "triton_tiled_gemm",
    "native_aten_reduction",
    "triton_row_reduction",
]


def E():
    return oo.load_op_options_native()


def cpu_ctx(numerics="inductor", tf32=False):
    e = E()
    strict = numerics == "strict_eager"
    native = e.CodegenContext(
        numerics=e.NumericsPolicy.STRICT_EAGER if strict else e.NumericsPolicy.INDUCTOR,
        device=e.DeviceKind.CUDA,
        compute_capability=90,
        max_shared_memory_bytes=232448,
        stock_fp32_matmul_precision=e.MatmulPrecision.TF32
        if tf32
        else e.MatmulPrecision.IEEE,
        enable_fp_fusion=not strict,
        disable_ftz=strict,
        division_rounding=False,
    )
    return oo.FrozenCodegenContext(native, 0, None)


def layout(sizes, strides=None, dtype=None, dynamic=False):
    e = E()
    if strides is None:
        strides = torch.empty(sizes, device="meta").stride()
    kind = e.ShapeKind.DYNAMIC if dynamic else e.ShapeKind.STATIC
    return e.TensorLayout(dtype or e.StorageDtype.F32, list(sizes), list(strides), kind)


def operand(name, sizes, strides=None, **kw):
    return E().TensorOperand(name, layout(sizes, strides, **kw))


def mm_sig_shapes(m, n, k, **kw):
    e = E()
    return e.OpSignature.mm(
        e.DeviceKind.CUDA,
        operand("a", (m, k), **kw),
        operand("b", (k, n)),
        operand("out", (m, n)),
    )


def mm_sig(a, b, out):
    e = E()
    return e.OpSignature.mm(
        e.DeviceKind.CUDA,
        oo.tensor_operand("a", a),
        oo.tensor_operand("b", b),
        oo.tensor_operand("out", out),
    )


def matmul_config(**kw):
    base = dict(
        block_m=32, block_n=32, block_k=16, group_m=8, num_warps=4, num_stages=2
    )
    base.update(kw)
    return E().MatmulConfig(**base)


def epilogue_bias_relu_scale(out_layout, bias_coeffs, scale=2.0, negate=False):
    """fused = relu(acc + bias) * scale  (negate: acc - bias via mul by -1)."""
    e, idx = E(), E().LoopIndex
    sizes, strides = list(out_layout.sizes), list(out_layout.strides)
    names = [f"d{i}" for i in range(len(sizes))]
    b = e.RegionBuilder(
        name="epi",
        inputs=["acc", "bias"],
        result="fused",
        result_elements=out_layout.extent(),
        ranges=[(n, 0, s) for n, s in zip(names, sizes)],
    )

    def lin(coeffs):
        expr = idx.const(0)
        for n, c in zip(names, coeffs):
            expr = expr + idx.var(n) * idx.const(c)
        return expr

    acc = b.load("acc", lin(strides))
    bias = b.load("bias", lin(bias_coeffs))
    if negate:
        bias = b.binary("mul", bias, b.constant(-1.0))
    y = b.relu(b.binary("add", acc, bias))
    y = b.binary("mul", y, b.constant(scale))
    b.store("fused", lin(strides), y)
    return b.build()


def fused_epilogue(body, extras):
    f = E().FusedLoops()
    f.set_epilogue("acc", body)
    for name, lay in extras:
        f.add_extra_input(name, lay)
    return f


def assert_rejected(test, reason, fn):
    with test.assertRaises(E().OptionRejected) as cm:
        fn()
    test.assertEqual(cm.exception.reason, reason, str(cm.exception))


class TestOpOptionsNativeUnit(TestCase):
    def test_standalone_cpp_registry(self):
        cxx = os.environ.get("CXX", "g++")
        srcs = [
            "../ir.cpp",
            "op_types.cpp",
            "op_config.cpp",
            "fused_region.cpp",
            "option_registry.cpp",
            "gemm_options.cpp",
            "reduction_options.cpp",
            "tests/op_options_cpu_test.cpp",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "op_options_cpu_test")
            subprocess.run(
                [cxx, "-std=c++17", "-O0", "-Wall", *srcs, "-o", exe],
                cwd=OPT_DIR,
                check=True,
            )
            out = subprocess.run(
                [exe], check=True, capture_output=True, text=True
            ).stdout
        self.assertIn("checks passed", out)


class TestOpOptionsRegistry(TestCase):
    def test_lists_only_implemented_options(self):
        e = E()
        reg = e.default_registry()
        self.assertEqual(reg.option_ids(), IMPLEMENTED)
        for kind in (
            e.OpKind.VAR_CORRECTION,
            e.OpKind.STD_CORRECTION,
            e.OpKind.SOFTMAX,
            e.OpKind.LOG_SOFTMAX,
        ):
            self.assertEqual(reg.options_for(kind), [])
        sig = e.OpSignature.reduction(
            e.OpKind.VAR_CORRECTION,
            e.DeviceKind.CUDA,
            operand("x", (4, 8)),
            [1],
            False,
            1.0,
            operand("o", (4,)),
        )
        self.assertEqual(reg.eligible_options(sig, cpu_ctx().native), [])
        assert_rejected(
            self, e.RejectReason.UNKNOWN_OPTION, lambda: reg.get("triton_template_mm")
        )
        opt = reg.get("triton_tiled_gemm")
        self.assertEqual(opt.fence, e.NumericsFence.INDUCTOR_ONLY)
        self.assertEqual(opt.strategy, e.CodegenStrategy.TRITON_GEMM)
        self.assertEqual(
            reg.get("native_aten_gemm").fence, e.NumericsFence.STRICT_EAGER
        )

    def test_schema_exposes_every_parameter(self):
        e = E()
        mm = e.config_schema(e.ConfigKind.MATMUL)
        self.assertEqual(
            [p.name for p in mm.params],
            [
                "block_m",
                "block_n",
                "block_k",
                "group_m",
                "num_warps",
                "num_stages",
                "split_k",
                "matrix_precision",
                "acc_dtype",
            ],
        )
        self.assertEqual(mm.param("split_k").legal_ints, [1])
        self.assertFalse(mm.param("split_k").tunable)
        self.assertEqual(mm.param("matrix_precision").legal_enums, ["ieee", "tf32"])
        red = e.config_schema(e.ConfigKind.REDUCTION)
        self.assertEqual(
            [p.name for p in red.params],
            ["block_x", "block_r", "num_warps", "num_stages", "acc_dtype"],
        )
        self.assertTrue(mm.constraints and red.constraints)

    def test_context_device_index_and_vendor_policy_in_key(self):
        e = E()
        sig = mm_sig_shapes(64, 64, 64)
        base = cpu_ctx()
        moved = e.CodegenContext(
            numerics=e.NumericsPolicy.INDUCTOR,
            device=e.DeviceKind.CUDA,
            compute_capability=90,
            max_shared_memory_bytes=232448,
            stock_fp32_matmul_precision=e.MatmulPrecision.IEEE,
            enable_fp_fusion=True,
            disable_ftz=False,
            division_rounding=False,
            device_index=1,
        )
        self.assertEqual(base.native.device_index, 0)
        self.assertEqual(base.native.addmm_lt_policy, e.AddmmLtPolicy.CUDA_DEFAULT)
        k0 = oo.codegen(
            "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), base
        ).cache_key
        k1 = oo.codegen(
            "triton_tiled_gemm",
            matmul_config(),
            sig,
            e.FusedLoops(),
            oo.FrozenCodegenContext(moved, 1, None),
        ).cache_key
        self.assertNotEqual(k0, k1)
        with self.assertRaisesRegex(ValueError, "device_index"):
            oo.FrozenCodegenContext(moved, 0, None)
        self.assertFalse(e.config_schema(e.ConfigKind.MATMUL).param("group_m").tunable)

    def test_unknown_or_missing_parameter_is_a_type_error(self):
        with self.assertRaises(TypeError):
            matmul_config(bogus=1)
        with self.assertRaises(TypeError):
            E().MatmulConfig(block_m=32)
        with self.assertRaises(TypeError):
            E().ReductionConfig(block_x=1, block_r=64, num_warps=4, vector=4)

    def test_invalid_configs_rejected(self):
        e = E()
        reg, ctx, sig = (
            e.default_registry(),
            cpu_ctx().native,
            mm_sig_shapes(128, 128, 128),
        )
        bad = [
            dict(block_m=24),
            dict(num_warps=3),
            dict(split_k=2),
            dict(group_m=4),
            dict(block_m=256, block_n=256, block_k=256, num_stages=5),
            dict(block_m=256, block_n=256),
            dict(matrix_precision=e.MatmulPrecision.TF32),
        ]
        for kw in bad:
            assert_rejected(
                self,
                e.RejectReason.CONFIG,
                lambda: reg.validate(
                    "triton_tiled_gemm", sig, matmul_config(**kw), e.FusedLoops(), ctx
                ),
            )
        assert_rejected(
            self,
            e.RejectReason.CONFIG,
            lambda: reg.validate(
                "triton_tiled_gemm", sig, e.NativeConfig(), e.FusedLoops(), ctx
            ),
        )
        rsig = e.OpSignature.reduction(
            e.OpKind.SUM_DIM,
            e.DeviceKind.CUDA,
            operand("x", (8, 64)),
            [1],
            False,
            None,
            operand("o", (8,)),
        )
        assert_rejected(
            self,
            e.RejectReason.CONFIG,
            lambda: reg.validate(
                "triton_row_reduction",
                rsig,
                e.ReductionConfig(block_x=128, block_r=4096, num_warps=4),
                e.FusedLoops(),
                ctx,
            ),
        )

    def test_eligibility_fences(self):
        e = E()
        reg = e.default_registry()
        cfg = matmul_config()
        sig = mm_sig_shapes(32, 32, 32)
        strict = cpu_ctx("strict_eager").native
        self.assertEqual(reg.eligible_options(sig, strict), ["native_aten_gemm"])
        self.assertEqual(
            reg.eligible_options(sig, cpu_ctx().native),
            ["native_aten_gemm", "native_aten_mm_bias_epilogue", "triton_tiled_gemm"],
        )
        assert_rejected(
            self,
            e.RejectReason.NUMERICS_FENCE,
            lambda: reg.validate("triton_tiled_gemm", sig, cfg, e.FusedLoops(), strict),
        )
        ctx = cpu_ctx().native
        bf16 = e.OpSignature.mm(
            e.DeviceKind.CUDA,
            *(
                operand(n, s, dtype=e.StorageDtype.BF16)
                for n, s in (("a", (8, 8)), ("b", (8, 8)), ("out", (8, 8)))
            ),
        )
        assert_rejected(
            self,
            e.RejectReason.DTYPE,
            lambda: reg.validate("triton_tiled_gemm", bf16, cfg, e.FusedLoops(), ctx),
        )
        assert_rejected(
            self,
            e.RejectReason.DTYPE,
            lambda: reg.validate(
                "native_aten_gemm", bf16, e.NativeConfig(), e.FusedLoops(), ctx
            ),
        )
        dyn = mm_sig_shapes(8, 8, 8, dynamic=True)
        assert_rejected(
            self,
            e.RejectReason.DYNAMIC_SHAPE,
            lambda: reg.validate("triton_tiled_gemm", dyn, cfg, e.FusedLoops(), ctx),
        )
        assert_rejected(
            self,
            e.RejectReason.RANK,
            lambda: e.OpSignature.mm(
                e.DeviceKind.CUDA,
                operand("a", (2, 8, 8)),
                operand("b", (8, 8)),
                operand("out", (8, 8)),
            ),
        )
        overlap = e.OpSignature.mm(
            e.DeviceKind.CUDA,
            operand("a", (8, 8)),
            operand("b", (8, 8)),
            operand("out", (8, 8), (1, 1)),
        )
        assert_rejected(
            self,
            e.RejectReason.LAYOUT,
            lambda: reg.validate(
                "triton_tiled_gemm", overlap, cfg, e.FusedLoops(), ctx
            ),
        )
        transposed_out = e.OpSignature.mm(
            e.DeviceKind.CUDA,
            operand("a", (8, 8)),
            operand("b", (8, 8)),
            operand("out", (8, 8), (1, 8)),
        )
        reg.validate("triton_tiled_gemm", transposed_out, cfg, e.FusedLoops(), ctx)
        assert_rejected(
            self,
            e.RejectReason.LAYOUT,
            lambda: reg.validate(
                "native_aten_gemm",
                transposed_out,
                e.NativeConfig(),
                e.FusedLoops(),
                ctx,
            ),
        )
        multi = e.OpSignature.reduction(
            e.OpKind.MEAN_DIM,
            e.DeviceKind.CUDA,
            operand("x", (3, 4, 5)),
            [0, 2],
            False,
            None,
            operand("o", (4,)),
        )
        assert_rejected(
            self,
            e.RejectReason.RANK,
            lambda: reg.validate(
                "triton_row_reduction",
                multi,
                e.ReductionConfig(block_x=1, block_r=16, num_warps=4),
                e.FusedLoops(),
                ctx,
            ),
        )

    def test_tf32_only_where_stock_allows(self):
        e = E()
        reg = e.default_registry()
        tf32 = matmul_config(matrix_precision=e.MatmulPrecision.TF32)
        big = mm_sig_shapes(64, 512, 512)
        reg.validate(
            "triton_tiled_gemm", big, tf32, e.FusedLoops(), cpu_ctx(tf32=True).native
        )
        for ctx, sig in (
            (cpu_ctx().native, big),
            (cpu_ctx(tf32=True).native, mm_sig_shapes(64, 512, 256)),
        ):
            assert_rejected(
                self,
                e.RejectReason.CONFIG,
                lambda: reg.validate(
                    "triton_tiled_gemm", sig, tf32, e.FusedLoops(), ctx
                ),
            )
        precisions = {
            c.matrix_precision
            for c in reg.candidate_configs("triton_tiled_gemm", big, cpu_ctx().native)
        }
        self.assertEqual(precisions, {e.MatmulPrecision.IEEE})

    def test_candidate_manifests(self):
        e = E()
        reg, ctx = e.default_registry(), cpu_ctx().native
        sig = mm_sig_shapes(1000, 1000, 1000)
        a = reg.candidate_configs("triton_tiled_gemm", sig, ctx)
        b = reg.candidate_configs("triton_tiled_gemm", sig, ctx)
        self.assertEqual([c.canonical() for c in a], [c.canonical() for c in b])
        self.assertTrue(1 < len(a) <= 21)
        self.assertEqual(len({c.canonical() for c in a}), len(a))
        rsig = e.OpSignature.reduction(
            e.OpKind.SUM_DIM,
            e.DeviceKind.CUDA,
            operand("x", (64, 700)),
            [1],
            False,
            None,
            operand("o", (64,)),
        )
        rc = reg.candidate_configs("triton_row_reduction", rsig, ctx)
        self.assertTrue(
            any(c.block_r >= 700 for c in rc) and any(c.block_r < 700 for c in rc)
        )
        self.assertEqual(len(reg.candidate_configs("native_aten_gemm", sig, ctx)), 1)

    def test_fusion_contract_validation(self):
        e = E()
        reg, ctx = e.default_registry(), cpu_ctx().native
        sig = mm_sig_shapes(8, 16, 4)
        out = sig.output.layout
        bias = layout((16,))
        body = epilogue_bias_relu_scale(out, [0, 1])
        sel = reg.validate(
            "triton_tiled_gemm",
            sig,
            matmul_config(),
            fused_epilogue(body, [("bias", bias)]),
            ctx,
        )
        kinds = [op.kind for op in sel.fusion.epilogue.ops]
        k = e.RegionOpKind
        self.assertEqual(
            kinds, [k.LOAD_SELF, k.LOAD_EXTRA, k.ADD, k.RELU, k.CONSTANT, k.MUL]
        )
        self.assertEqual(list(sel.fusion.epilogue.ops[1].access.coeffs), [0, 1])
        # Native GEMM has no fusion contract; native addmm epilogue only accepts acc + bias.
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: reg.validate(
                "native_aten_gemm",
                sig,
                e.NativeConfig(),
                fused_epilogue(body, [("bias", bias)]),
                ctx,
            ),
        )
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: reg.validate(
                "native_aten_mm_bias_epilogue",
                sig,
                e.NativeConfig(),
                fused_epilogue(body, [("bias", bias)]),
                ctx,
            ),
        )
        # Out-of-bounds broadcast map is caught by the core LoopsIR verifier.
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: reg.validate(
                "triton_tiled_gemm",
                sig,
                matmul_config(),
                fused_epilogue(body, [("bias", layout((15,)))]),
                ctx,
            ),
        )
        # Non-f32 extra input.
        assert_rejected(
            self,
            e.RejectReason.DTYPE,
            lambda: reg.validate(
                "triton_tiled_gemm",
                sig,
                matmul_config(),
                fused_epilogue(
                    body, [("bias", layout((16,), dtype=e.StorageDtype.F16))]
                ),
                ctx,
            ),
        )
        # Accumulator read at a transposed map is not available in the tile.
        transposed = epilogue_bias_relu_scale(layout((8, 16), (1, 8)), [0, 1])
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: reg.validate(
                "triton_tiled_gemm",
                sig,
                matmul_config(),
                fused_epilogue(transposed, [("bias", bias)]),
                ctx,
            ),
        )

    def test_unsupported_regions_rejected(self):
        e, idx = E(), E().LoopIndex
        reg, ctx = e.default_registry(), cpu_ctx().native
        sig = mm_sig_shapes(8, 16, 4)
        canon = idx.var("i") * idx.const(16) + idx.var("j")
        ranges = [("i", 0, 8), ("j", 0, 16)]

        def build(index, red=False):
            b = e.RegionBuilder(
                name="epi",
                inputs=["acc"],
                result="fused",
                result_elements=128,
                ranges=ranges,
                reduction_ranges=[("r", 0, 2)] if red else [],
                accumulators=["s"] if red else [],
            )
            if red:
                b.accumulate([b.reduction_load("acc", canon)])
                v = b.read_accumulator(0)
            else:
                v = b.load("acc", index)
            b.store("fused", canon, v)
            return b.build()

        for body in (build(canon // idx.const(1)), build(canon, red=True)):
            f = e.FusedLoops()
            f.set_epilogue("acc", body)
            assert_rejected(
                self,
                e.RejectReason.FUSION,
                lambda: reg.validate("triton_tiled_gemm", sig, matmul_config(), f, ctx),
            )
        # Codegen re-validates and raises; it never falls back to another option.
        f = e.FusedLoops()
        f.set_epilogue("acc", build(canon // idx.const(1)))
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: oo.codegen("triton_tiled_gemm", matmul_config(), sig, f, cpu_ctx()),
        )
        # Prologue on the addmm bias slot is not part of the contract.
        addmm = e.OpSignature.addmm(
            e.DeviceKind.CUDA,
            operand("bias", (16,)),
            operand("a", (8, 4)),
            operand("b", (4, 16)),
            operand("out", (8, 16)),
            1.0,
            1.0,
        )
        b = e.RegionBuilder(
            name="pro",
            inputs=["bias"],
            result="p",
            result_elements=16,
            ranges=[("j", 0, 16)],
        )
        b.store("p", idx.var("j"), b.load("bias", idx.var("j")))
        f = e.FusedLoops()
        f.add_prologue(0, b.build())
        assert_rejected(
            self,
            e.RejectReason.FUSION,
            lambda: reg.validate("triton_tiled_gemm", addmm, matmul_config(), f, ctx),
        )

    def test_cache_key_and_source_track_config_and_fusion(self):
        e = E()
        sig = mm_sig_shapes(64, 64, 64)
        ctx = cpu_ctx()
        k1 = oo.codegen("triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), ctx)
        k1b = oo.codegen("triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), ctx)
        k2 = oo.codegen(
            "triton_tiled_gemm",
            matmul_config(block_n=64, num_warps=8),
            sig,
            e.FusedLoops(),
            ctx,
        )
        self.assertEqual((k1.cache_key, k1.source), (k1b.cache_key, k1b.source))
        self.assertNotEqual(k1.cache_key, k2.cache_key)
        self.assertIn("tl.arange(0, 32)", k1.source)
        self.assertIn("tl.arange(0, 64)", k2.source)
        self.assertIn("'num_warps': 8", k2.source)
        self.assertIn("input_precision='ieee'", k1.source)
        body = epilogue_bias_relu_scale(sig.output.layout, [0, 1])
        k3 = oo.codegen(
            "triton_tiled_gemm",
            matmul_config(),
            sig,
            fused_epilogue(body, [("bias", layout((64,)))]),
            ctx,
        )
        self.assertNotEqual(k1.cache_key, k3.cache_key)
        self.assertIn("tl.PropagateNan.ALL", k3.source)
        self.assertEqual(
            [a.role for a in k3.launch.args],
            [
                oo.ArgRole.INPUT,
                oo.ArgRole.INPUT,
                oo.ArgRole.EXTRA_INPUT,
                oo.ArgRole.OUTPUT,
            ],
        )
        nat = oo.codegen(
            "native_aten_gemm",
            e.NativeConfig(),
            sig,
            e.FusedLoops(),
            cpu_ctx("strict_eager"),
        )
        self.assertEqual(nat.target, "aten.mm.default")


# ---------------------------------------------------------------------------
# GPU accuracy (isolated compile + launch; no timing).
# ---------------------------------------------------------------------------

GEMM_SIZES = [(1, 1, 1), (16, 16, 16), (17, 33, 9), (64, 128, 257), (127, 65, 300)]


def make_matrix(rows, cols, transposed, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if transposed:
        return (torch.randn(cols, rows, device="cuda", generator=g) * scale).t()
    return torch.randn(rows, cols, device="cuda", generator=g) * scale


@requires_cuda
class TestOpOptionsGpu(TestCase):
    def setUp(self):
        super().setUp()
        # The reference profile is the stock default; this suite never changes it.
        self.assertNotEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.ctx = oo.freeze_codegen_context(
            torch.device("cuda", torch.cuda.current_device()), "inductor"
        )
        self.strict = oo.freeze_codegen_context(
            torch.device("cuda", torch.cuda.current_device()), "strict_eager"
        )
        self.reg = E().default_registry()

    def run_generated(self, option, config, sig, fused, inputs, out):
        kernel = oo.codegen(option, config, sig, fused, self.ctx)
        compiled = oo.compile_generated_kernel(kernel, self.ctx)
        compiled.launch(*inputs, out)
        torch.cuda.synchronize()
        return kernel, compiled

    def gemm_configs(self, sig):
        cands = self.reg.candidate_configs("triton_tiled_gemm", sig, self.ctx.native)
        return [cands[0], cands[-1]] if len(cands) > 1 else cands

    def test_mm_layouts_and_boundary_sizes(self):
        e = E()
        for m, n, k in GEMM_SIZES:
            for ta in (False, True):
                for tb in (False, True):
                    # Unit-variance outputs: entries scaled by K**-0.25.
                    scale = max(k, 1) ** -0.25
                    a, b = (
                        make_matrix(m, k, ta, 1, scale),
                        make_matrix(k, n, tb, 2, scale),
                    )
                    ref = torch.mm(a, b)
                    for out_t in (False, True):
                        out = (
                            torch.empty(n, m, device="cuda").t()
                            if out_t
                            else torch.empty(m, n, device="cuda")
                        )
                        sig = mm_sig(a, b, out)
                        cfg = self.gemm_configs(sig)[
                            int(out_t) % len(self.gemm_configs(sig))
                        ]
                        self.run_generated(
                            "triton_tiled_gemm", cfg, sig, e.FusedLoops(), (a, b), out
                        )
                        self.assertEqual(
                            out,
                            ref,
                            msg=f"mm {m}x{n}x{k} ta={ta} tb={tb} out_t={out_t} {cfg}",
                        )
                    out = torch.empty(m, n, device="cuda")
                    call = oo.codegen(
                        "native_aten_gemm",
                        e.NativeConfig(),
                        mm_sig(a, b, out),
                        e.FusedLoops(),
                        self.strict,
                    )
                    self.assertTrue(torch.equal(oo.load_native_call(call)(a, b), ref))

    def test_mm_unscaled_inputs_stock_fp64_accuracy_rule(self):
        # Unscaled randn (output std ~ sqrt(K)) differs from cuBLAS beyond the
        # f32 assert_close default by K-order rounding; check with the stock
        # fp64-reference accuracy rule used for TorchBench accuracy instead.
        from torch._dynamo.utils import same

        e = E()
        m, n, k = 127, 65, 300
        a, b = make_matrix(m, k, False, 1), make_matrix(k, n, True, 2)
        out = torch.empty(m, n, device="cuda")
        sig = mm_sig(a, b, out)
        ref, ref64 = torch.mm(a, b), torch.mm(a.double(), b.double())
        for cfg in self.gemm_configs(sig):
            self.run_generated(
                "triton_tiled_gemm", cfg, sig, e.FusedLoops(), (a, b), out
            )
            self.assertTrue(same(ref, out, fp64_ref=ref64), msg=str(cfg))

    def test_addmm_alpha_beta_broadcast(self):
        e = E()
        m, n, k = 37, 70, 45
        a, b = make_matrix(m, k, False, 3), make_matrix(k, n, True, 4)
        for bias_shape in ((n,), (1, n), (m, 1), (m, n)):
            for alpha, beta in ((1.0, 1.0), (0.5, 2.0), (-1.25, 0.0)):
                bias = torch.randn(bias_shape, device="cuda")
                if beta == 0.0:
                    bias = torch.full(bias_shape, float("nan"), device="cuda")
                ref = torch.addmm(bias, a, b, beta=beta, alpha=alpha)
                out = torch.empty(m, n, device="cuda")
                sig = e.OpSignature.addmm(
                    e.DeviceKind.CUDA,
                    oo.tensor_operand("bias", bias),
                    oo.tensor_operand("a", a),
                    oo.tensor_operand("b", b),
                    oo.tensor_operand("out", out),
                    alpha,
                    beta,
                )
                for cfg in self.gemm_configs(sig):
                    self.run_generated(
                        "triton_tiled_gemm", cfg, sig, e.FusedLoops(), (bias, a, b), out
                    )
                    self.assertEqual(
                        out,
                        ref,
                        msg=f"addmm bias={bias_shape} alpha={alpha} beta={beta} {cfg}",
                    )
                call = oo.codegen(
                    "native_aten_gemm",
                    e.NativeConfig(),
                    sig,
                    e.FusedLoops(),
                    self.strict,
                )
                self.assertTrue(torch.equal(oo.load_native_call(call)(bias, a, b), ref))

    def test_bmm(self):
        e = E()
        for bsz, m, n, k in ((1, 1, 1, 1), (3, 17, 33, 9), (4, 64, 128, 257)):
            a = torch.randn(bsz, m, k, device="cuda")
            b = torch.randn(bsz, n, k, device="cuda").transpose(1, 2)
            ref = torch.bmm(a, b)
            out = torch.empty(bsz, m, n, device="cuda")
            sig = e.OpSignature.bmm(
                e.DeviceKind.CUDA,
                oo.tensor_operand("a", a),
                oo.tensor_operand("b", b),
                oo.tensor_operand("out", out),
            )
            for cfg in self.gemm_configs(sig):
                self.run_generated(
                    "triton_tiled_gemm", cfg, sig, e.FusedLoops(), (a, b), out
                )
                self.assertEqual(out, ref, msg=f"bmm {bsz}x{m}x{n}x{k} {cfg}")
            call = oo.codegen(
                "native_aten_gemm", e.NativeConfig(), sig, e.FusedLoops(), self.strict
            )
            self.assertTrue(torch.equal(oo.load_native_call(call)(a, b), ref))

    def test_addmm_alpha_beta_zero_nan_semantics_match_eager(self):
        """Live eager CUDA addmm: Lt bias path propagates A/B NaN at alpha=0,
        the gemm path never reads A/B at alpha=0, beta=+-0 ignores the bias."""
        e = E()
        m, n = 33, 70
        cases = [
            (0.0, 1.0),
            (-0.0, 1.0),
            (0.0, 0.0),
            (0.0, -0.0),
            (0.0, 2.5),
            (1.0, 0.0),
            (1.0, -0.0),
            (2.0, 1.0),
        ]
        for k in (45, 1):
            a, b = make_matrix(m, k, False, 11), make_matrix(k, n, True, 12)
            a[3, 0], a[5, k - 1], b[0, 9] = float("nan"), float("inf"), float("-inf")
            makers = (
                lambda: torch.randn(n, device="cuda"),
                lambda: torch.randn(1, n, device="cuda"),
                lambda: torch.randn(m, 1, device="cuda"),
                lambda: torch.randn(m, n, device="cuda"),
                lambda: torch.randn(n, 2, device="cuda")[:, 0],
            )
            for make in makers:
                for alpha, beta in cases:
                    bb = make()
                    flat = bb.view(-1) if bb.is_contiguous() else bb
                    flat[0] = -0.0
                    if beta == 0.0:
                        flat[1], flat[2] = float("nan"), float("inf")
                    ref = torch.addmm(bb, a, b, beta=beta, alpha=alpha)
                    out = torch.empty(m, n, device="cuda")
                    sig = e.OpSignature.addmm(
                        e.DeviceKind.CUDA,
                        oo.tensor_operand("bias", bb),
                        oo.tensor_operand("a", a),
                        oo.tensor_operand("b", b),
                        oo.tensor_operand("out", out),
                        alpha,
                        beta,
                    )
                    sel = self.reg.validate(
                        "triton_tiled_gemm",
                        sig,
                        matmul_config(),
                        e.FusedLoops(),
                        self.ctx.native,
                    )
                    kernel, _ = self.run_generated(
                        "triton_tiled_gemm",
                        matmul_config(),
                        sig,
                        e.FusedLoops(),
                        (bb, a, b),
                        out,
                    )
                    msg = (
                        f"k={k} bias={tuple(bb.shape)} contig={bb.is_contiguous()} alpha={alpha!r} beta={beta!r} "
                        f"plan={sel.addmm.canonical()}"
                    )
                    self.assertTrue(torch.equal(out.isnan(), ref.isnan()), msg=msg)
                    self.assertTrue(
                        torch.equal(out.isinf() & (out > 0), ref.isinf() & (ref > 0)),
                        msg=msg,
                    )
                    if not sel.addmm.read_operands:
                        self.assertNotIn("tl.dot", kernel.source, msg=msg)
                        self.assertTrue(
                            torch.equal(out.view(torch.int32), ref.view(torch.int32)),
                            msg=msg,
                        )
                    else:
                        self.assertEqual(out, ref, equal_nan=True, msg=msg)

    def test_addmm_alpha_zero_skips_prologue_reads(self):
        e, idx = E(), E().LoopIndex
        m, n, k = 20, 24, 30
        a, b, bias = (
            make_matrix(m, k, False, 13),
            make_matrix(k, n, False, 14),
            torch.randn(m, n, device="cuda"),
        )
        poison = torch.full((k,), float("nan"), device="cuda")
        out = torch.empty(m, n, device="cuda")
        sig = e.OpSignature.addmm(
            e.DeviceKind.CUDA,
            oo.tensor_operand("bias", bias),
            oo.tensor_operand("a", a),
            oo.tensor_operand("b", b),
            oo.tensor_operand("out", out),
            0.0,
            1.0,
        )
        pb = e.RegionBuilder(
            name="pro_a",
            inputs=["a", "poison"],
            result="a_p",
            result_elements=m * k,
            ranges=[("i", 0, m), ("kk", 0, k)],
        )
        a_idx = idx.var("i") * idx.const(k) + idx.var("kk")
        pb.store(
            "a_p",
            a_idx,
            pb.binary("mul", pb.load("a", a_idx), pb.load("poison", idx.var("kk"))),
        )
        fused = e.FusedLoops()
        fused.add_prologue(1, pb.build())
        fused.add_extra_input("poison", oo.tensor_layout(poison))
        kernel, _ = self.run_generated(
            "triton_tiled_gemm", matmul_config(), sig, fused, (bias, a, b, poison), out
        )
        self.assertNotIn("tl.dot", kernel.source)
        self.assertTrue(
            torch.equal(out, torch.addmm(bias, a * poison, b, beta=1.0, alpha=0.0))
        )

    def test_addmm_lt_disabled_policy_subprocess(self):
        script = (
            "import torch\n"
            "from torch._inductor.codegen import loops_ir_op_options as oo\n"
            "e = oo.load_op_options_native()\n"
            "ctx = oo.freeze_codegen_context(torch.device('cuda', 0), 'inductor')\n"
            "assert ctx.native.addmm_lt_policy == e.AddmmLtPolicy.CUDA_DISABLED\n"
            "a = torch.randn(33, 45, device='cuda'); a[0, 0] = float('nan')\n"
            "b = torch.randn(45, 70, device='cuda'); bias = torch.randn(70, device='cuda')\n"
            "ref = torch.addmm(bias, a, b, beta=1.0, alpha=0.0)\n"
            "assert not ref.isnan().any()\n"
            "out = torch.empty(33, 70, device='cuda')\n"
            "sig = e.OpSignature.addmm(e.DeviceKind.CUDA, oo.tensor_operand('bias', bias), oo.tensor_operand('a', a),\n"
            "    oo.tensor_operand('b', b), oo.tensor_operand('out', out), 0.0, 1.0)\n"
            "cfg = e.MatmulConfig(block_m=32, block_n=32, block_k=16, group_m=8, num_warps=4, num_stages=2)\n"
            "k = oo.codegen('triton_tiled_gemm', cfg, sig, e.FusedLoops(), ctx)\n"
            "oo.compile_generated_kernel(k, ctx).launch(bias, a, b, out); torch.cuda.synchronize()\n"
            "assert torch.equal(out, ref.contiguous()), 'mismatch'\n"
            "print('lt-disabled ok')\n"
        )
        env = {**os.environ, "DISABLE_ADDMM_CUDA_LT": "1"}
        res = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(res.returncode, 0, res.stderr[-3000:])
        self.assertIn("lt-disabled ok", res.stdout)

    def test_launch_rejects_overlap_and_short_storage(self):
        e, idx = E(), E().LoopIndex
        n = 32
        storage = torch.randn(3 * n * n, device="cuda")
        a, b = storage[: n * n].view(n, n), torch.randn(n, n, device="cuda")
        sig = mm_sig(a, b, torch.empty(n, n, device="cuda"))
        kernel = oo.codegen(
            "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), self.ctx
        )
        compiled = oo.compile_generated_kernel(kernel, self.ctx)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            compiled.launch(a, b, storage[n * n // 2 : n * n // 2 + n * n].view(n, n))
        disjoint = storage[n * n : 2 * n * n].view(n, n)
        compiled.launch(a, b, disjoint)
        torch.cuda.synchronize()
        self.assertEqual(disjoint, torch.mm(a, b))
        # Epilogue store_offset: the launch enforces the touched storage range.
        bld = e.RegionBuilder(
            name="epi",
            inputs=["acc"],
            result="y",
            result_elements=2 + n * n,
            ranges=[("i", 0, n), ("j", 0, n)],
        )
        canon = idx.var("i") * idx.const(n) + idx.var("j")
        bld.store("y", canon + idx.const(2), bld.relu(bld.load("acc", canon)))
        fused = e.FusedLoops()
        fused.set_epilogue("acc", bld.build(), store_offset=2)
        kernel = oo.codegen("triton_tiled_gemm", matmul_config(), sig, fused, self.ctx)
        self.assertEqual(kernel.launch.args[-1].required_elements, 2 + n * n)
        compiled = oo.compile_generated_kernel(kernel, self.ctx)
        with self.assertRaisesRegex(ValueError, "storage holds"):
            compiled.launch(a, b, torch.empty(n, n, device="cuda"))
        root = torch.full((2 + n * n,), float("nan"), device="cuda")
        compiled.launch(a, b, torch.as_strided(root, (n, n), (n, 1)))
        torch.cuda.synchronize()
        self.assertTrue(root[:2].isnan().all())
        self.assertEqual(root[2:].view(n, n), torch.relu(torch.mm(a, b)))

    def test_zero_sized_gemms(self):
        e = E()
        a, b = torch.randn(0, 5, device="cuda"), torch.randn(5, 7, device="cuda")
        out = torch.empty(0, 7, device="cuda")
        sig = mm_sig(a, b, out)
        kernel, _ = self.run_generated(
            "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), (a, b), out
        )
        self.assertEqual(kernel.launch.grid[0], 0)
        a, b = torch.randn(9, 0, device="cuda"), torch.randn(0, 7, device="cuda")
        bias = torch.randn(7, device="cuda")
        out = torch.full((9, 7), float("nan"), device="cuda")
        sig = e.OpSignature.addmm(
            e.DeviceKind.CUDA,
            oo.tensor_operand("bias", bias),
            oo.tensor_operand("a", a),
            oo.tensor_operand("b", b),
            oo.tensor_operand("out", out),
            1.0,
            3.0,
        )
        self.run_generated(
            "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), (bias, a, b), out
        )
        self.assertEqual(out, torch.addmm(bias, a, b, beta=3.0))

    def test_gemm_epilogue_and_prologue_fusion(self):
        e, idx = E(), E().LoopIndex
        m, n, k = 65, 96, 70
        a, b = make_matrix(m, k, False, 5), make_matrix(k, n, False, 6)
        bias = torch.randn(n, device="cuda")
        out = torch.empty(m, n, device="cuda")
        sig = mm_sig(a, b, out)
        for negate in (False, True):
            body = epilogue_bias_relu_scale(
                sig.output.layout, [0, 1], scale=0.75, negate=negate
            )
            fused = fused_epilogue(body, [("bias", oo.tensor_layout(bias))])
            for cfg in self.gemm_configs(sig):
                self.run_generated(
                    "triton_tiled_gemm", cfg, sig, fused, (a, b, bias), out
                )
                signed = -bias if negate else bias
                self.assertEqual(out, torch.relu(torch.mm(a, b) + signed) * 0.75)
        # Prologue: A' = relu(A) * colscale[k]  (broadcast extra input).
        colscale = torch.randn(k, device="cuda")
        pb = e.RegionBuilder(
            name="pro_a",
            inputs=["a", "colscale"],
            result="a_p",
            result_elements=m * k,
            ranges=[("i", 0, m), ("kk", 0, k)],
        )
        a_idx = idx.var("i") * idx.const(k) + idx.var("kk")
        v = pb.binary(
            "mul", pb.relu(pb.load("a", a_idx)), pb.load("colscale", idx.var("kk"))
        )
        pb.store("a_p", a_idx, v)
        fused = e.FusedLoops()
        fused.add_prologue(0, pb.build())
        fused.add_extra_input("colscale", oo.tensor_layout(colscale))
        for cfg in self.gemm_configs(sig):
            self.run_generated(
                "triton_tiled_gemm", cfg, sig, fused, (a, b, colscale), out
            )
            self.assertEqual(out, torch.mm(torch.relu(a) * colscale, b))

    def test_native_mm_bias_epilogue_is_one_addmm(self):
        e = E()
        a, b = make_matrix(33, 20, True, 7), make_matrix(20, 48, False, 8)
        bias = torch.randn(48, device="cuda")
        out = torch.empty(33, 48, device="cuda")
        sig = mm_sig(a, b, out)
        e_idx = E().LoopIndex
        bld = e.RegionBuilder(
            name="epi",
            inputs=["acc", "bias"],
            result="fused",
            result_elements=33 * 48,
            ranges=[("i", 0, 33), ("j", 0, 48)],
        )
        o = e_idx.var("i") * e_idx.const(48) + e_idx.var("j")
        bld.store(
            "fused",
            o,
            bld.binary("add", bld.load("bias", e_idx.var("j")), bld.load("acc", o)),
        )
        fused = fused_epilogue(bld.build(), [("bias", oo.tensor_layout(bias))])
        assert_rejected(
            self,
            e.RejectReason.NUMERICS_FENCE,
            lambda: oo.codegen(
                "native_aten_mm_bias_epilogue",
                e.NativeConfig(),
                sig,
                fused,
                self.strict,
            ),
        )
        call = oo.codegen(
            "native_aten_mm_bias_epilogue", e.NativeConfig(), sig, fused, self.ctx
        )
        self.assertEqual(call.target, "aten.addmm.default")
        self.assertTrue(
            torch.equal(oo.load_native_call(call)(a, b, bias), torch.addmm(bias, a, b))
        )

    def reduction_case(self, x, dim, keepdim, kind_name):
        e = E()
        kind = e.OpKind.SUM_DIM if kind_name == "sum" else e.OpKind.MEAN_DIM
        ref = (torch.sum if kind_name == "sum" else torch.mean)(x, dim, keepdim=keepdim)
        out = torch.empty_like(ref, memory_format=torch.contiguous_format)
        sig = e.OpSignature.reduction(
            kind,
            e.DeviceKind.CUDA,
            oo.tensor_operand("x", x),
            [dim],
            keepdim,
            None,
            oo.tensor_operand("out", out),
        )
        return sig, ref, out

    def test_row_reductions(self):
        e = E()
        shapes = [
            ((5, 1), 1),
            ((7, 13), 1),
            ((33, 1000), 1),
            ((9, 5000), -1),
            ((6, 4, 37), 1),
            ((3, 0), 1),
        ]
        for shape, dim in shapes:
            for transposed in (False, True):
                x = (
                    torch.randn(shape[::-1], device="cuda").t()
                    if transposed and len(shape) == 2
                    else torch.randn(shape, device="cuda")
                )
                for kind_name in ("sum", "mean"):
                    for keepdim in (False, True):
                        sig, ref, out = self.reduction_case(x, dim, keepdim, kind_name)
                        cands = self.reg.candidate_configs(
                            "triton_row_reduction", sig, self.ctx.native
                        )
                        for cfg in {
                            cands[0].canonical(): cands[0],
                            cands[-1].canonical(): cands[-1],
                        }.values():
                            self.run_generated(
                                "triton_row_reduction",
                                cfg,
                                sig,
                                e.FusedLoops(),
                                (x,),
                                out,
                            )
                            self.assertEqual(
                                out,
                                ref,
                                equal_nan=True,
                                msg=f"{kind_name} {shape} dim={dim} t={transposed} keep={keepdim} {cfg}",
                            )
                        if x.is_contiguous() or ref.dim() <= 1:
                            call = oo.codegen(
                                "native_aten_reduction",
                                e.NativeConfig(),
                                sig,
                                e.FusedLoops(),
                                self.strict,
                            )
                            self.assertTrue(
                                torch.equal(oo.load_native_call(call)(x), ref)
                                or (
                                    ref.isnan().all()
                                    and oo.load_native_call(call)(x).isnan().all()
                                )
                            )

    def test_reduction_prologue_epilogue_fusion(self):
        e, idx = E(), E().LoopIndex
        rows, r = 24, 1500
        x = torch.randn(rows, r, device="cuda")
        colw = torch.randn(r, device="cuda")
        shift = torch.randn(1, device="cuda")
        for kind_name in ("sum", "mean"):
            sig, _, out = self.reduction_case(x, 1, False, kind_name)
            pb = e.RegionBuilder(
                name="pro",
                inputs=["x", "colw"],
                result="xp",
                result_elements=rows * r,
                ranges=[("i", 0, rows), ("j", 0, r)],
            )
            xi = idx.var("i") * idx.const(r) + idx.var("j")
            v = pb.binary(
                "add",
                pb.binary("mul", pb.load("x", xi), pb.load("colw", idx.var("j"))),
                pb.constant(1.0),
            )
            pb.store("xp", xi, v)
            eb = e.RegionBuilder(
                name="epi",
                inputs=["acc", "shift"],
                result="y",
                result_elements=rows,
                ranges=[("i", 0, rows)],
            )
            y = eb.relu(
                eb.binary(
                    "add", eb.load("acc", idx.var("i")), eb.load("shift", idx.const(0))
                )
            )
            eb.store("y", idx.var("i"), y)
            fused = e.FusedLoops()
            fused.add_prologue(0, pb.build())
            fused.set_epilogue("acc", eb.build())
            fused.add_extra_input("colw", oo.tensor_layout(colw))
            fused.add_extra_input("shift", oo.tensor_layout(shift))
            red = torch.sum if kind_name == "sum" else torch.mean
            ref = torch.relu(red(x * colw + 1.0, 1) + shift)
            for cfg in self.reg.candidate_configs(
                "triton_row_reduction", sig, self.ctx.native
            )[:3]:
                self.run_generated(
                    "triton_row_reduction", cfg, sig, fused, (x, colw, shift), out
                )
                self.assertEqual(out, ref, msg=f"{kind_name} fused {cfg}")

    def test_config_changes_compiled_kernel(self):
        e = E()
        a, b = make_matrix(64, 64, False, 9, 0.35), make_matrix(64, 64, False, 10, 0.35)
        out = torch.empty(64, 64, device="cuda")
        sig = mm_sig(a, b, out)
        small = matmul_config(num_warps=2, num_stages=2)
        large = matmul_config(
            block_m=64, block_n=64, block_k=64, num_warps=8, num_stages=3
        )
        meta = []
        for cfg in (small, large):
            _, compiled = self.run_generated(
                "triton_tiled_gemm", cfg, sig, e.FusedLoops(), (a, b), out
            )
            self.assertEqual(out, torch.mm(a, b))
            meta.append(compiled.metadata())
            ptx = compiled.ptx()
            if ptx is not None:  # best effort: the static launcher keeps only the cubin
                self.assertIn(f".maxntid {cfg.num_warps * 32}, 1, 1", ptx)
        self.assertEqual([(m.num_warps, m.num_stages) for m in meta], [(2, 2), (8, 3)])
        self.assertLess(meta[0].shared_bytes, meta[1].shared_bytes)

    def test_frozen_context_ignores_later_globals(self):
        e = E()
        sig = mm_sig_shapes(64, 512, 512)
        before = oo.codegen(
            "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), self.ctx
        )
        saved = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            after = oo.codegen(
                "triton_tiled_gemm", matmul_config(), sig, e.FusedLoops(), self.ctx
            )
            refrozen = oo.freeze_codegen_context(
                torch.device("cuda", self.ctx.device_index), "inductor"
            )
        finally:
            torch.backends.cuda.matmul.fp32_precision = saved
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision, saved)
        self.assertEqual(
            (before.cache_key, before.source), (after.cache_key, after.source)
        )
        self.assertEqual(
            refrozen.native.stock_fp32_matmul_precision, e.MatmulPrecision.TF32
        )
        cands = self.reg.candidate_configs("triton_tiled_gemm", sig, refrozen.native)
        self.assertEqual({c.matrix_precision for c in cands}, {e.MatmulPrecision.TF32})


if __name__ == "__main__":
    run_tests()
