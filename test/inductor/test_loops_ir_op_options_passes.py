# Owner(s): ["module: inductor"]
"""decode_op_signatures / form_fused_op_regions / enumerate_op_candidates.

CPU: standalone C++ pass unit and Python-built core KernelPrograms. GPU:
real frontend programs, and isolated execution of plan variants through the
option codegen (no backend integration, no timing, no selection)."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch._inductor.codegen import loops_ir_op_options as oo
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()
OPT_DIR = (
    Path(oo.__file__).resolve().parents[2]
    / "csrc/inductor/experimental/loops_ir/op_options"
)
PASS_SOURCES = [
    "../ir.cpp",
    "op_types.cpp",
    "op_config.cpp",
    "fused_region.cpp",
    "option_registry.cpp",
    "gemm_options.cpp",
    "reduction_options.cpp",
    "program_facts.cpp",
    "op_decode.cpp",
    "op_fusion_regions.cpp",
    "op_candidates.cpp",
]
aten = torch.ops.aten


def core():
    from torch._inductor.loops_ir import load_cpp_ir

    return load_cpp_ir()


def E():
    return oo.load_op_options_native()


def contiguous(sizes):
    return list(torch.empty(sizes, device="meta").stride())


class ProgramBuilder:
    """Core KernelProgram construction through the core extension bindings."""

    def __init__(self, numerics="strict_eager"):
        self.c = core()
        self.p = self.c.KernelProgram("prog", numerics)

    def input(self, name, elems):
        self.p.add_input(self.c.Buffer(name, "f32", elems))

    def desc(self, name, buf, sizes, strides=None, off=0):
        return self.c.Descriptor(
            name, buf, list(sizes), strides or contiguous(sizes), off
        )

    def extern(self, name, target, args, kwargs, arg_descs, out_sizes):
        n = 1
        for s in out_sizes:
            n *= s
        descs = arg_descs + [self.desc("result", name, out_sizes)]
        self.p.add_extern_call(
            self.c.ExternCall.structured(
                name,
                target,
                args,
                [self.c.CallKeyword(k, v) for k, v in kwargs],
                [self.c.Buffer(name, "f32", n)],
                descs,
            )
        )

    def output(self, buf, sizes):
        self.p.add_output(self.desc(f"out{len(self.p.outputs)}", buf, sizes))

    def T(self, name):
        return self.c.CallArg.tensor(name)

    def F(self, v):
        return self.c.CallArg.floating(v)


def strict_ctx_cpu(numerics="strict_eager"):
    e = E()
    strict = numerics == "strict_eager"
    native = e.CodegenContext(
        numerics=e.NumericsPolicy.STRICT_EAGER if strict else e.NumericsPolicy.INDUCTOR,
        device=e.DeviceKind.CUDA,
        compute_capability=90,
        max_shared_memory_bytes=232448,
        stock_fp32_matmul_precision=e.MatmulPrecision.IEEE,
        enable_fp_fusion=not strict,
        disable_ftz=strict,
        division_rounding=False,
    )
    return oo.FrozenCodegenContext(native, 0, None)


class TestOpPassesNativeUnit(TestCase):
    def test_standalone_cpp_passes(self):
        cxx = os.environ.get("CXX", "g++")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "op_passes_cpu_test")
            subprocess.run(
                [
                    cxx,
                    "-std=c++17",
                    "-O0",
                    "-Wall",
                    *PASS_SOURCES,
                    "tests/op_passes_cpu_test.cpp",
                    "-o",
                    exe,
                ],
                cwd=OPT_DIR,
                check=True,
            )
            out = subprocess.run(
                [exe], check=True, capture_output=True, text=True
            ).stdout
        self.assertIn("checks passed", out)


class TestOpPassesCoreProgram(TestCase):
    def mm_addmm_program(self, numerics="strict_eager"):
        b = ProgramBuilder(numerics)
        for name, n in (("a", 20), ("w", 35), ("c", 7), ("s", 4)):
            b.input(name, n)
        b.extern(
            "mm",
            "aten.mm.default",
            [],
            [("self", b.T("a")), ("mat2", b.T("w"))],
            [b.desc("arg0", "a", (4, 5)), b.desc("arg1", "w", (5, 7))],
            (4, 7),
        )
        b.extern(
            "addmm",
            "aten.addmm.default",
            [b.T("c"), b.T("a"), b.T("w")],
            [("beta", b.F(-0.0)), ("alpha", b.F(0.1))],
            [
                b.desc("arg0", "c", (4, 7), [0, 1]),
                b.desc("arg1", "a", (4, 5)),
                b.desc("arg2", "w", (5, 7)),
            ],
            (4, 7),
        )
        b.extern(
            "r",
            "aten.rand_like.default",
            [b.T("s")],
            [],
            [b.desc("arg0", "s", (4,))],
            (4,),
        )
        for name, sizes in (("mm", (4, 7)), ("addmm", (4, 7)), ("r", (4,))):
            b.output(name, sizes)
        return b

    def test_decode_from_python_built_core_program(self):
        e = E()
        b = self.mm_addmm_program()
        d = oo.decode_op_signatures(b.p)
        self.assertEqual(
            [op.status for op in d.ops],
            [
                e.DecodeStatus.DECODED,
                e.DecodeStatus.DECODED,
                e.DecodeStatus.UNKNOWN_TARGET,
            ],
        )
        mm, addmm, rand = d.ops
        self.assertEqual([t.name for t in mm.signature.inputs], ["a", "w"])
        self.assertEqual(addmm.signature.matmul.alpha.hex(), (0.1).hex())
        self.assertEqual(str(addmm.signature.matmul.beta), "-0.0")
        self.assertEqual(list(addmm.signature.inputs[0].layout.strides), [0, 1])
        self.assertIsNone(rand.signature)
        self.assertEqual(rand.fallback.target, "aten.rand_like.default")
        self.assertEqual(mm.fallback.name, "mm")
        with self.assertRaises(AttributeError):
            mm.status = e.DecodeStatus.UNSUPPORTED
        with self.assertRaises(AttributeError):
            mm.input_bindings[0].root_offset = 3

    def test_unknown_kwarg_and_wrong_arg_kind_fail_closed(self):
        e = E()
        b = ProgramBuilder()
        b.input("a", 20)
        b.input("w", 35)
        b.extern(
            "mm",
            "aten.mm.default",
            [b.T("a"), b.c.CallArg.integer(3)],
            [],
            [b.desc("arg0", "a", (4, 5))],
            (4, 7),
        )
        (op,) = oo.decode_op_signatures(b.p).ops
        self.assertEqual(op.status, e.DecodeStatus.MALFORMED_ABI)
        self.assertIsNone(op.signature)

    def test_strict_plan_is_native_baseline_only(self):
        e = E()
        b = self.mm_addmm_program()
        d = oo.decode_op_signatures(b.p)
        r = oo.form_fused_op_regions(b.p, d)
        plan = oo.enumerate_op_candidates(r, d, strict_ctx_cpu())
        self.assertEqual(len(plan.ops), 2)
        for op in plan.ops:
            self.assertEqual(len(op.candidates), 1)
            (base,) = op.candidates
            self.assertTrue(base.baseline)
            self.assertEqual(
                (base.kind, base.option_id),
                (e.CandidateKind.NATIVE_OPTION, "native_aten_gemm"),
            )
            self.assertTrue(
                any(rj.reason == e.RejectReason.NUMERICS_FENCE for rj in op.rejections)
            )
        with self.assertRaises(e.OptionRejected):
            oo.enumerate_op_candidates(r, d, strict_ctx_cpu("inductor"))

    def test_inductor_plan_enumerates_registry_manifests(self):
        e = E()
        b = self.mm_addmm_program("inductor")
        d = oo.decode_op_signatures(b.p)
        ctx = strict_ctx_cpu("inductor")
        plan = oo.enumerate_op_candidates(oo.form_fused_op_regions(b.p, d), d, ctx)
        reg = e.default_registry()
        for op in plan.ops:
            base, *rest = op.candidates
            self.assertTrue(base.baseline and base.option_id == "native_aten_gemm")
            self.assertFalse(any(c.baseline for c in rest))
            generated = [
                c.config.canonical()
                for c in rest
                if c.kind == e.CandidateKind.GENERATED_OPTION
            ]
            manifest = reg.candidate_configs(
                "triton_tiled_gemm", op.variants[0].signature, ctx.native
            )
            self.assertEqual(generated, [c.canonical() for c in manifest])
            self.assertEqual(
                len({c.cache_key for c in op.candidates if c.cache_key}),
                len(op.candidates),
            )


def capture_env(gm, args):
    env = {}

    class Capture(torch.fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            env[n.name] = out
            return out

    Capture(gm).run(*args)
    return env


def frontend_kernel_program(fn, args):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor.loops_ir import lower_fx_to_cpp

    gm = make_fx(fn)(*args)
    program = lower_fx_to_cpp(gm, args, executable=True)
    return gm, core().run_pipeline(program).kernel_program


def model(x, w, bias, b1, b2, y):
    m = aten.mm.default(torch.relu(x), w)
    e = torch.relu(m + bias)
    a = aten.addmm.default(bias, x, w, beta=0.5, alpha=2)
    bm = aten.bmm.default(b1, b2.transpose(1, 2))
    s = aten.sum.dim_IntList(torch.relu(y), [-1], True)
    mn = aten.mean.dim(y.t(), [0])
    return e, a, bm, s, mn


def model_args():
    g = torch.Generator(device="cuda").manual_seed(0)

    def r(*s):
        return torch.randn(*s, device="cuda", generator=g)

    return [r(37, 48), r(48, 65), r(65), r(3, 17, 33), r(3, 9, 33), r(21, 70)]


@unittest.skipUnless(HAS_CUDA, "requires CUDA")
class TestOpPassesFrontend(TestCase):
    def setUp(self):
        super().setUp()
        self.assertNotEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        self.args = model_args()
        self.gm, self.kp = frontend_kernel_program(model, self.args)
        self.decoded = oo.decode_op_signatures(self.kp)

    def test_decode_real_frontend_program(self):
        e = E()
        ops = {
            op.fallback.target: op
            for op in self.decoded.ops
            if op.status == e.DecodeStatus.DECODED
        }
        self.assertEqual(
            set(ops),
            {
                "aten.mm.default",
                "aten.addmm.default",
                "aten.bmm.default",
                "aten.sum.dim_IntList",
                "aten.mean.dim",
            },
        )
        addmm = ops["aten.addmm.default"].signature.matmul
        self.assertEqual((addmm.alpha, addmm.beta), (2.0, 0.5))
        s = ops["aten.sum.dim_IntList"].signature.reduction_attrs
        self.assertEqual((list(s.dims), s.keepdim), ([1], True))
        mean = ops["aten.mean.dim"].signature
        self.assertEqual(list(mean.inputs[0].layout.strides), [1, 70])
        bmm = ops["aten.bmm.default"].signature
        self.assertEqual(
            (bmm.gemm_dims.batch, list(bmm.inputs[1].layout.strides)), (3, [297, 1, 33])
        )

    def test_regions_from_frontend(self):
        regions = oo.form_fused_op_regions(self.kp, self.decoded)
        by_target = {
            self.decoded.ops[s.decoded_index].fallback.target: s for s in regions.ops
        }
        mm = by_target["aten.mm.default"]
        shapes = sorted((list(v.prologue_operands), v.epilogue) for v in mm.variants)
        self.assertEqual(shapes, [([], False), ([], True), ([0], False), ([0], True)])
        full = next(v for v in mm.variants if v.epilogue and v.prologue_operands)
        self.assertIsNone(full.input_bindings[0])
        self.assertEqual(len(full.consumed_nodes), 3)
        self.assertEqual(full.placement, max(full.consumed_nodes))
        red = by_target["aten.sum.dim_IntList"]
        self.assertTrue(any(list(v.prologue_operands) == [0] for v in red.variants))
        # x feeds both relu(x)->mm and addmm directly: addmm has no prologue.
        self.assertEqual(len(by_target["aten.addmm.default"].variants), 1)

    def bind(self, root_env, binding, layout):
        root = root_env[binding.root]
        return torch.as_strided(
            root,
            list(layout.sizes),
            list(layout.strides),
            root.storage_offset() + binding.root_offset,
        )

    def run_candidate(self, op, cand, ctx, root_env):
        e = E()
        variant = op.variants[cand.variant]
        sig = variant.signature
        inputs = [
            self.bind(root_env, b, sig.inputs[i].layout)
            for i, b in enumerate(variant.input_bindings)
            if b is not None
        ]
        extras = [root_env[b.root] for b in variant.extra_bindings]
        if cand.kind == e.CandidateKind.NATIVE_OPTION:
            call = oo.codegen(
                cand.option_id, cand.config, sig, variant.fused_loops, ctx
            )
            return oo.load_native_call(call)(*inputs, *extras)
        out_root = torch.full(
            (variant.output_binding.root_elements,), float("nan"), device="cuda"
        )
        out = torch.as_strided(
            out_root, list(sig.output.layout.sizes), list(sig.output.layout.strides)
        )
        kernel = oo.codegen(cand.option_id, cand.config, sig, variant.fused_loops, ctx)
        oo.compile_generated_kernel(kernel, ctx).launch(*inputs, *extras, out)
        return out_root

    def environment(self):
        env = capture_env(self.gm, self.args)
        roots = {
            n: v.contiguous().reshape(-1)
            for n, v in env.items()
            if isinstance(v, torch.Tensor)
        }
        return env, roots

    def test_strict_baselines_execute_bitwise(self):
        env, roots = self.environment()
        ctx = oo.freeze_codegen_context(
            torch.device("cuda", torch.cuda.current_device()), "strict_eager"
        )
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(self.kp, self.decoded), self.decoded, ctx
        )
        self.assertEqual(len(plan.ops), 5)
        for op in plan.ops:
            (base,) = op.candidates
            got = self.run_candidate(op, base, ctx, roots)
            self.assertTrue(
                torch.equal(got, env[op.variants[0].produces]), msg=op.fallback.target
            )

    def test_inductor_variants_execute_against_eager(self):
        e = E()
        env, roots = self.environment()
        ctx = oo.freeze_codegen_context(
            torch.device("cuda", torch.cuda.current_device()), "inductor"
        )
        # Plans are policy-checked against the program, so plan an inductor-numerics
        # lowering of the same graph.
        program = core().run_pipeline(self.inductor_program()).kernel_program
        decoded = oo.decode_op_signatures(program)
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(program, decoded), decoded, ctx
        )
        ran = 0
        for op in plan.ops:
            first_generated = {}
            for cand in op.candidates:
                if cand.kind == e.CandidateKind.GENERATED_OPTION:
                    first_generated.setdefault(cand.variant, cand)
            chosen = [op.candidates[0], *first_generated.values()]
            chosen += [
                c for c in op.candidates[1:] if c.kind == e.CandidateKind.NATIVE_OPTION
            ]
            for cand in chosen:
                expected = env[op.variants[cand.variant].produces]
                got = self.run_candidate(op, cand, ctx, roots)
                got = torch.as_strided(got, expected.shape, expected.stride())
                msg = f"{op.fallback.target} variant={cand.variant} {cand.option_id} {cand.config!r}"
                if cand.baseline:
                    self.assertTrue(torch.equal(got, expected), msg=msg)
                else:
                    self.assertEqual(got, expected, msg=msg)
                ran += 1
        self.assertGreaterEqual(ran, 14)

    def inductor_program(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._inductor.loops_ir import lower_fx_to_cpp

        program = lower_fx_to_cpp(
            make_fx(model)(*self.args), self.args, executable=True
        )
        rebuilt = core().Program(program.name, "inductor")
        for b in program.inputs:
            rebuilt.add_input(b)
        for d in program.input_descriptors:
            rebuilt.add_input_descriptor(d)
        for node in program.nodes:
            kind = type(node).__name__
            getattr(
                rebuilt,
                {
                    "View": "add_view",
                    "Compute": "add_compute",
                    "ExternCall": "add_extern_call",
                }[kind],
            )(node)
        for d in program.outputs:
            rebuilt.add_output(d)
        return rebuilt


@unittest.skipUnless(HAS_CUDA, "requires CUDA")
class TestOpPassesOffsets(TestCase):
    def test_prologue_and_epilogue_with_view_and_store_offsets(self):
        e, c = E(), core()
        idx = c.Index

        def lin(terms, off):
            expr = idx.constant(off)
            for var, coeff in terms:
                expr = idx.binary(
                    "add",
                    expr,
                    idx.binary("mul", idx.variable(var), idx.constant(coeff)),
                )
            return expr

        m, k, n = 6, 5, 7
        b = ProgramBuilder("inductor")
        b.input("x", 3 + m * k)
        b.input("w", k * n)
        b.input("bias", n)
        pro = c.Compute(
            "pro",
            ["x"],
            [c.Buffer("pro", "f32", 3 + m * k)],
            [c.Range("i", 0, m), c.Range("j", 0, k)],
            [],
            [],
        )
        pro.add_post_instruction(
            c.Instruction.load(0, "x", lin([("i", k), ("j", 1)], 3))
        )
        pro.add_post_instruction(c.Instruction.relu(1, 0))
        pro.add_post_instruction(
            c.Instruction.store("pro", lin([("i", k), ("j", 1)], 3), 1)
        )
        b.p.add_function(c.KernelFunction("k_pro", [pro], "inductor"))
        b.p.add_view(c.View("pro_v", "pro", [m, k], [k, 1], 3))
        b.extern(
            "mm",
            "aten.mm.default",
            [b.T("pro_v"), b.T("w")],
            [],
            [b.desc("arg0", "pro_v", (m, k)), b.desc("arg1", "w", (k, n))],
            (m, n),
        )
        epi = c.Compute(
            "y",
            ["mm", "bias"],
            [c.Buffer("y", "f32", 2 + m * n)],
            [c.Range("i", 0, m), c.Range("j", 0, n)],
            [],
            [],
        )
        epi.add_post_instruction(
            c.Instruction.load(0, "mm", lin([("i", n), ("j", 1)], 0))
        )
        epi.add_post_instruction(c.Instruction.load(1, "bias", lin([("j", 1)], 0)))
        epi.add_post_instruction(c.Instruction.binary(2, "add", 0, 1))
        epi.add_post_instruction(
            c.Instruction.store("y", lin([("i", n), ("j", 1)], 2), 2)
        )
        b.p.add_function(c.KernelFunction("k_y", [epi], "inductor"))
        b.p.add_output(b.desc("out0", "y", (m, n), None, 2))

        decoded = oo.decode_op_signatures(b.p)
        (op,) = [o for o in decoded.ops if o.status == e.DecodeStatus.DECODED]
        self.assertEqual(
            (op.input_bindings[0].root, op.input_bindings[0].root_offset), ("pro", 3)
        )
        regions = oo.form_fused_op_regions(b.p, decoded)
        (variants,) = [s.variants for s in regions.ops]
        full = next(
            v for v in variants if v.epilogue and list(v.prologue_operands) == [0]
        )
        self.assertEqual(sorted(full.eliminated_buffers), ["mm", "pro"])
        self.assertTrue(full.accepted_by("triton_tiled_gemm"))
        # Plain acc + bias epilogue writing at offset 2: native addmm (fresh tensor) must refuse it.
        epi_only = next(v for v in variants if v.epilogue and not v.prologue_operands)
        self.assertFalse(epi_only.accepted_by("native_aten_mm_bias_epilogue"))

        ctx = oo.freeze_codegen_context(
            torch.device("cuda", torch.cuda.current_device()), "inductor"
        )
        plan = oo.enumerate_op_candidates(regions, decoded, ctx)
        (planned,) = plan.ops
        vi = next(
            i
            for i, v in enumerate(planned.variants)
            if v.epilogue and list(v.prologue_operands) == [0]
        )
        cand = next(cd for cd in planned.candidates if cd.variant == vi)
        variant = planned.variants[vi]
        x, w, bias = (
            torch.randn(3 + m * k, device="cuda"),
            torch.randn(k, n, device="cuda"),
            torch.randn(n, device="cuda"),
        )
        roots = {"x": x, "w": w.reshape(-1), "bias": bias}
        sig = variant.signature
        inputs = [
            torch.as_strided(
                roots[bd.root],
                list(sig.inputs[i].layout.sizes),
                list(sig.inputs[i].layout.strides),
                bd.root_offset,
            )
            for i, bd in enumerate(variant.input_bindings)
            if bd is not None
        ]
        extras = [roots[bd.root] for bd in variant.extra_bindings]
        y_root = torch.full(
            (variant.output_binding.root_elements,), float("nan"), device="cuda"
        )
        out = torch.as_strided(
            y_root, list(sig.output.layout.sizes), list(sig.output.layout.strides)
        )
        kernel = oo.codegen(cand.option_id, cand.config, sig, variant.fused_loops, ctx)
        oo.compile_generated_kernel(kernel, ctx).launch(*inputs, *extras, out)
        expected = torch.relu(x[3:].reshape(m, k)) @ w + bias
        self.assertTrue(torch.isnan(y_root[:2]).all())
        self.assertEqual(y_root[2:].reshape(m, n), expected)


if __name__ == "__main__":
    run_tests()
