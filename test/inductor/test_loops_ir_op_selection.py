# Owner(s): ["module: inductor"]
"""Isolated measured selection over LoopsIR op candidates.

CPU: fully mocked codegen/compile/load with the real ``autotune_op``
service (mocked timer). GPU: tiny GEMM / reduction programs through the
real SDK (``decode -> regions -> enumerate -> select``) with genuine
vetted serial timing (``benchmark_fn=None`` -> ``benchmark_gpu`` with
``is_vetted_benchmarking=True``). No custom CUDA-event timers anywhere.

Only ``loops_ir_op_selection`` (and its tests) are new: no backend, IR,
wrapper, or autotuner edits. Fusion is explicitly deferred (unfused-only
adapter): fused variants are counted, never measured.
"""

import unittest
from types import SimpleNamespace

import torch
from torch._inductor.codegen import loops_ir_op_options as oo
from torch._inductor.codegen import loops_ir_op_selection as sel
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()
aten = torch.ops.aten


def E():
    return oo.load_op_options_native()


def contiguous(sizes):
    return list(torch.empty(sizes, device="meta").stride())


def _bench_seq(values):
    it = iter(values)

    def bench(zero_arg):
        zero_arg()  # exercise the callable; timing is mocked
        return float(next(it))

    return bench


class FakeCache:
    """In-memory JSON cache mimicking LocalAutotuneCache serde."""

    def __init__(self):
        self.d: dict[str, object] = {}
        self.puts = 0

    def get(self, key):
        import copy

        v = self.d.get(key)
        return copy.deepcopy(v)

    def put(self, key, value):
        import copy

        self.d[key] = copy.deepcopy(value)
        self.puts += 1


def _mock_ctx(policy="inductor"):
    e = E()
    strict = policy == "strict_eager"
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


def _layout(sizes, strides=None):
    return SimpleNamespace(
        dtype="F32",
        sizes=list(sizes),
        strides=list(strides if strides is not None else contiguous(sizes)),
    )


def _binding(root, offset=0, elements=0):
    return SimpleNamespace(root=root, root_offset=offset, root_elements=elements)


def _mock_variant(node=3, n=4, bindings=None, extras=(), fused_extra_variants=0):
    layout = _layout((n,))
    sig = SimpleNamespace(
        inputs=[
            SimpleNamespace(name="a", layout=layout),
            SimpleNamespace(name="b", layout=layout),
        ],
        output=SimpleNamespace(name="out", layout=layout),
    )
    v0 = SimpleNamespace(
        signature=sig,
        fused_loops=SimpleNamespace(),
        input_bindings=list(bindings)
        if bindings is not None
        else [_binding("a"), _binding("b")],
        extra_bindings=list(extras),
        output_binding=_binding("out"),
        consumed_nodes=[node],
        placement=node,
        produces="out",
        eliminated_buffers=[],
    )
    variants = [v0] + [SimpleNamespace() for _ in range(fused_extra_variants)]
    return v0, variants


def _mock_cand(
    variant=0, kind="NATIVE_OPTION", option_id="native_aten_gemm", baseline=False, pos=0
):
    return SimpleNamespace(
        variant=variant,
        kind=kind,
        option_id=option_id,
        option_version=1,
        config=f"cfg-{pos}",
        cache_key=f"key-{pos}",
        baseline=baseline,
    )


def _mock_op(
    node=3,
    n=4,
    cands=None,
    fused_extra_variants=1,
    bindings=None,
    extras=(),
    target="aten.mm.default",
):
    v0, variants = _mock_variant(
        node=node,
        n=n,
        bindings=bindings,
        extras=extras,
        fused_extra_variants=fused_extra_variants,
    )
    if cands is None:
        cands = [
            _mock_cand(kind="GENERATED_OPTION", option_id="triton_tiled_gemm", pos=0),
            _mock_cand(
                kind="NATIVE_OPTION", option_id="native_aten_gemm", baseline=True, pos=1
            ),
        ]
    return SimpleNamespace(
        decoded_index=0,
        node_index=node,
        variants=variants,
        candidates=cands,
        fallback=SimpleNamespace(target=target),
    )


def _add_loader(call):
    def run(*tensors):
        return tensors[0] + tensors[1]

    return run


def _add_compiler(obj, ctx):
    class Launcher:
        def launch(self, *tensors):
            out = tensors[-1]
            out.copy_(tensors[0] + tensors[1])

    return Launcher()


def _mock_codegen(option_id, config, sig, fused_loops, ctx):
    return SimpleNamespace(source=f"mock-kernel:{option_id}:{config}")


class TestOpSelectionMock(TestCase):
    def test_fastest_generated_wins_records_frozen_facts(self):
        op = _mock_op(fused_extra_variants=2)
        ctx = _mock_ctx("inductor")
        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([5.0, 1.0]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.option_id, "triton_tiled_gemm")
        self.assertEqual(got.kind, "generated")
        self.assertEqual(got.variant, 0)
        self.assertTrue(got.measured and got.autotuned and not got.cache_hit)
        self.assertEqual(got.origin, "cold")
        self.assertEqual(got.consumed_nodes, (3,))
        self.assertEqual(got.placement, 3)
        self.assertEqual(got.produces, "out")
        self.assertEqual(got.fused_variants_deferred, 2)
        self.assertEqual(got.fusion_deferred_reason, sel.FUSION_DEFERRED_REASON)
        self.assertFalse(got.native_fallback)
        self.assertEqual(got.fallback_target, "aten.mm.default")
        self.assertEqual(got.numerics, "inductor")
        self.assertEqual(len(got.source_hash), 64)
        self.assertEqual(
            {k for k, _ in got.measurements_ms},
            {"v0-triton_tiled_gemm-000", "v0-native_aten_gemm-001"},
        )
        back = sel.SelectedOp.from_json(got.to_json())
        self.assertEqual(back, got)

    def test_strict_vetoes_inductor_only_generated(self):
        op = _mock_op()
        ctx = _mock_ctx("strict_eager")
        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([7.0]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.option_id, "native_aten_gemm")
        self.assertTrue(got.native_fallback)
        self.assertEqual(len(got.measurements_ms), 1)
        self.assertTrue(got.measured)

    def test_strict_singleton_baseline_not_measured(self):
        op = _mock_op(
            cands=[
                _mock_cand(
                    kind="NATIVE_OPTION",
                    option_id="native_aten_gemm",
                    baseline=True,
                    pos=0,
                )
            ],
            fused_extra_variants=0,
        )
        ctx = _mock_ctx("strict_eager")
        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.origin, "singleton")
        self.assertFalse(got.measured or got.autotuned)

    def test_cache_hit_no_remeasure(self):
        op = _mock_op()
        ctx = _mock_ctx("inductor")
        kw = dict(
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            device_str="cpu",
        )
        cache = FakeCache()
        r1 = sel.select_op_unfused(
            op, ctx, benchmark_fn=_bench_seq([1.0, 9.0]), cache=cache, **kw
        )
        self.assertFalse(r1.cache_hit)
        self.assertEqual(r1.origin, "cold")

        def boom(zero_arg):
            raise AssertionError("must not remeasure on cache hit")

        r2 = sel.select_op_unfused(op, ctx, benchmark_fn=boom, cache=cache, **kw)
        self.assertTrue(r2.cache_hit)
        self.assertEqual(r2.origin, "cache")
        self.assertEqual(r2.option_id, r1.option_id)
        self.assertEqual(r2.measured_ms, r1.measured_ms)

    def test_invalid_compiled_candidate_falls_back_to_native(self):
        op = _mock_op()
        ctx = _mock_ctx("inductor")

        def boom_compile(obj, ctx):
            raise RuntimeError("triton compile boom")

        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=boom_compile,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([4.0]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.option_id, "native_aten_gemm")
        self.assertTrue(got.native_fallback)

    def test_invalid_codegen_candidate_skipped(self):
        op = _mock_op()
        ctx = _mock_ctx("inductor")

        def picky_codegen(option_id, config, sig, fused_loops, ctx):
            if option_id == "triton_tiled_gemm":
                raise E().OptionRejected("bad config")
            return SimpleNamespace(source="native-src")

        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=picky_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([2.0]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.option_id, "native_aten_gemm")

    def test_all_fail_records_native_fallback(self):
        op = _mock_op()
        ctx = _mock_ctx("inductor")

        def boom_load(call):
            def run(*tensors):
                raise RuntimeError("eager boom")

            return run

        def boom_compile(obj, ctx):
            raise RuntimeError("compile boom")

        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=boom_compile,
            load_native_fn=boom_load,
            benchmark_fn=_bench_seq([1.0, 1.0]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.origin, "fallback")
        self.assertTrue(got.native_fallback)
        self.assertFalse(got.measured or got.autotuned)
        self.assertEqual(got.fallback_target, "aten.mm.default")
        self.assertEqual(got.fused_variants_deferred, 1)

    def test_fused_shaped_variant0_refuses_selection(self):
        # A None input binding means "computed by a prologue": the adapter
        # must not fake an unfused measurement; it records fallback and
        # defers fusion explicitly.
        op = _mock_op(bindings=[None, _binding("b")], fused_extra_variants=3)
        ctx = _mock_ctx("inductor")
        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.origin, "fallback")
        self.assertEqual(got.fused_variants_deferred, 3)
        self.assertEqual(got.fusion_deferred_reason, sel.FUSION_DEFERRED_REASON)

    def test_original_extern_baseline_only_falls_back(self):
        op = _mock_op(
            cands=[
                SimpleNamespace(
                    variant=0,
                    kind="ORIGINAL_EXTERN",
                    option_id="",
                    option_version=0,
                    config="",
                    cache_key="",
                    baseline=True,
                )
            ]
        )
        ctx = _mock_ctx("inductor")
        got = sel.select_op_unfused(
            op,
            ctx,
            op_pos=0,
            shared_nodes=(),
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([]),
            use_cache=False,
            device_str="cpu",
        )
        self.assertEqual(got.origin, "fallback")

    def test_bind_tensor_applies_base_plus_root_offset(self):
        base = torch.arange(20)
        root = torch.as_strided(base, (10,), (1,), storage_offset=5)
        got = sel.bind_tensor(root, _binding("r", offset=3), _layout((4,)))
        self.assertEqual(got.storage_offset(), 8)
        self.assertTrue(torch.equal(got, torch.tensor([8, 9, 10, 11])))
        spec = sel.spec_from_layout(_layout((2, 3), (6, 2)), 4, "cpu")
        self.assertEqual(
            (spec.dtype, spec.sizes, spec.strides, spec.offset),
            ("float32", (2, 3), (6, 2), 4),
        )

    def test_select_plan_rejects_overlapping_consumed_nodes(self):
        ctx = _mock_ctx("inductor")
        plan = SimpleNamespace(
            shared_nodes=[7], ops=[_mock_op(node=3), _mock_op(node=3)]
        )
        with self.assertRaises(ValueError):
            sel.select_plan(
                plan,
                ctx,
                codegen_fn=_mock_codegen,
                compile_fn=_add_compiler,
                load_native_fn=_add_loader,
                benchmark_fn=_bench_seq([1.0, 2.0] * 4),
                use_cache=False,
                device_str="cpu",
            )

    def test_select_plan_disjoint_freezes_shared_set(self):
        ctx = _mock_ctx("inductor")
        plan = SimpleNamespace(
            shared_nodes=[9], ops=[_mock_op(node=3), _mock_op(node=5)]
        )
        res = sel.select_plan(
            plan,
            ctx,
            codegen_fn=_mock_codegen,
            compile_fn=_add_compiler,
            load_native_fn=_add_loader,
            benchmark_fn=_bench_seq([1.0, 2.0] * 4),
            use_cache=False,
            device_str="cpu",
        )
        self.assertTrue(res.all_disjoint)
        self.assertEqual(res.shared_nodes, (9,))
        self.assertEqual([s.op_pos for s in res.selections], [0, 1])
        for s in res.selections:
            self.assertEqual(s.shared_nodes, (9,))
            self.assertTrue(s.shared_claims_disjoint)


def core():
    from torch._inductor.loops_ir import load_cpp_ir

    return load_cpp_ir()


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


def frontend_kernel_program(fn, args):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor.loops_ir import lower_fx_to_cpp

    gm = make_fx(fn)(*args)
    program = lower_fx_to_cpp(gm, args, executable=True)
    return gm, program


def rebuild_inductor(program):
    c = core()
    rebuilt = c.Program(program.name, "inductor")
    for b in program.inputs:
        rebuilt.add_input(b)
    for d in program.input_descriptors:
        rebuilt.add_input_descriptor(d)
    kinds = {
        "View": "add_view",
        "Compute": "add_compute",
        "ExternCall": "add_extern_call",
    }
    for node in program.nodes:
        getattr(rebuilt, kinds[type(node).__name__])(node)
    for d in program.outputs:
        rebuilt.add_output(d)
    return core().run_pipeline(rebuilt).kernel_program


@unittest.skipUnless(HAS_CUDA, "requires CUDA")
class TestOpSelectionGpuMeasured(TestCase):
    """Genuine vetted GPU timing (benchmark_fn=None) on tiny programs."""

    def test_tiny_gemm_native_vs_generated_measured(self):
        torch.manual_seed(0)
        dev = torch.device("cuda", torch.cuda.current_device())
        M, K, N = 32, 64, 32
        x = torch.randn(M, K, device=dev)
        w = torch.randn(K, N, device=dev)

        def fn(x, w):
            return torch.relu(x @ w)

        _, kp = frontend_kernel_program(fn, (x, w))
        program = rebuild_inductor(kp)
        decoded = oo.decode_op_signatures(program)
        ctx = oo.freeze_codegen_context(dev, "inductor")
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(program, decoded), decoded, ctx
        )
        self.assertEqual(len(plan.ops), 1)
        (op,) = plan.ops
        self.assertGreaterEqual(len(op.variants), 2)  # unfused + relu epilogue
        res = sel.select_plan(
            plan, ctx, benchmark_fn=None, use_cache=False, rtol=1e-4, atol=1e-4
        )
        (winner,) = res.selections
        self.assertTrue(winner.measured and winner.autotuned)
        self.assertEqual(winner.origin, "cold")
        self.assertEqual(winner.variant, 0)
        self.assertFalse(winner.cache_hit)
        self.assertGreaterEqual(winner.fused_variants_deferred, 1)
        self.assertEqual(winner.fusion_deferred_reason, sel.FUSION_DEFERRED_REASON)
        self.assertEqual(winner.fallback_target, "aten.mm.default")
        # Every unfused candidate (native baseline + full manifest) measured.
        unfused = [c for c in op.candidates if int(c.variant) == 0 and c.cache_key]
        self.assertGreaterEqual(len(unfused), 2)
        self.assertEqual(len(winner.measurements_ms), len(unfused))
        for _, ms in winner.measurements_ms:
            self.assertTrue(
                ms == ms and ms not in (float("inf"), float("-inf")) and ms >= 0
            )
        self.assertEqual(sel.SelectedOp.from_json(winner.to_json()), winner)
        self.assertTrue(res.all_disjoint)

    def test_tiny_reduction_native_vs_generated_measured(self):
        torch.manual_seed(1)
        dev = torch.device("cuda", torch.cuda.current_device())
        y = torch.randn(16, 256, device=dev)

        def fn(y):
            return aten.sum.dim_IntList(torch.relu(y), [-1], False)

        _, kp = frontend_kernel_program(fn, (y,))
        program = rebuild_inductor(kp)
        decoded = oo.decode_op_signatures(program)
        ctx = oo.freeze_codegen_context(dev, "inductor")
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(program, decoded), decoded, ctx
        )
        self.assertEqual(len(plan.ops), 1)
        (op,) = plan.ops
        self.assertGreaterEqual(len(op.variants), 2)  # relu prologue deferred
        res = sel.select_plan(
            plan, ctx, benchmark_fn=None, use_cache=False, rtol=1e-4, atol=1e-4
        )
        (winner,) = res.selections
        self.assertTrue(winner.measured and winner.autotuned)
        self.assertEqual(winner.origin, "cold")
        self.assertEqual(winner.fallback_target, "aten.sum.dim_IntList")
        self.assertGreaterEqual(winner.fused_variants_deferred, 1)
        self.assertGreaterEqual(len(winner.measurements_ms), 2)
        self.assertEqual(sel.SelectedOp.from_json(winner.to_json()), winner)

    def test_strict_tiny_gemm_is_native_singleton(self):
        torch.manual_seed(2)
        dev = torch.device("cuda", torch.cuda.current_device())
        x = torch.randn(16, 16, device=dev)
        w = torch.randn(16, 16, device=dev)

        def fn(x, w):
            return x @ w

        _, prog = frontend_kernel_program(fn, (x, w))  # strict_eager program
        kp = core().run_pipeline(prog).kernel_program
        decoded = oo.decode_op_signatures(kp)
        ctx = oo.freeze_codegen_context(dev, "strict_eager")
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(kp, decoded), decoded, ctx
        )
        (op,) = plan.ops
        self.assertEqual(len(op.candidates), 1)
        res = sel.select_plan(plan, ctx, benchmark_fn=None, use_cache=False)
        (winner,) = res.selections
        self.assertEqual(winner.origin, "singleton")
        self.assertFalse(winner.measured or winner.autotuned)
        self.assertTrue(winner.native_fallback)
        self.assertEqual(winner.option_id, "native_aten_gemm")

    def test_offset_input_binding_measured(self):
        # mm reading operand `a` through a view at a nonzero root offset:
        # variant-0 bindings carry root_offset and selection binds
        # root.storage_offset() + root_offset for every timed candidate.
        e = E()
        core()  # loader side effect; ProgramBuilder reloads below
        M, K, N, OFF = 16, 32, 16, 5
        b = ProgramBuilder("inductor")
        b.input("a", OFF + M * K)
        b.input("w", K * N)
        b.extern(
            "mm",
            "aten.mm.default",
            [],
            [("self", b.T("a")), ("mat2", b.T("w"))],
            [b.desc("arg0", "a", (M, K), [K, 1], OFF), b.desc("arg1", "w", (K, N))],
            (M, N),
        )
        b.output("mm", (M, N))
        decoded = oo.decode_op_signatures(b.p)
        (dop,) = [o for o in decoded.ops if o.status == e.DecodeStatus.DECODED]
        self.assertEqual(dop.input_bindings[0].root_offset, OFF)
        dev = torch.device("cuda", torch.cuda.current_device())
        ctx = oo.freeze_codegen_context(dev, "inductor")
        plan = oo.enumerate_op_candidates(
            oo.form_fused_op_regions(b.p, decoded), decoded, ctx
        )
        res = sel.select_plan(
            plan, ctx, benchmark_fn=None, use_cache=False, rtol=1e-4, atol=1e-4
        )
        (winner,) = res.selections
        self.assertTrue(winner.measured and winner.autotuned)
        self.assertEqual(winner.variant, 0)
        self.assertGreaterEqual(len(winner.measurements_ms), 2)


if __name__ == "__main__":
    run_tests()
