# Owner(s): ["module: inductor"]
"""Scoped native BatchNorm/RNG boundary tests.

CPU-runnable contract tests (import, effect ABI, FX rewrite shape, fake
kernels, autograd formulas, rejections) plus CUDA-gated graph-execution and
AOT gates (tiny native shapes, no performance benchmarking).
"""

import operator
import unittest

import torch
from torch import Tensor

from torch._inductor.fx_passes.loops_ir_native_boundaries import (
    BOUNDARY_MODULE_VERSION,
    BatchNormNativeBoundary,
    BoundaryEffect,
    BoundaryKind,
    BoundaryMutation,
    BoundaryReport,
    BoundarySpec,
    LoopsIrNativeBoundaryError,
    NativeBoundaryPreGradPass,
    SUPPORTED_BN_DTYPES,
    batch_norm_reserve_size,
    boundary_kind_of,
    canonical_target_name,
    is_native_boundary_target,
    rewrite_batch_norm_boundaries,
    rewrite_dropout_boundaries,
    rewrite_native_boundaries,
    select_batch_norm_backend,
)
from torch._inductor.test_case import run_tests, TestCase


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _byte_view(t: Tensor) -> Tensor:
    flat = t.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8)


def _assert_same_bytes(a: Tensor, b: Tensor, msg: str = "") -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{msg} shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{msg} dtype {a.dtype} vs {b.dtype}")
    ba, bb = _byte_view(a), _byte_view(b)
    if not bool((ba == bb).all()):
        raise AssertionError(f"{msg} byte mismatch")


class TestBoundaryEffectABI(TestCase):
    def test_module_version_pinned(self):
        self.assertEqual(BOUNDARY_MODULE_VERSION, "0.2.0")

    def test_canonical_names(self):
        self.assertEqual(
            canonical_target_name(torch.ops.aten.add.Tensor), "aten::add.Tensor"
        )
        self.assertEqual(canonical_target_name(torch.ops.aten.add), "aten::add")
        self.assertIsNone(canonical_target_name(torch.nn.functional.dropout))
        self.assertIsNone(canonical_target_name(object()))

    def test_known_symbols_include_default_overload(self):
        import torch._inductor.fx_passes.loops_ir_native_boundaries as B

        overloads = [
            torch.ops.loops_ir_native_bn.bn_cudnn_fwd.default,
            torch.ops.loops_ir_native_bn.bn_cudnn_bwd.default,
            torch.ops.loops_ir_native_dropout.dropout_fwd.default,
        ]
        for o in overloads:
            self.assertTrue(is_native_boundary_target(o), str(o))
        # Packets resolve too (via post-import accessors).
        self.assertTrue(is_native_boundary_target(B.native_bn_fwd()))
        self.assertTrue(is_native_boundary_target(B.native_bn_bwd()))
        self.assertTrue(is_native_boundary_target(B.native_dropout_fwd()))
        self.assertFalse(is_native_boundary_target(torch.ops.aten.add.Tensor))
        self.assertFalse(is_native_boundary_target("aten.native_batch_norm.default"))
        self.assertFalse(is_native_boundary_target(object()))

    def test_kind_and_effect_typing(self):
        self.assertIs(
            boundary_kind_of(torch.ops.loops_ir_native_bn.bn_cudnn_fwd.default),
            BoundaryKind.BN_FUNCTIONAL,
        )
        self.assertIs(
            boundary_kind_of(torch.ops.loops_ir_native_bn.bn_cudnn_bwd.default),
            BoundaryKind.BN_FUNCTIONAL,
        )
        self.assertIs(
            boundary_kind_of(torch.ops.loops_ir_native_dropout.dropout_fwd.default),
            BoundaryKind.RNG_STATE_UPDATE,
        )
        self.assertIsNone(boundary_kind_of(torch.ops.aten.add.Tensor))
        with self.assertRaises(Exception):
            BoundaryEffect("rng_magic")  # type: ignore[call-arg]

    def test_spec_stored_typed_not_dict(self):
        spec = BoundarySpec(
            kind=BoundaryKind.BN_FUNCTIONAL,
            symbol="loops_ir_native_bn::bn_cudnn_fwd",
            version="0.2.0:bn2",
            expected_impl=1,
            effect=BoundaryEffect.BN_RUNNING_STATS_FUNCTIONAL,
        )
        self.assertIsInstance(spec, BoundarySpec)
        self.assertEqual(spec.expected_impl, 1)

    def test_registered_op_tags(self):
        fwd = torch.ops.loops_ir_native_dropout.dropout_fwd
        self.assertIn(torch.Tag.nondeterministic_seeded, fwd.default.tags)
        bn = torch.ops.loops_ir_native_bn.bn_cudnn_fwd
        self.assertNotIn(torch.Tag.nondeterministic_seeded, bn.default.tags)

    def test_pass_uuid_is_stable_method(self):
        import inspect

        self.assertTrue(inspect.isfunction(NativeBoundaryPreGradPass.uuid))
        a = NativeBoundaryPreGradPass(numerics="strict_eager")
        b = NativeBoundaryPreGradPass(numerics="strict_eager")
        self.assertEqual(a.uuid(), b.uuid())
        self.assertIn(BOUNDARY_MODULE_VERSION, a.uuid())
        self.assertNotEqual(
            a.uuid(), NativeBoundaryPreGradPass(numerics="inductor").uuid()
        )
        with self.assertRaises(LoopsIrNativeBoundaryError):
            NativeBoundaryPreGradPass(numerics="bogus")

    def test_supported_bn_scope_declared(self):
        self.assertIn(torch.float32, SUPPORTED_BN_DTYPES)
        self.assertNotIn(torch.float16, SUPPORTED_BN_DTYPES)

    def test_typed_reports(self):
        import torch.fx as fx

        def f(x):
            return torch.nn.functional.dropout(x, p=0.5, training=True)

        gm = fx.symbolic_trace(f)
        bn_rep, do_rep = rewrite_native_boundaries(gm, numerics="strict_eager")
        self.assertIsInstance(bn_rep, BoundaryReport)
        self.assertIsInstance(do_rep, BoundaryReport)
        self.assertIsInstance(do_rep.rejections, tuple)
        self.assertEqual(do_rep.rewrites, 1)


class TestReserveMeta(TestCase):
    def test_reserve_query_available_and_zero_on_supported(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        for shape in ((4, 8, 4, 4), (4, 8, 32, 32)):
            x = torch.randn(*shape, device="cuda")
            self.assertEqual(batch_norm_reserve_size(x, True), 0)
            self.assertEqual(batch_norm_reserve_size(x, False), 0)

    def test_backend_selection_matches_eager_cudnn(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        x = torch.randn(4, 8, 4, 4, device="cuda")
        w = torch.randn(8, device="cuda")
        b = torch.randn(8, device="cuda")
        rm = torch.zeros(8, device="cuda")
        rv = torch.ones(8, device="cuda")
        backend = select_batch_norm_backend(x, w, b, rm, rv, True, 1e-5)
        name = getattr(backend, "name", str(backend).rsplit(".", 1)[-1])
        self.assertEqual(name, "Cudnn")


def _dynamo_shape_bn_gm():
    """Dynamo-shape GM: (x, rm, rv, w, b, training, momentum, eps) + nbt add_."""
    import torch.fx as fx

    gm = fx.GraphModule({}, fx.Graph())
    g = gm.graph
    ph_rm = g.placeholder("rm")
    ph_rv = g.placeholder("rv")
    ph_w = g.placeholder("w")
    ph_b = g.placeholder("b")
    ph_nbt = g.placeholder("nbt")
    ph_x = g.placeholder("x")
    bn = g.call_function(
        torch.nn.functional.batch_norm,
        (ph_x, ph_rm, ph_rv, ph_w, ph_b, True, 0.1, 1e-5),
    )
    g.call_method("add_", (ph_nbt, 1))
    s = g.call_method("sum", (bn,))
    g.output((s,))
    return gm, bn


class TestFXRewrites(TestCase):
    def _bn_gm_cpu(self):
        import torch.fx as fx

        def f(x, w, b, rm, rv):
            return torch.ops.aten.batch_norm(x, w, b, rm, rv, True, 0.1, 1e-5, True)

        gm = fx.symbolic_trace(f)
        for n in gm.graph.nodes:
            n.meta["example_value"] = None
        for n in gm.graph.nodes:
            if n.op == "call_function" and "batch_norm" in str(n.target):
                n.meta["example_value"] = torch.randn(4, 8, 4, 4)
        return gm

    def test_bn_rewrite_rejects_without_cuda_fake(self):
        gm = self._bn_gm_cpu()
        report = rewrite_batch_norm_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 0)
        self.assertTrue(report.rejections)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(any("batch_norm" in t for t in targets))

    def test_dropout_rewrite_p_trains_only(self):
        import torch.fx as fx

        def f(x):
            return torch.nn.functional.dropout(x, p=0.5, training=True) + 0.0

        gm = fx.symbolic_trace(f)
        report = rewrite_dropout_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 1)
        texts = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(any("dropout_fwd" in t for t in texts))

    def test_dropout_p0_and_eval_left_alone(self):
        import torch.fx as fx

        def f_p0(x):
            return torch.nn.functional.dropout(x, p=0.0, training=True)

        def f_eval(x):
            return torch.nn.functional.dropout(x, p=0.5, training=False)

        for f in (f_p0, f_eval):
            gm = fx.symbolic_trace(f)
            report = rewrite_dropout_boundaries(gm, numerics="strict_eager")
            self.assertEqual(report.rewrites, 0)

    def test_dropout_inplace_rejects(self):
        import torch.fx as fx

        def f(x):
            return torch.nn.functional.dropout(x, p=0.5, training=True, inplace=True)

        gm = fx.symbolic_trace(f)
        report = rewrite_dropout_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 0)
        self.assertTrue(report.rejections)

    def test_native_dropout_multioutput_contract(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x):
            return torch.ops.aten.native_dropout(x, 0.5, True)

        gm = make_fx(f)(torch.randn(2, 8))
        report = rewrite_dropout_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 1)
        getitems = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == operator.getitem
        ]
        idxs = sorted(n.args[1] for n in getitems)
        self.assertEqual(idxs, [0, 1])

    def test_bn_emits_copy_glue_and_typed_meta(self):
        gm, _ = _dynamo_shape_bn_gm()
        x0 = torch.randn(4, 8, 4, 4)
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["example_value"] = x0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["example_value"] = x0
        report = rewrite_batch_norm_boundaries(gm, numerics="strict_eager")
        # CPU metadata: must reject (fp32 CUDA only); nothing half-rewritten.
        self.assertEqual(report.rewrites, 0)
        self.assertTrue(report.rejections)


class TestDropoutBoundarySemantics(TestCase):
    def test_fake_shapes_and_mask_dtype(self):
        x = torch.randn(2, 8)
        out, mask = torch.ops.loops_ir_native_dropout.dropout_fwd(x, 0.5, True)
        self.assertEqual(tuple(out.shape), (2, 8))
        self.assertEqual(tuple(mask.shape), (2, 8))
        self.assertEqual(mask.dtype, torch.bool)

    def test_p1_matches_eager_signed_zeros(self):
        x = torch.randn(2, 8)
        out, mask = torch.ops.loops_ir_native_dropout.dropout_fwd(x, 1.0, True)
        eager_out, eager_mask = torch.ops.aten.native_dropout(x, 1.0, True)
        self.assertTrue(bool((~mask).all()))
        self.assertTrue(bool((~eager_mask).all()))
        _assert_same_bytes(out, eager_out, "p=1")

    def test_eval_and_p0_reject_loudly(self):
        x = torch.randn(2, 8)
        with self.assertRaises(Exception):
            torch.ops.loops_ir_native_dropout.dropout_fwd(x, 0.5, False)
        with self.assertRaises(Exception):
            torch.ops.loops_ir_native_dropout.dropout_fwd(x, 0.0, True)

    def test_backward_scale_matches_eager_derivative(self):
        torch.manual_seed(0)
        x = torch.randn(2, 8, requires_grad=True)
        out, mask = torch.ops.loops_ir_native_dropout.dropout_fwd(x, 0.5, True)
        out.sum().backward()
        expected = mask.to(torch.float32) * 2.0
        _assert_same_bytes(x.grad, expected, "dropout grad")

    def test_rng_order_single_call_matches_eager_cuda(self):
        _require_cuda()
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        x = torch.randn(2, 8, device="cuda")
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        e, _ = torch.ops.aten.native_dropout(x, 0.5, True)
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        b, _ = torch.ops.loops_ir_native_dropout.dropout_fwd(x, 0.5, True)
        _assert_same_bytes(b, e, "single-call RNG")


class TestBatchNormGraphExecution(TestCase):
    """Actual graph execution (not symbol checks): distinct w/b/rm/rv."""

    def _inputs(self, shape=(4, 8, 4, 4), dtype=torch.float32):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        w = torch.randn(shape[1], device="cuda", dtype=dtype) * 2 + 1
        b = torch.randn(shape[1], device="cuda", dtype=dtype) * 3 - 1
        rm = torch.randn(shape[1], device="cuda", dtype=dtype)
        rv = torch.rand(shape[1], device="cuda", dtype=dtype) + 0.5
        nbt = torch.tensor(5)
        return x, w, b, rm, rv, nbt

    def _rewritten_gm(self, example):
        gm, bn_node = _dynamo_shape_bn_gm()
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["example_value"] = example[node.target]
        bn_node.meta["example_value"] = torch.randn(
            4, 8, 4, 4, device="cuda", dtype=torch.float32
        )
        bn_rep, do_rep = rewrite_native_boundaries(gm, numerics="strict_eager")
        self.assertEqual(bn_rep.rewrites, 1, bn_rep.rejections)
        gm.graph.lint()
        return gm

    def test_graph_exec_bytes_with_distinct_stats(self):
        _require_cuda()
        x, w, b, rm, rv, nbt = self._inputs()
        gm = self._rewritten_gm(
            {"x": x, "w": w, "b": b, "rm": rm, "rv": rv, "nbt": nbt}
        )
        xe, we, be = x.clone(), w.clone(), b.clone()
        rme, rve, nbte = rm.clone(), rv.clone(), nbt.clone()
        nbte.add_(1)
        oe = torch.nn.functional.batch_norm(xe, rme, rve, we, be, True, 0.1, 1e-5).sum()
        xg, wg, bg = x.clone(), w.clone(), b.clone()
        rmg, rvg, nbtg = rm.clone(), rv.clone(), nbt.clone()
        og = gm(rmg, rvg, wg, bg, nbtg, xg)[0]
        _assert_same_bytes(oe, og, "out")
        _assert_same_bytes(rme, rmg, "rm")
        _assert_same_bytes(rve, rvg, "rv")
        self.assertEqual(nbte.item(), nbtg.item())
        # copy_ glue present as real users with typed mutation record.
        copies = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and canonical_target_name(n.target) == "aten::copy_.default"
        ]
        self.assertEqual(len(copies), 2)
        boundary_calls = [
            n for n in gm.graph.nodes if is_native_boundary_target(n.target)
        ]
        self.assertEqual(len(boundary_calls), 1)
        mut = boundary_calls[0].meta["loops_ir_mutation"]
        self.assertIsInstance(mut, BoundaryMutation)

    def test_dce_keeps_copy_users(self):
        _require_cuda()
        x, w, b, rm, rv, nbt = self._inputs()
        gm = self._rewritten_gm(
            {"x": x, "w": w, "b": b, "rm": rm, "rv": rv, "nbt": nbt}
        )
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        copies = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and canonical_target_name(n.target) == "aten::copy_.default"
        ]
        self.assertEqual(len(copies), 2)
        # Still executes byte-equal after DCE.
        xg, wg, bg = x.clone(), w.clone(), b.clone()
        rmg, rvg, nbtg = rm.clone(), rv.clone(), nbt.clone()
        og = gm(rmg, rvg, wg, bg, nbtg, xg)[0]
        xe, we, be = x.clone(), w.clone(), b.clone()
        rme, rve = rm.clone(), rv.clone()
        oe = torch.nn.functional.batch_norm(xe, rme, rve, we, be, True, 0.1, 1e-5).sum()
        _assert_same_bytes(oe, og, "out-post-dce")
        _assert_same_bytes(rme, rmg, "rm-post-dce")

    def test_op_exec_fwd_bwd_bytes_and_state(self):
        _require_cuda()
        x0, w0, b0, rm0, rv0, _ = self._inputs()

        def run_eager(x, w, b, rm, rv):
            xc = x.clone().requires_grad_(True)
            wc = w.clone().requires_grad_(True)
            bc = b.clone().requires_grad_(True)
            rmc, rvc = rm.clone(), rv.clone()
            o = torch.nn.functional.batch_norm(xc, rmc, rvc, wc, bc, True, 0.1, 1e-5)
            o.sum().backward()
            return o.detach(), xc.grad, wc.grad, bc.grad, rmc, rvc

        def run_boundary(x, w, b, rm, rv):
            xc = x.clone().requires_grad_(True)
            wc = w.clone().requires_grad_(True)
            bc = b.clone().requires_grad_(True)
            rmc, rvc = rm.clone(), rv.clone()
            o, _, _, _, nrm, nrv = torch.ops.loops_ir_native_bn.bn_cudnn_fwd(
                xc, wc, bc, rmc, rvc, True, 0.1, 1e-5, True, 1
            )
            o.sum().backward()
            return o.detach(), xc.grad, wc.grad, bc.grad, nrm, nrv

        a = run_eager(x0, w0, b0, rm0, rv0)
        bb = run_boundary(x0, w0, b0, rm0, rv0)
        for name, i in (
            ("fwd", 0),
            ("gx", 1),
            ("gw", 2),
            ("gb", 3),
            ("rm", 4),
            ("rv", 5),
        ):
            _assert_same_bytes(a[i], bb[i], name)

    def test_impl_mismatch_rejects(self):
        _require_cuda()
        x0, w0, b0, rm0, rv0, _ = self._inputs()
        with self.assertRaises(Exception):
            torch.ops.loops_ir_native_bn.bn_cudnn_fwd(
                x0, w0, b0, rm0, rv0, True, 0.1, 1e-5, True, 999
            )

    def test_multistep_bytes_channels_last_and_conv_shape(self):
        _require_cuda()
        for shape, fmt in (
            ((4, 8, 4, 4), None),
            ((8, 16, 8, 8), None),
            ((4, 8, 4, 4), torch.channels_last),
        ):
            with self.subTest(shape=shape, format=fmt):
                kw = {"device": "cuda", "dtype": torch.float32}
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                w0 = torch.randn(shape[1], **kw)
                b0 = torch.randn(shape[1], **kw)
                rm0 = torch.zeros(shape[1], **kw)
                rv0 = torch.ones(shape[1], **kw)
                per_name: dict[str, tuple] = {}
                for name in ("eager", "boundary"):
                    w = w0.clone().requires_grad_(True)
                    b = b0.clone().requires_grad_(True)
                    rm, rv = rm0.clone(), rv0.clone()
                    opt = torch.optim.SGD([w, b], lr=0.01, momentum=0.9)
                    for s in range(3):
                        torch.manual_seed(100 + s)
                        torch.cuda.manual_seed_all(100 + s)
                        x = torch.randn(*shape, **kw)
                        if fmt is not None:
                            x = x.to(memory_format=fmt)
                        x.requires_grad_(True)
                        if name == "eager":
                            o = torch.nn.functional.batch_norm(
                                x, rm, rv, w, b, True, 0.1, 1e-5
                            )
                        else:
                            o, _, _, _, nrm, nrv = (
                                torch.ops.loops_ir_native_bn.bn_cudnn_fwd(
                                    x, w, b, rm, rv, True, 0.1, 1e-5, True, 1
                                )
                            )
                            rm, rv = nrm.detach(), nrv.detach()
                        (o.sum() + (w**2).sum()).backward()
                        opt.step()
                        opt.zero_grad()
                    per_name[name] = (
                        w.detach(),
                        b.detach(),
                        rm,
                        rv,
                        opt.state_dict(),
                    )
                for key, i in (("w", 0), ("b", 1), ("rm", 2), ("rv", 3)):
                    _assert_same_bytes(
                        per_name["eager"][i], per_name["boundary"][i], key
                    )
                se = per_name["eager"][4]["state"]
                sb = per_name["boundary"][4]["state"]
                self.assertEqual(set(se), set(sb))
                for pid in se:
                    for kk in se[pid]:
                        _assert_same_bytes(se[pid][kk], sb[pid][kk], f"opt.{kk}")

    def test_full_aot_nn_batchnorm_two_steps(self):
        _require_cuda()
        import torch.nn as nn
        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_module

        gm, bn_node = _dynamo_shape_bn_gm()
        example = {
            "rm": torch.zeros(8, device="cuda"),
            "rv": torch.ones(8, device="cuda"),
            "w": torch.randn(8, device="cuda"),
            "b": torch.randn(8, device="cuda"),
            "nbt": torch.tensor(6),
            "x": torch.randn(4, 8, 4, 4, device="cuda"),
        }
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["example_value"] = example[node.target]
        bn_node.meta["example_value"] = torch.randn(4, 8, 4, 4, device="cuda")
        bn_rep, _ = rewrite_native_boundaries(gm, numerics="strict_eager")
        self.assertEqual(bn_rep.rewrites, 1)

        class W(nn.Module):
            def __init__(self, gm_):
                super().__init__()
                self.gm = gm_
                self.register_buffer("rm", example["rm"].clone())
                self.register_buffer("rv", example["rv"].clone())
                self.register_buffer("nbt", example["nbt"].clone())
                self.w = nn.Parameter(example["w"].clone())
                self.b = nn.Parameter(example["b"].clone())

            def forward(self, x):
                return self.gm(self.rm, self.rv, self.w, self.b, self.nbt, x)

        fw_targets: list[str] = []
        bw_targets: list[str] = []

        def pf(gm_, ins):
            for n in gm_.graph.nodes:
                fw_targets.append(canonical_target_name(n.target) or str(n.target))
            return make_boxed_func(gm_.forward)

        def pb(gm_, ins):
            for n in gm_.graph.nodes:
                bw_targets.append(canonical_target_name(n.target) or str(n.target))
            return make_boxed_func(gm_.forward)

        eager_mod = nn.BatchNorm2d(8).cuda().train()
        eager_mod.weight.data.copy_(example["w"])
        eager_mod.bias.data.copy_(example["b"])
        eager_mod.running_mean.data.copy_(example["rm"])
        eager_mod.running_var.data.copy_(example["rv"])
        eager_mod.num_batches_tracked.data.copy_(example["nbt"])
        compiled_mod = W(gm).cuda().train()
        opt_e = torch.optim.SGD(
            [eager_mod.weight, eager_mod.bias], lr=0.01, momentum=0.9
        )
        opt_c = torch.optim.SGD([compiled_mod.w, compiled_mod.b], lr=0.01, momentum=0.9)
        # One compiled artifact, stepped twice: dynamo/aot re-traces per call
        # under the hood, but the *returned* compiled function is reusable.
        # (Re-calling aot_module per step rebuilds the joint closure over the
        # same freed saved tensors -- that was the "second backward" error.)
        f = aot_module(compiled_mod, fw_compiler=pf, bw_compiler=pb)
        for step in range(2):
            torch.manual_seed(100 + step)
            torch.cuda.manual_seed_all(100 + step)
            xe = torch.randn(4, 8, 4, 4, device="cuda")
            torch.manual_seed(100 + step)
            torch.cuda.manual_seed_all(100 + step)
            xc = torch.randn(4, 8, 4, 4, device="cuda")
            loss_e = eager_mod(xe).sum()
            loss_e.backward()
            opt_e.step()
            opt_e.zero_grad()
            loss_c = f(xc)
            (loss_c[0] if isinstance(loss_c, (list, tuple)) else loss_c).backward()
            opt_c.step()
            opt_c.zero_grad()
        fw_text = "\n".join(fw_targets)
        bw_text = "\n".join(bw_targets)
        self.assertIn("loops_ir_native_bn::bn_cudnn_fwd", fw_text)
        self.assertIn("loops_ir_native_bn::bn_cudnn_bwd", bw_text)
        for banned in ("_native_batch_norm_legit", "var_mean", "rsqrt"):
            self.assertNotIn(banned, bw_text)
        _assert_same_bytes(eager_mod.weight, compiled_mod.w, "w-2step")
        _assert_same_bytes(eager_mod.bias, compiled_mod.b, "b-2step")
        _assert_same_bytes(eager_mod.running_mean, compiled_mod.rm, "rm-2step")
        _assert_same_bytes(eager_mod.running_var, compiled_mod.rv, "rv-2step")
        self.assertEqual(eager_mod.num_batches_tracked.item(), compiled_mod.nbt.item())


class TestNegativeScope(TestCase):
    def test_eval_bn_rewrite_rejects(self):
        import torch.fx as fx

        def f(x):
            return torch.nn.functional.batch_norm(
                x,
                torch.zeros(8),
                torch.ones(8),
                torch.randn(8),
                torch.randn(8),
                False,
                0.1,
                1e-5,
            )

        gm = fx.symbolic_trace(f)
        adapter = BatchNormNativeBoundary(numerics="strict_eager")
        for node in list(gm.graph.nodes):
            if adapter.applies(node):
                self.assertFalse(adapter.rewrite(gm, node))
        self.assertTrue(adapter.rejections)

    def test_cudnn_disabled_rejects(self):
        import torch.fx as fx

        def f(x, w, b, rm, rv):
            return torch.ops.aten.batch_norm(x, w, b, rm, rv, True, 0.1, 1e-5, False)

        gm = fx.symbolic_trace(f)
        report = rewrite_batch_norm_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 0)
        self.assertTrue(
            any("cudnn_enabled" in r for r in report.rejections), report.rejections
        )

    def test_fp16_bn_rewrite_rejects(self):
        _require_cuda()
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x, w, b, rm, rv):
            return torch.ops.aten.batch_norm(x, w, b, rm, rv, True, 0.1, 1e-5, True)

        x = torch.randn(4, 8, 4, 4, device="cuda", dtype=torch.float16)
        w = torch.randn(8, device="cuda", dtype=torch.float16)
        b = torch.randn(8, device="cuda", dtype=torch.float16)
        rm = torch.zeros(8, device="cuda", dtype=torch.float16)
        rv = torch.ones(8, device="cuda", dtype=torch.float16)
        gm = make_fx(f)(x, w, b, rm, rv)
        for n in gm.graph.nodes:
            if n.op == "call_function" and "batch_norm" in str(n.target):
                n.meta["example_value"] = x
        report = rewrite_batch_norm_boundaries(gm, numerics="strict_eager")
        self.assertEqual(report.rewrites, 0)

    def test_backward_symbol_rejects_unknown_impl(self):
        x = torch.randn(2, 4)
        g = torch.randn(2, 4)
        w = torch.randn(4)
        sm = torch.randn(4)
        sv = torch.rand(4) + 0.5
        res = torch.empty(0, dtype=torch.uint8)
        with self.assertRaises(Exception):
            torch.ops.loops_ir_native_bn.bn_cudnn_bwd(
                x, g, w, sm, sv, res, 0, True, 1e-5
            )


if __name__ == "__main__":
    run_tests()
