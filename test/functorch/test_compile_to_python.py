# Owner(s): ["oncall: pt2"]
import ast
import contextlib
import copy
import importlib
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

import torch
import torch._functorch._aot_autograd.to_standalone_python as to_standalone_python
import torch._functorch.config as functorch_config
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.codegen import GeneratedSource
from torch._functorch._aot_autograd.to_standalone_python import (
    _capture_autograd_specs,
    _compose_standalone_module,
    _find_effectful_op,
    _known_helper_table,
    _module_level_names,
    _select_training_spec,
)
from torch._functorch.aot_autograd import compile_to_python, load_from_python
from torch._higher_order_ops.effects import _get_effect, hop_print
from torch._inductor.utils import fresh_cache
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


# ``torch._inductor.standalone_compile`` the attribute is a function that shadows
# the submodule of the same name, so bind the module explicitly.
standalone_compile = importlib.import_module("torch._inductor.standalone_compile")


def _capture(m, x, tracing_mode="real"):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), the same shape ``torch.compiler.precompile`` feeds the AOT lowering. The
    flat-input ordering returned by ``_flat_inputs`` MUST match this order."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(flat_fn, tracing_mode=tracing_mode)(pb + [x])


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


# A stand-in for the authoritative inner-call placeholder that compile_to_python threads
# into _compose_standalone_module. The guard tests below build wrappers by hand and never
# reference the real inner call, so any distinct object serves as its identity.
_SENTINEL_INNER_CALL = object()


def _make_holder(value):
    # A plain importable module-level callable used by test_unwired_chain_wrapper_rejected
    # as a stand-in inner-ref global, so it resolves cleanly as source (isolating the
    # "unwired wrapper" rejection from a resolution failure).
    return value


class _NewObjEx:
    # A baked-global fixture whose reduce (__getnewargs_ex__ + dict state) emits a
    # ``_rebuild(...)`` call, used by test_rebuild_helper_spliced_and_runs_in_composed_module
    # to check the composed module splices and runs the _rebuild helper.
    def __new__(cls, a, b):
        obj = object.__new__(cls)
        obj.a = a
        obj.b = b
        return obj

    def __getnewargs_ex__(self):
        return ((self.a,), {"b": self.b})

    def __eq__(self, other):
        return isinstance(other, _NewObjEx) and self.a == other.a and self.b == other.b


class _Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2.0 + 1.0)


class _ViewAlias(torch.nn.Module):
    def forward(self, x):
        return x.view(-1)


class _SumDim1(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class _MultiOut(torch.nn.Module):
    def forward(self, x):
        return x * 2.0, x.sum(dim=1), torch.relu(x)


class _BufferMutate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x):
        self.b.add_(x.sum())
        return x + self.b


def _two_views_of_intermediate(x, w):
    h = torch.tanh(x @ w)
    return h[:2], h[2:]


def _view_of_input_and_dense(x, w):
    return x.view(-1), torch.tanh(x @ w)


class _MatMul(torch.nn.Module):
    # addmm is an autocast-to-bf16 op, so a float32 input under torch.autocast engages
    # autocast and bakes the casts into the graph -- the setup the _DisableAutocast_ test
    # needs (a Linear's addmm behaves the same way).
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.l(x)


def _compose(m, x):
    gm = _capture(m, x)
    return compile_to_python(gm, _flat_inputs(m, x))


def _assert_composed(test, src):
    # Structural markers proving this is the COMPOSED module (not just the inner inductor
    # output): the outer entry takes flat_inputs, the inner call is captured as
    # _inner_call, and the AOTAutograd orchestration is inlined as a real def that the
    # outer call invokes directly by name. (All wrappers are inlined now -- no _exec_wrapper
    # / source-string blobs anywhere; the chain-wrapper test checks that directly.)
    test.assertIn("def call(flat_inputs):", src)
    test.assertIn("_inner_call = call", src)
    test.assertIn("def _runtime_wrapper(", src)
    test.assertIn("return _runtime_wrapper(", src)
    # Auditability guarantee: no pickle.loads / base64 blob in the emitted module.
    # _load_from_bytes is the storage-reduce callable that embeds raw weight bytes and
    # base64 is the only other opaque-blob encoding that could smuggle them in, so the
    # absence of all three is what makes the comment's no-blob promise actually hold.
    test.assertNotIn("pickle.loads", src)
    test.assertNotIn("_load_from_bytes", src)
    test.assertNotIn("base64", src)


# Effectful targets the composition must reject. Both kinds must be caught: an effectful
# OpOverload (aten._print) and an effectful HigherOrderOperator (hop_print, ORDERED). A HOP
# is NOT an OpOverload, so an OpOverload-only gate would silently let the HOP through -- so
# the HOP subtest is the one that fails against the narrower gate.
_EFFECTFUL_TARGETS = [
    subtest(torch.ops.aten._print.default, name="op_overload"),
    subtest(hop_print, name="hop"),
]


@instantiate_parametrized_tests
class TestAOTCompileToPython(TestCase):
    # End-to-end coverage of the functorch composition layer: compile_to_python composes
    # AOTAutograd's codegen'd runtime wrappers (prelude/epilogue) around the inner Inductor
    # call into one standalone module, and the emitted module must match eager. All CPU.

    def test_pointwise_runs_like_eager(self):
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, cache = _compose(m, x)
        _assert_composed(self, src)
        # Return contract: cache is the opaque acceleration bytes or None.
        self.assertIsInstance(cache, (bytes, type(None)))
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_load_from_python_standalone_and_warm(self):
        # load_from_python is the inverse of compile_to_python: python_code runs standalone
        # (kernels JIT on first use), and the forwarded inductor cache is a pure accelerator
        # (warms the kernel caches so exec loads precompiled binaries). Both paths must match
        # eager; run each in a fresh cache dir so the standalone path is a genuine cold load.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, cache = _compose(m, x)
        # Standalone: no cache, the module JIT-compiles its own kernels.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src)(_flat_inputs(m, x))[0], m(x))
        # Warm: the forwarded cache accelerates the same module; result is identical. cache
        # is None on cache-disabled shards, where only the standalone path applies.
        if cache is not None:
            with fresh_cache(), torch.no_grad():
                self.assertEqual(
                    load_from_python(src, cache)(_flat_inputs(m, x))[0], m(x)
                )

    @skipIfTorchDynamo(
        "make_fx of a training closure cannot trace under an outer dynamo"
    )
    def test_inline_backward_graph_is_not_lowered_as_inference(self):
        # A graph that differentiates INLINE (make_fx tracing through a
        # .backward()) has no joint, so a joint-only check would call it
        # inference. Inductor's decide_layout_opt branches on that and can
        # convert a conv to channels-last, which makes cuDNN serve a TF32 NHWC
        # kernel -- a silent ~2e-4 relative difference in the gradients. The ops
        # decide, not the graph count.
        from torch._functorch._aot_autograd.to_standalone_python import (
            _graph_differentiates,
        )

        conv = torch.nn.Conv2d(4, 8, 3, padding=1)
        x = torch.randn(2, 4, 8, 8)

        self.assertFalse(_graph_differentiates(_capture(conv, x)))

        def train_step(*args):
            names = [n for n, _ in conv.named_parameters()]
            params = dict(zip(names, args[: len(names)]))
            with stateless._reparametrize_module(conv, params):
                conv(args[-1]).sum().backward()

        flat = [t.detach().requires_grad_(True) for t in _flat_inputs(conv, x)]
        with torch.enable_grad():
            traced = make_fx(train_step)(*flat)
        self.assertTrue(_graph_differentiates(traced))
        self.assertTrue(
            any("convolution_backward" in str(n.target) for n in traced.graph.nodes)
        )

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_graph_composes_forward_and_backward(self):
        # grad_enabled with inputs that require grad makes AOTAutograd emit a
        # JOINT forward+backward: two dense graphs, bridged by an autograd
        # Function the composer emits (its forward/backward bodies are
        # AOTAutograd's own codegen'd source). The served output must therefore
        # carry grad_fn and its .backward() must run the compiled backward.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4, requires_grad=True)
        eager_out = m(x)
        eager_out.sum().backward()
        expected = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
        expected_x_grad = x.grad.detach().clone()

        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        # Both inductor modules and the backward wrappers are inlined as source.
        self.assertIn("_inner_call_fw", src)
        self.assertIn("_inner_call_bw", src)
        self.assertIn("def _backward_prologue(", src)
        self.assertIn("class _CompiledFunction(torch.autograd.Function):", src)
        self.assertNotIn("pickle.loads", src)

        out = _exec(src)(_flat_inputs(m, x))
        out = out[0] if isinstance(out, (list, tuple)) else out
        self.assertIsNotNone(out.grad_fn)
        self.assertEqual(out, eager_out)
        for p in m.parameters():
            p.grad = None
        x.grad = None
        out.sum().backward()
        for name, param in m.named_parameters():
            self.assertEqual(param.grad, expected[name])
        self.assertEqual(x.grad, expected_x_grad)

    def test_training_rebuild_helper_spliced_and_runs(self):
        # A baked global that reconstructs through the pickle-reduce-as-source
        # path emits ``_rebuild(...)``, and the training composer must splice the
        # helper like the inference one does. No real metadata reduces this way,
        # so plant one on a captured wrapper right before the composer sees it.
        orig = to_standalone_python._compose_training_module

        def plant(fw, bw, captured, spec, *placeholders):
            gen = next(g for g in captured if g.artifact_name == "backward_prologue")
            gen.globals_dict["_baked"] = _NewObjEx(1, b=2)
            return orig(fw, bw, captured, spec, *placeholders)

        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with (
            torch.enable_grad(),
            mock.patch.object(to_standalone_python, "_compose_training_module", plant),
        ):
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        self.assertIn("def _rebuild", src)
        self.assertIn("_baked = _rebuild(", src)
        ns = {"__name__": "_compiled"}
        exec(compile(src, "<compiled>", "exec"), ns)
        self.assertEqual(ns["_baked"], _NewObjEx(1, b=2))
        self.assertEqual(ns["call"](_flat_inputs(m, x))[0], m(x))

    def test_training_compose_refuses_fakified_forward(self):
        # fakify_first_call makes AOTAutograd wrap the compiled forward in a
        # FakifiedOutWrapper. The emitted _CompiledFunction wires compiled_fw
        # straight to the inner call, so the composer must refuse rather than
        # silently drop the wrapper (keeping the caller's pickled-bundle fallback).
        orig = standalone_compile._standalone_context

        @contextlib.contextmanager
        def fakify(*args, **kwargs):
            with orig(*args, **kwargs):
                torch._guards.TracingContext.get().fakify_first_call = True
                yield

        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with (
            torch.enable_grad(),
            mock.patch.object(standalone_compile, "_standalone_context", fakify),
            self.assertRaisesRegex(NotImplementedError, "FakifiedOutWrapper"),
        ):
            compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)

    def test_training_backward_under_compiled_autograd_raises_not_implemented(self):
        # Compiled autograd reads _lazy_backward_info off the forward class before
        # anything else and, on None, blames AOTAutogradCache -- advice that cannot
        # apply to a source artifact. The emitted class refuses with its own reason.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        out = _exec(src)(_flat_inputs(m, x))
        out = out[0] if isinstance(out, (list, tuple)) else out
        with (
            torch._dynamo.compiled_autograd._enable(lambda gm: gm),
            self.assertRaisesRegex(
                NotImplementedError,
                "compiled autograd is not supported for a standalone training artifact",
            ),
        ):
            out.sum().backward()

    def test_training_forward_and_backward_do_not_share_names(self):
        # The two inductor modules are spliced into ONE namespace and both define
        # call / Runner / their kernels. A module resolves those as late-bound
        # globals when INVOKED, so without per-module renaming the forward runs
        # the backward's kernels -- which surfaces as an arity error, or worse.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        tree = ast.parse(src)
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(len(names), len(set(names)), f"duplicate top-level: {names}")

    def test_training_compose_refuses_unmodeled_wrappers(self):
        # Aliased + mutated inputs drive AOTSyntheticBaseWrapper, which the
        # training compose does not splice. It must REFUSE: silently dropping
        # the wrapper composes a module whose inner forward was compiled for
        # the merged synthetic-base calling convention, so it only fails (or
        # miscomputes) at serve time -- replacing the caller's working
        # pickled-bundle fallback with a broken artifact.
        def f(a, b, w):
            a.mul_(2)
            return ((a + b) * w).sum()

        def make():
            base = torch.arange(4, dtype=torch.float32) + 1
            return base[:], base  # a aliases b

        w = torch.randn(4, requires_grad=True)
        with torch.enable_grad():
            gm = make_fx(f)(*make(), w)
        a0, b0 = make()
        with (
            torch.enable_grad(),
            self.assertRaisesRegex(NotImplementedError, "cannot yet model"),
        ):
            compile_to_python(gm, [a0, b0, w], grad_enabled=True)

    def test_training_compose_refuses_inlineable_saved_tensors_hooks(self):
        # Inlineable (GraphModule) ambient hooks are traced INTO the joint
        # graph, and at runtime AOTAutograd disables the ambient hooks around
        # the compiled call so they do not ALSO fire on ctx.save_for_backward.
        # That disable is plain Python the compose cannot splice, so composing
        # must refuse: serving would pack already-packed activations.
        pack_gm = torch.fx.symbolic_trace(lambda x: x * 2)
        unpack_gm = torch.fx.symbolic_trace(lambda x: x / 2)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with (
            torch.enable_grad(),
            torch.autograd.graph.saved_tensors_hooks(pack_gm, unpack_gm),
            self.assertRaisesRegex(NotImplementedError, "saved_tensors_hooks"),
        ):
            compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)

    def test_training_backward_lowers_with_is_backward(self):
        # is_backward gates GraphLowering's backward-only require_contiguous
        # safeguard for untagged implicit-fallback aten ops (#140452); the
        # composed training backward must lower the way torch.compile's
        # backward does. Spy on compile_fx_inner because the safeguard itself
        # only engages on an op with no lowering, which no small graph has.
        import torch._inductor.compile_fx as compile_fx_mod

        seen = []
        orig = compile_fx_mod.compile_fx_inner

        def spy(gm, inputs, **kwargs):
            seen.append(kwargs.get("is_backward", False))
            return orig(gm, inputs, **kwargs)

        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with (
            torch.enable_grad(),
            mock.patch.object(compile_fx_mod, "compile_fx_inner", spy),
        ):
            compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        self.assertEqual(seen, [False, True])  # forward, then backward

    def test_training_compiled_function_mirrors_runtime_class_attributes(self):
        # The emitted _CompiledFunction stands in for AOTAutograd's CompiledFunction,
        # whose class attributes the runtime and compiled autograd read
        # (_compiled_autograd_key, metadata, _bw_prologue_fn, ...). Every name the
        # real class body binds must exist on the emitted one.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        ns = {"__name__": "_compiled"}
        exec(compile(src, "<compiled>", "exec"), ns)
        emitted = ns["_CompiledFunction"]

        torch._dynamo.reset()
        real = torch.compile(m, backend="aot_eager")(x).grad_fn._forward_cls
        expected = [n for n in vars(real) if not n.startswith("__")]
        self.assertIn("_compiled_autograd_key", expected)
        for name in expected:
            # dir(), not hasattr: _lazy_backward_info is a descriptor that
            # raises (an AttributeError subclass), so hasattr reads it as absent.
            self.assertIn(name, dir(emitted), f"missing {name}")
        self.assertIs(emitted.compiled_fw, ns["_inner_call_fw"])
        self.assertIs(emitted.compiled_bw, ns["_inner_call_bw"])
        self.assertIsNone(emitted.maybe_subclass_metadata)
        with self.assertRaisesRegex(NotImplementedError, "compiled autograd"):
            emitted._lazy_backward_info
        self.assertEqual(emitted.num_symints_saved_for_bw, 0)
        self.assertEqual(
            emitted.metadata.num_forward_returns, real.metadata.num_forward_returns
        )

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    @functorch_config.patch(donated_buffer=True)
    def test_training_donated_buffer_retain_graph_matches_torch_compile(self):
        # The composed backward is lowered under a fresh TracingContext with no
        # fw_metadata, so inductor never marks a donated buffer in it and nothing
        # can be overwritten: retain_graph=True must simply work, and the
        # accumulated gradients must match torch.compile's, whose backward also
        # forgoes donation when its first backward runs with retain_graph=True.
        def f(x, w):
            return torch.tanh(x @ w).sum()

        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)
        with torch.enable_grad():
            gm = make_fx(f)(x, w)
            src, _cache = compile_to_python(gm, [x, w], grad_enabled=True)
        self.assertNotIn("_get_current_graph_task_keep_graph", src)
        out = _exec(src)([x, w])[0]
        self.assertEqual(out, f(x, w))
        out.backward(retain_graph=True)
        out.backward(retain_graph=True)
        served = (x.grad.clone(), w.grad.clone())

        x.grad = None
        w.grad = None
        torch._dynamo.reset()
        with fresh_cache():
            compiled_out = torch.compile(f, backend="inductor")(x, w)
            compiled_out.backward(retain_graph=True)
            compiled_out.backward(retain_graph=True)
        self.assertEqual(served, (x.grad, w.grad))

    def test_training_artifact_execs_in_fresh_process(self):
        # The artifact's imports are only exercised in a process where nothing has
        # imported torch yet: a cold ``from ...runtime_wrappers import`` is a
        # circular import with _dynamo only when it is the first torch import, so
        # the invariant is that ``import torch`` comes first.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        imports = [l for l in src.splitlines() if l.startswith(("import", "from"))]
        self.assertEqual(imports[0], "import torch")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            path = fh.name
        driver = textwrap.dedent(
            f"""
            import torch
            ns = {{"__name__": "_fresh_artifact"}}
            with open({path!r}) as fh:
                exec(compile(fh.read(), {path!r}, "exec"), ns)
            torch.manual_seed(0)
            w = torch.randn(3, 4, requires_grad=True)
            b = torch.randn(3, requires_grad=True)
            x = torch.randn(5, 4)
            out = ns["call"]([w, b, x])[0]
            assert torch.allclose(out, torch.nn.functional.linear(x, w, b)), "output"
            out.sum().backward()
            assert torch.allclose(w.grad, x.sum(0).expand(3, 4)), "grad"
            print("FRESH_OK")
            """
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                capture_output=True,
                text=True,
                timeout=600,
            )
        finally:
            os.remove(path)
        self.assertEqual(
            proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        self.assertIn("FRESH_OK", proc.stdout)

    def test_restride_rebuilds_symbolic_strides_and_refuses_unknown_symbols(self):
        # CompiledFxGraph.output_strides entries are PRINTED stride
        # expressions (strings). A symbolic one is rebuilt over the placeholder's
        # shape symbols and compared proof-only, so a matching stride leaves the
        # symbolic placeholder alone; a symbol the shape env never created, or an
        # unevaluable string, refuses, which keeps the pickled-bundle fallback.
        import types as types_mod

        from torch._functorch._aot_autograd.to_standalone_python import (
            _restride_backward_placeholders,
        )

        spec = types_mod.SimpleNamespace(
            fw_metadata=types_mod.SimpleNamespace(
                tensors_saved_for_backwards_slice=slice(0, 1)
            ),
            num_symints_saved_for_bw=0,
        )
        static = make_fx(lambda x: x * 2)(torch.randn(2, 3))
        _restride_backward_placeholders(static, [("3", "1")], spec)
        with self.assertRaisesRegex(NotImplementedError, "unevaluable"):
            _restride_backward_placeholders(static, [("3*s0", "1")], spec)

        symbolic = make_fx(lambda x: x * 2, tracing_mode="symbolic")(torch.randn(2, 3))
        ph = next(n for n in symbolic.graph.nodes if n.op == "placeholder")
        val = ph.meta["val"]
        self.assertIsInstance(val.stride()[0], torch.SymInt)
        _restride_backward_placeholders(symbolic, [(str(val.stride()[0]), "1")], spec)
        self.assertIs(ph.meta["val"], val)
        with self.assertRaisesRegex(NotImplementedError, "unevaluable"):
            _restride_backward_placeholders(symbolic, [("u0", "1")], spec)
        # A genuinely different (transposed) layout is applied, kept symbolic (stride s0, hint 2).
        _restride_backward_placeholders(symbolic, [("1", str(val.size()[0]))], spec)
        self.assertEqual(tuple(ph.meta["val"].stride()), (1, 2))

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_backed_dynamic_dims_match_eager_across_shapes(self):
        # A symbolically traced Linear has symbolic saved-activation strides, so
        # the backward restride has to evaluate them rather than refuse; the one
        # composed module then serves several shapes with eager's gradients.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4, requires_grad=True)
        gm = _capture(m, x, tracing_mode="symbolic")
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        self.assertIn("class _CompiledFunction(torch.autograd.Function):", src)
        fn = _exec(src)
        for n in (5, 7):
            xi = torch.randn(n, 4, requires_grad=True)
            m.zero_grad(set_to_none=True)
            eager = m(xi)
            eager.sum().backward()
            expected = [p.grad.clone() for p in m.parameters()] + [xi.grad.clone()]
            m.zero_grad(set_to_none=True)
            xi.grad = None
            out = fn(_flat_inputs(m, xi))[0]
            self.assertEqual(out, eager)
            out.sum().backward()
            got = [p.grad for p in m.parameters()] + [xi.grad]
            self.assertEqual(got, expected)

    def test_training_symbolic_metadata_keeps_dynamic_saved_tensors_idxs(self):
        # dynamic_saved_tensors_idxs is set in ViewAndMutationMeta.__post_init__ and
        # ViewAndMutationMeta.__eq__ is blind to it, so a value-reduction of the
        # metadata drops it. The composer reassigns it after construction; a symbolic
        # capture (its saved activations carry symbolic dims) is where the dict is
        # non-empty, so it guards the reassignment -- it is empty without the fix.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4, requires_grad=True)
        gm = _capture(m, x, tracing_mode="symbolic")
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        ns = {"__name__": "_compiled"}
        exec(compile(src, "<compiled>", "exec"), ns)
        emitted = ns["_CompiledFunction"]
        dsi = emitted.metadata.dynamic_saved_tensors_idxs
        self.assertTrue(dsi)
        self.assertTrue(
            all(
                isinstance(k, int) and all(isinstance(v, int) for v in vs)
                for k, vs in dsi.items()
            )
        )

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    @parametrize(
        "f",
        [
            subtest(_two_views_of_intermediate, name="two_views_of_intermediate"),
            subtest(_view_of_input_and_dense, name="view_of_input"),
        ],
    )
    def test_training_output_alias_composes_and_matches_eager(self, f):
        # An aliased output makes the orchestration close over the codegen'd
        # output_alias_wrapper (_alias_fn / gen_alias_from_base), which the
        # training composer splices like the inference one. Values, aliasing,
        # requires_grad and gradients must all match eager.
        x0, w0 = torch.randn(4, 4), torch.randn(4, 4)

        def inputs():
            return x0.clone().requires_grad_(), w0.clone().requires_grad_()

        with torch.enable_grad():
            gm = make_fx(f)(*inputs())
            src, _cache = compile_to_python(gm, list(inputs()), grad_enabled=True)
        self.assertIn("def _alias_fn(", src)
        self.assertIn("gen_alias_from_base", src)

        ex, ew = inputs()
        expected = f(ex, ew)
        sum(o.sum() for o in expected).backward()
        x, w = inputs()
        outs = _exec(src)([x, w])
        self.assertEqual(len(outs), len(expected))
        for got, want in zip(outs, expected):
            self.assertEqual(got, want)
            self.assertEqual(got.requires_grad, want.requires_grad)
        # The views share storage exactly as eager's do.
        ptr = lambda t: t.untyped_storage().data_ptr()  # noqa: E731
        self.assertEqual(
            ptr(outs[0]) == ptr(outs[1]), ptr(expected[0]) == ptr(expected[1])
        )
        if f is _view_of_input_and_dense:
            self.assertEqual(ptr(outs[0]), ptr(x))
        sum(o.sum() for o in outs).backward()
        self.assertEqual((x.grad, w.grad), (ex.grad, ew.grad))

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_input_mutation_on_non_leaf_composes_and_matches_eager(self):
        # Mutating an input that requires grad keeps the mutation OUT of the
        # graph (it is replayed by the codegen'd mutation_epilogue, which the
        # training composer splices). The caller's tensor must end up mutated,
        # and the gradient must flow through the mutation to its base.
        def f(x, w):
            x.mul_(2)
            return torch.tanh(x @ w)

        x0, w0 = torch.randn(4, 4), torch.randn(4, 4)

        def inputs():
            base = x0.clone().requires_grad_()
            return base, base * 1, w0.clone().requires_grad_()

        with torch.enable_grad():
            _, x, w = inputs()
            gm = make_fx(f)(x, w)
            _, x, w = inputs()
            src, _cache = compile_to_python(gm, [x, w], grad_enabled=True)
        self.assertIn("def _apply_mutations(", src)

        e_base, ex, ew = inputs()
        expected = f(ex, ew)
        expected.sum().backward()
        base, x, w = inputs()
        out = _exec(src)([x, w])[0]
        self.assertEqual(out, expected)
        self.assertEqual(x, ex)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertEqual((base.grad, w.grad), (e_base.grad, ew.grad))

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    @parametrize(
        "seed_backward",
        [
            subtest(lambda outs: outs[0].sum().backward(), name="partial_outputs"),
            subtest(
                lambda outs: outs[0].backward(torch.arange(15.0).view(3, 5).t()),
                name="noncontiguous_tangent",
            ),
        ],
    )
    def test_training_multi_output_backward_seeds_match_eager(self, seed_backward):
        # A backward through ONE of several outputs hands the emitted
        # _CompiledFunction None tangents for the rest, and a non-contiguous
        # tangent has to be coerced to the layout the backward was traced for
        # (the codegen'd backward prologue's tangent processing). Both must
        # land on eager's gradients.
        def f(x, w):
            return torch.tanh(x @ w), (x * 2).sum(0), torch.relu(x)

        x0, w0 = torch.randn(5, 4), torch.randn(4, 3)

        def inputs():
            return x0.clone().requires_grad_(), w0.clone().requires_grad_()

        with torch.enable_grad():
            gm = make_fx(f)(*inputs())
            src, _cache = compile_to_python(gm, list(inputs()), grad_enabled=True)
        ex, ew = inputs()
        seed_backward(f(ex, ew))
        x, w = inputs()
        seed_backward(_exec(src)([x, w]))
        self.assertEqual((x.grad, w.grad), (ex.grad, ew.grad))

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_optimizer_step_matches_eager(self):
        # The served module's gradients land on the caller's parameters, so an
        # optimizer step over them must produce eager's updated values.
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        served = copy.deepcopy(m)
        m(x).sum().backward()
        torch.optim.SGD(m.parameters(), lr=0.1).step()
        _exec(src)(_flat_inputs(served, x))[0].sum().backward()
        torch.optim.SGD(served.parameters(), lr=0.1).step()
        self.assertEqual(dict(served.named_parameters()), dict(m.named_parameters()))

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_partial_requires_grad_matches_eager(self):
        # An input that does not require grad is non-differentiable at the AOT
        # boundary: it gets no gradient, and the others get eager's.
        m = torch.nn.Linear(4, 3)
        m.bias.requires_grad_(False)
        x = torch.randn(5, 4, requires_grad=True)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        m(x).sum().backward()
        expected = (m.weight.grad.clone(), x.grad.clone())
        m.zero_grad(set_to_none=True)
        x.grad = None
        _exec(src)(_flat_inputs(m, x))[0].sum().backward()
        self.assertIsNone(m.bias.grad)
        self.assertEqual((m.weight.grad, x.grad), expected)

    def test_linear_addmm_runs_like_eager(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_sequential_linear_relu_runs_like_eager(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_runs_like_eager(self):
        m = _SumDim1().eval()
        x = torch.randn(6, 7)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_graph_has_dynamic_shapes_reads_example_value(self):
        # A Dynamo graph stashes its fake under "example_value", not "val".
        from torch._functorch._aot_autograd.to_standalone_python import (
            _graph_has_dynamic_shapes,
        )
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertFalse(_graph_has_dynamic_shapes(gm))
        mode = FakeTensorMode(shape_env=ShapeEnv())
        node.meta["example_value"] = mode.from_tensor(
            torch.randn(3), static_shapes=False
        )
        self.assertTrue(_graph_has_dynamic_shapes(gm))

    def test_dynamic_shapes_runs_at_multiple_shapes(self):
        # compile_to_python has no dynamic_shapes knob: it auto-detects symbolic shapes
        # (_graph_has_dynamic_shapes) and picks the internal shapes_mode accordingly. A
        # symbolically-traced graph selects "from_graph", composing one module keyed on
        # symbolic sizes rather than baked constants, and that single module runs at
        # multiple shapes. (A statically-traced graph selects "from_example_inputs", which
        # specializes to the example shapes instead.)
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        _assert_composed(self, src)
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    def test_multi_output_runs_like_eager(self):
        # Exercises the output epilogue's multi-output count/ordering: the composed module
        # must return all outputs in the captured order, each equal to eager.
        m = _MultiOut().eval()
        x = torch.randn(6, 7)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        eager = m(x)
        with torch.no_grad():
            out = _exec(src)(_flat_inputs(m, x))
        self.assertEqual(len(out), len(eager))
        for got, want in zip(out, eager):
            self.assertEqual(got, want)

    def test_input_mutation_copy_back_runs_like_eager(self):
        # A buffer mutated in place exercises AOTAutograd's mutation epilogue (input copy-
        # back): the composed call must reflect the mutation onto the passed-in buffer
        # tensor, exactly as eager mutates m.b. Compare both the output AND the mutated
        # input.
        m = _BufferMutate().eval()
        x = torch.randn(4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)

        eager = _BufferMutate().eval()
        eager_out = eager(x)

        buf = torch.zeros(4)
        with torch.no_grad():
            composed_out = _exec(src)([buf, x])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(buf, eager.b)

    def test_output_alias_regen_runs_like_eager(self):
        # An output that aliases an input exercises AOTAutograd's output-alias regeneration
        # (the _alias_fn / gen_alias_from_base path, which the orchestration closes over
        # directly). The composed output must both equal eager AND alias the input's
        # storage, exactly as eager's view does.
        m = _ViewAlias().eval()
        x = torch.randn(4, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("gen_alias_from_base", src)
        # Pin the view-replay reconstruction branches this PR adds (the new
        # ViewMetaSequence._from_parts factory + the ViewMeta as_tuple branch). The
        # numerics/aliasing asserts alone would not catch a wrong reconstruction on this
        # tiny view, so assert the emission explicitly.
        self.assertIn("ViewMetaSequence._from_parts(", src)
        self.assertIn("ViewMeta(", src)
        xc = x.clone()
        with torch.no_grad():
            out = _exec(src)([xc])[0]
        self.assertEqual(out, m(x))
        self.assertEqual(
            out.untyped_storage().data_ptr(), xc.untyped_storage().data_ptr()
        )

    def test_dedup_mutated_duplicate_input_runs_like_eager(self):
        # A mutated DUPLICATE input drives AOTDedupeWrapper -- a CompilerWrapper applied
        # AROUND the orchestration (graph_compile._aot_stage2c_make_inference_function),
        # NOT an inner chain wrapper. The composed module must nest it OUTSIDE the
        # orchestration -- dedup(orchestration(inner)) -- so the orchestration sees the
        # DEDUPED args. Composed inside-out (the pre-fix bug) the version bump / copy-back
        # index the raw pre-dedup args and land on the wrong (duplicate) tensor, so the
        # version-counter assertion below fails; the ``_orchestration_entry`` adapter is
        # the structural marker of the correct outer nesting (absent pre-fix).
        def f(a, b, c, d):
            d.mul_(2)
            return a + d

        x = torch.randn(4)
        y = torch.randn(4)
        gm = make_fx(f)(x, x, y, y)
        src, _cache = compile_to_python(gm, [x, x, y, y])
        self.assertIn("def call(flat_inputs):", src)
        self.assertIn("_orchestration_entry", src)  # outer-wrapper adapter over orch
        self.assertIn("deduped_args", src)  # the dedup wrapper is spliced

        xe, ye = x.clone(), y.clone()
        eager_out = f(xe, xe, ye, ye)

        xc, yc = x.clone(), y.clone()
        with torch.no_grad():
            composed_out = _exec(src)([xc, xc, yc, yc])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(yc, ye)  # mutation landed on the right tensor
        self.assertEqual(yc._version, ye._version)  # and bumped the right version

    def test_synthetic_base_aliased_mutated_input_runs_like_eager(self):
        # Aliased + mutated inputs drive AOTSyntheticBaseWrapper, also applied AROUND the
        # orchestration. It collapses the aliased inputs into a synthetic base BEFORE the
        # orchestration runs, so it must be composed OUTSIDE it (same requirement as dedup):
        # the ``_orchestration_entry`` adapter marks the correct nesting (absent pre-fix,
        # which inverted the chain to orchestration(synthetic_base(inner))).
        def f(a, b):
            a.mul_(2)
            return a + b

        def make():
            base = torch.arange(4, dtype=torch.float32) + 1
            return base[:], base  # a aliases b

        gm = make_fx(f)(*make())
        a0, b0 = make()
        src, _cache = compile_to_python(gm, [a0, b0])
        self.assertIn("_orchestration_entry", src)
        self.assertIn("_synthetic_base_wrapper", src)

        ae, be = make()
        eager_out = f(ae, be)

        ac, bc = make()
        with torch.no_grad():
            composed_out = _exec(src)([ac, bc])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(ac, ae)
        self.assertEqual(bc, be)
        self.assertEqual(bc._version, be._version)

    def test_tensor_subclass_wrap_unwrap_runs_like_eager(self):
        # The headline feature: a tensor-subclass input exercises AOTAutograd's subclass
        # flatten/unflatten wrapper plus baked subclass metadata. The composed module must
        # unwrap the subclass for the inner dense call and re-wrap the output as the same
        # subclass, matching eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            return x * 2.0 + 1.0

        tt = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_multiple_subclass_inputs_runs_like_eager(self):
        # Two tensor-subclass inputs make the subclass wrapper flatten/unflatten more than
        # one subclass (each into its constituents), which the single-input subclass test
        # above does not. NOTE: this stays a single INNER chain wrapper -- a >= 2-link inner
        # chain (e.g. subclass + functionalized-RNG) is only exercised by the order-
        # inversion guard unit test, since a supported multi-inner-chain forward graph is
        # not readily constructible (subclass+RNG hits an internal AOTAutograd assertion,
        # and duplicate subclass inputs collapse into one subclass wrapper). Mutated
        # duplicate / aliased inputs DO add OUTER wrappers (dedup / synthetic base), but
        # those wrap the orchestration rather than the inner call -- see the dedup /
        # synthetic-base tests above.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(a, b):
            return a * 2.0 + b * 3.0

        ta = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        tb = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(ta, tb)
        src, _cache = compile_to_python(gm, [ta, tb])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([ta, tb])[0]
        eager = f(ta, tb)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_autocast_disable_autocast_runs_like_eager(self):
        # disable_amp is read by AOTAutograd at compile time via _is_any_autocast_enabled,
        # so the inner compile MUST run under autocast for the orchestration to emit
        # _DisableAutocast_. The graph is also traced under autocast so the bf16 casts are
        # baked in; the orchestration then disables autocast at runtime to keep the dense
        # call from double-casting. Equivalence is checked against eager run UNDER autocast.
        m = _MatMul().eval()
        x = torch.randn(5, 4)
        pb = _flat_inputs(m, x)

        pnames = [n for n, _ in m.named_parameters()]
        k = len(pnames)
        bnames = [n for n, _ in m.named_buffers()]

        def flat_fn(flat):
            params = dict(zip(pnames, flat[:k]))
            buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
            with stateless._reparametrize_module(
                m, {**params, **buffers}, tie_weights=True
            ):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = m(flat[-1])
            return pytree.tree_flatten(out)[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn, tracing_mode="real")(pb)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            src, _cache = compile_to_python(gm, pb)
        _assert_composed(self, src)
        self.assertIn("_DisableAutocast_", src)
        with torch.no_grad():
            out = _exec(src)(pb)[0]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            eager = m(x)
        self.assertEqual(out, eager)

    def test_options_passthrough_runs_like_eager(self):
        # compile_to_python forwards ``options`` straight to the inner inductor compile in a
        # single line. Use nan_asserts=True (default False) as the probe: it is observable in
        # the inner source as an ``isnan`` check, so the assertion below FAILS if the forward
        # were dropped (the option would fall back to the False default and emit no isnan) --
        # unlike an option whose value equals its default, which could not detect a dropped
        # forward. nan_asserts only adds runtime checks, so numerics still match eager.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"nan_asserts": True}
        )
        _assert_composed(self, src)
        self.assertIn("isnan", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_orchestration_inlined_as_real_def(self):
        # The orchestration is spliced as a real top-level ``def _runtime_wrapper`` that the
        # outer ``call`` invokes directly by name -- no string re-exec, no ``_orchestration``
        # alias. All wrappers are inlined now, so ``_exec_wrapper`` no longer exists in any
        # composed module; the module reads as ordinary code and must still exec like eager.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, _cache = _compose(m, x)
        self.assertIn("def _runtime_wrapper(", src)
        self.assertIn("return _runtime_wrapper(", src)
        self.assertNotIn("_orchestration", src)  # redundant alias removed
        self.assertNotIn("_exec_wrapper", src)
        # Pin the deliberate drop of the first-invocation context / profiler prologue:
        # the orchestration is invoked with contextlib.nullcontext + a no-op in those two
        # positional slots. A future change re-threading a real context here would fail.
        self.assertIn(", contextlib.nullcontext, lambda: None,", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_chain_wrapper_inlined_as_real_def(self):
        # A graph with a chain wrapper (tensor subclass -> ``inner_fn``, which closes over
        # the inner via a ``compiled_fn`` global) is now inlined too: the wrapper is a real
        # top-level def with ``compiled_fn`` hoisted to a module-scope assignment, no exec /
        # string blob anywhere. Numerics must match eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            return x * 2.0 + 1.0

        tt = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        self.assertNotIn("_exec_wrapper", src)
        self.assertNotIn("_src = ", src)  # no re-exec'd source-string blobs
        self.assertIn("def inner_fn(", src)
        self.assertIn("compiled_fn = _inner_call", src)
        self.assertIn("def _runtime_wrapper(", src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "functionalize_rng_ops threads CUDA RNG state via CUDARngStateHelper, which "
        "requires a CUDA device (the graph itself lowers through the CPU backend).",
    )
    def test_functionalized_rng_runs_like_eager(self):
        # functionalize_rng_ops rewrites the RNG op into a functional form during the inner
        # AOTAutograd lowering, producing a FunctionalizedRngRuntimeWrapper that threads RNG
        # state via CUDARngStateHelper. ``CUDARngStateHelper`` in the source is the RNG-specific
        # signal (it appears only under functionalize_rng_ops and exercises the helper-table
        # rows); the wrapper's ``_compiled_fn_`` inner-name is real but not asserted here, as
        # that token is the orchestration's first parameter and appears in every composed
        # module. Seeded so dropout's mask is deterministic for the eager comparison.
        class _Dropout(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.dropout(x, p=0.5, training=True)

        m = _Dropout()
        x = torch.randn(8, 4)

        def flat_fn(flat):
            return pytree.tree_flatten(m(flat[-1]))[0]

        with functorch_config.patch(functionalize_rng_ops=True):
            with torch.enable_grad():
                gm = make_fx(flat_fn, tracing_mode="real")([x])
            src, _cache = compile_to_python(gm, [x])
        _assert_composed(self, src)
        self.assertIn("CUDARngStateHelper", src)
        fn = _exec(src)
        torch.manual_seed(123)
        with torch.no_grad():
            out = fn([x])[0]
        torch.manual_seed(123)
        with torch.no_grad():
            eager = m(x)
        self.assertEqual(out, eager)

    def test_helpers_imported_from_standalone_runtime_surface(self):
        # End-to-end lock for the stability contract: a graph closing over a runtime helper
        # (here gen_alias_from_base, via output-alias regen) must import it from the
        # standalone_runtime surface, not its internal AOTAutograd location. A dropped or
        # aliased _known_helper_table entry would silently fall through to the internal
        # module. (ViewMetaSequence legitimately imports functional_utils, so this checks the
        # specific helper expression, not the bare module name.)
        m = _ViewAlias().eval()
        x = torch.randn(4, 4)
        src, _cache = _compose(m, x)
        self.assertIn(
            "from torch._functorch._aot_autograd.standalone_runtime import "
            "gen_alias_from_base",
            src,
        )
        self.assertNotIn("functional_utils.gen_alias_from_base", src)

    def test_training_helpers_imported_from_standalone_runtime_surface(self):
        # A training artifact's backward closes over runtime helpers
        # (AOTDispatchAutograd.process_runtime_tangent, TensorAlias); those must import
        # from the standalone_runtime surface, not their internal AOTAutograd module.
        # An unused BackwardState must not leak an import either (compose refuses
        # compiled autograd, so the branch that binds it is never emitted).
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4, requires_grad=True)
        gm = _capture(m, x)
        with torch.enable_grad():
            src, _cache = compile_to_python(gm, _flat_inputs(m, x), grad_enabled=True)
        self.assertIn(
            "from torch._functorch._aot_autograd.standalone_runtime import "
            "AOTDispatchAutograd",
            src,
        )
        self.assertIn(
            "from torch._functorch._aot_autograd.standalone_runtime import TensorAlias",
            src,
        )
        self.assertNotIn("import torch._functorch._aot_autograd.runtime_wrappers", src)
        self.assertNotIn("_backward_state", src)

    @parametrize("target", _EFFECTFUL_TARGETS)
    def test_rejects_effectful_op(self, target):
        # A graph carrying an effectful op is rejected up front with a concrete
        # NotImplementedError -- effect tokens thread through a calling convention the
        # standalone composition does not reproduce. Both an effectful OpOverload
        # (aten._print) and an effectful HigherOrderOperator (hop_print) must be caught;
        # the HOP case would slip past an OpOverload-only gate.
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(target, ("hello",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        with self.assertRaisesRegex(NotImplementedError, "effectful op"):
            compile_to_python(gm, [torch.randn(3)])

    def test_rejects_non_graphmodule(self):
        # The effectful-op scan dereferences gm.graph before reaching inductor's own check,
        # so the functorch layer must reject a non-GraphModule with a clean TypeError rather
        # than an opaque AttributeError.
        with self.assertRaisesRegex(TypeError, "expects a post-AOTAutograd"):
            compile_to_python("not a graph module", [])

    def test_reentrant_compile_to_python_under_held_lock(self):
        # The entry point takes a re-entrant lock (RLock) so a nested on-thread compile (a
        # custom backend / inductor pass re-entering during compile) does not self-deadlock.
        # Holding the lock and compiling exercises that re-entry: a plain Lock would hang
        # here (surfacing as a CI timeout), so this pins the RLock choice behaviorally.
        from torch._functorch._aot_autograd.to_standalone_python import _COMPILE_LOCK

        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        with _COMPILE_LOCK:
            src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_concurrent_compile_to_python_smoke(self):
        # End-to-end concurrency smoke test: _COMPILE_LOCK serializes the entry point (the
        # underlying cache-state swap is process-global), so two threads compiling different
        # graphs run one-at-a-time and must each still produce their own correct module.
        # This exercises the lock end-to-end (no deadlock, both succeed); the thread-local
        # sink isolation it relies on is pinned by test_capture_sink_is_thread_local.
        #
        # Only compile_to_python runs concurrently. The graphs are captured SERIALLY up
        # front because make_fx (fx symbolic tracing) is NOT thread-safe: it mutates
        # process-global tracing state -- the CURRENT_PATCHER global in
        # torch.fx._symbolic_trace and the fx.traceback node-meta globals. Capturing inside
        # the worker threads would race that state against the make_fx run nested inside the
        # OTHER thread's compile_to_python (which _COMPILE_LOCK covers, but capture is not
        # the serialized entry point), intermittently raising "CURRENT_PATCHER is None in
        # finally block" -- a make_fx concurrency artifact unrelated to the lock under test.
        import traceback

        specs = [
            (_Pointwise().eval(), torch.randn(8, 4)),
            (_SumDim1().eval(), torch.randn(6, 7)),
        ]
        captured = [(_capture(m, x), m, x) for m, x in specs]
        results: dict = {}
        errors: dict = {}

        def run(i, gm, m, x):
            try:
                results[i] = (compile_to_python(gm, _flat_inputs(m, x))[0], m, x)
            except Exception:
                # Surface the full traceback so a future failure is diagnosable from CI
                # logs; the bare {1: <Exception>} the swallow used to leave hid the cause.
                errors[i] = traceback.format_exc()

        threads = [
            threading.Thread(target=run, args=(i, gm, m, x))
            for i, (gm, m, x) in enumerate(captured)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, {}, "\n".join(errors.values()))
        for src, m, x in results.values():
            _assert_composed(self, src)
            with torch.no_grad():
                self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_capture_sink_is_thread_local(self):
        # The capture sink MUST be thread-local: two threads forced (via a barrier) to be
        # mid-capture simultaneously must each record ONLY their own codegen'd wrapper. A
        # process-global sink would bleed wrappers across threads and fail this -- the direct
        # pin for the thread-local contract that the lock-serialized smoke test above cannot
        # exercise (the lock prevents real overlap there).
        from torch._functorch._aot_autograd.codegen import (
            _compile_and_exec_source,
            capture_generated_sources,
        )

        barrier = threading.Barrier(2)
        sinks: dict = {}

        def run(key):
            into: list = []
            with capture_generated_sources(into):
                barrier.wait(
                    timeout=60
                )  # both threads now inside their capture context
                _compile_and_exec_source(
                    f"def {key}_fn(args):\n    return args\n", {}, f"{key}_fn", key
                )
                barrier.wait(
                    timeout=60
                )  # hold both contexts open across the other's codegen
            sinks[key] = [g.fn_name for g in into]

        threads = [threading.Thread(target=run, args=(k,)) for k in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sinks["a"], ["a_fn"])
        self.assertEqual(sinks["b"], ["b_fn"])

    def test_capture_autograd_specs_is_thread_local(self):
        # Same contract as the wrapper-source sink: two threads mid-capture at once
        # must each record only the spec built on their own thread, even though the
        # factory.build hook itself is installed process-wide.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _AOTDispatchAutogradFunctionFactory as factory,
        )

        barrier = threading.Barrier(2)
        sinks: dict = {}

        def run(key):
            into: list = []
            with _capture_autograd_specs(into):
                barrier.wait(timeout=60)
                factory(spec=key).build()
                barrier.wait(timeout=60)
            sinks[key] = [spec for _origin, spec in into]

        with mock.patch.object(factory, "build", lambda self: None):
            threads = [threading.Thread(target=run, args=(k,)) for k in ("a", "b")]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(sinks, {"a": ["a"], "b": ["b"]})


@instantiate_parametrized_tests
class TestComposerHelpers(TestCase):
    # Unit coverage of the composer's own helpers: the _known_helper_table stable-import
    # contract, _module_level_names (inner-binding collision seeding), and the recursive
    # _find_effectful_op scan. (Source-emission helper tests live in test_source_emit.py.)

    def test_known_helper_table_imports_are_stable_surface(self):
        # Stability contract: every runtime helper the composer recognizes must emit an
        # import via the stable standalone_runtime surface (or `import torch` for public
        # torch paths), never a deep AOTAutograd-internal module. Lock the table so a new
        # entry pointing at an unstable location is caught.
        for import_stmt, _expr in _known_helper_table().values():
            self.assertTrue(
                import_stmt == "import torch"
                or import_stmt.startswith(
                    "from torch._functorch._aot_autograd.standalone_runtime import "
                ),
                f"helper import {import_stmt!r} bypasses the standalone_runtime surface",
            )

    def test_select_training_spec_ignores_foreign_build(self):
        # A re-entrant lowering during the capture window builds its own
        # autograd Function under a DISTINCT TracingContext. Its spec is tagged
        # with that context, like the codegen'd wrappers are, so only the spec
        # built alongside the captured orchestration is selected.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            _AOTDispatchAutogradFunctionFactory as factory,
        )

        ours, foreign = object(), object()
        specs = []
        with (
            mock.patch.object(factory, "build", lambda self: None),
            _capture_autograd_specs(specs),
        ):
            ctx = torch._guards.TracingContext(None)
            with torch._guards.tracing(ctx):
                factory(spec=foreign).build()
                factory(spec=ours).build()
                origin = id(ctx)
            with torch._guards.tracing(torch._guards.TracingContext(None)):
                factory(spec=foreign).build()
        self.assertEqual(len(specs), 3)
        orch = GeneratedSource(
            "runtime_wrapper_orchestration", "_runtime_wrapper", "", {}, None, origin
        )
        # Two specs share the orchestration's context: refuse rather than guess.
        with self.assertRaisesRegex(NotImplementedError, "exactly one autograd spec"):
            _select_training_spec(specs, [orch])
        del specs[0]
        self.assertIs(_select_training_spec(specs, [orch]), ours)
        self.assertIs(_select_training_spec(specs, [orch, orch]), ours)

    def test_module_level_names_excludes_deleted(self):
        # Inductor's inner module binds then dels a name (async_compile = AsyncCompile();
        # del async_compile) at module scope. A del'd name does not persist, so it must not
        # be reserved -- otherwise a hoisted wrapper global of the same name would trip a
        # spurious _reserve collision.
        tree = ast.parse("a = 1\nb = 2\ndel a\n")
        names = _module_level_names(tree)
        self.assertIn("b", names)
        self.assertNotIn("a", names)

    @parametrize("target", _EFFECTFUL_TARGETS)
    def test_find_effectful_op_top_level(self, target):
        # The scan must surface an effectful OpOverload AND an effectful
        # HigherOrderOperator (hop_print); the HOP case would return None under an
        # OpOverload-only isinstance check.
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(target, ("hi",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        self.assertIs(_find_effectful_op(gm, _get_effect), target)

    def test_find_effectful_op_nested_in_subgraph(self):
        # An effect nested inside a child GraphModule reached via get_attr must be found.
        child = fx.Graph()
        ca = child.placeholder("a")
        child.call_function(torch.ops.aten._print.default, ("hi",))
        child.output((ca,))
        child_gm = fx.GraphModule(torch.nn.Module(), child)

        parent = fx.Graph()
        pa = parent.placeholder("a")
        parent.get_attr("sub")
        parent.output((pa,))
        root = torch.nn.Module()
        root.sub = child_gm
        parent_gm = fx.GraphModule(root, parent)
        self.assertIs(
            _find_effectful_op(parent_gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_nested_in_container_arg(self):
        # A child GraphModule reached via a container-nested node ARG (not get_attr) -- the
        # form HOPs use to pass a body callable -- must still be scanned for effects.
        child = fx.Graph()
        ca = child.placeholder("a")
        child.call_function(torch.ops.aten._print.default, ("hi",))
        child.output((ca,))
        child_gm = fx.GraphModule(torch.nn.Module(), child)

        parent = fx.Graph()
        pa = parent.placeholder("a")
        parent.call_function(torch.relu, (pa,), {"bodies": [child_gm]})
        parent.output((pa,))
        parent_gm = fx.GraphModule(torch.nn.Module(), parent)
        self.assertIs(
            _find_effectful_op(parent_gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_none_when_pure(self):
        m = _Pointwise().eval()
        gm = _capture(m, torch.randn(4, 4))
        self.assertIsNone(_find_effectful_op(gm, _get_effect))


@requires_cuda_and_triton
class TestAOTCompileToPythonCuda(TestCase):
    # The composition is device-agnostic source manipulation, but its wrappers must also
    # compose correctly around Inductor's @triton.jit kernels and on CUDA tensors. Mirror
    # the key e2e cases on CUDA; the inner-kernel codegen itself is covered by
    # test/inductor/test_compile_to_python.py's CUDA class.
    def test_pointwise_runs_like_eager(self):
        m = _Pointwise().eval().cuda()
        x = torch.randn(8, 4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("@triton.jit", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_output_alias_regen_runs_like_eager(self):
        m = _ViewAlias().eval().cuda()
        x = torch.randn(4, 4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("gen_alias_from_base", src)
        xc = x.clone()
        with torch.no_grad():
            out = _exec(src)([xc])[0]
        self.assertEqual(out, m(x))
        self.assertEqual(
            out.untyped_storage().data_ptr(), xc.untyped_storage().data_ptr()
        )

    def test_tensor_subclass_wrap_unwrap_runs_like_eager(self):
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            return x * 2.0 + 1.0

        tt = TwoTensor(
            torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda")
        )
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_input_mutation_copy_back_runs_like_eager(self):
        # The mutation epilogue's copy-back is the most plausibly device-sensitive wrapper
        # path (it writes updated values back onto the passed-in CUDA tensors), so mirror
        # the CPU mutation case on CUDA in addition to the pointwise/alias/subclass cases.
        m = _BufferMutate().eval().cuda()
        x = torch.randn(4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)

        eager = _BufferMutate().eval().cuda()
        eager_out = eager(x)

        buf = torch.zeros(4, device="cuda")
        with torch.no_grad():
            composed_out = _exec(src)([buf, x])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(buf, eager.b)

    @skipIfTorchDynamo(
        "the emitted _CompiledFunction refuses compiled autograd with "
        "NotImplementedError (no fx bw_module to inline); a feature limitation, "
        "see the module's not-covered list"
    )
    def test_training_conv_restride_matches_eager(self):
        # Conv nets are what the backward restride exists for: inductor's
        # layout optimization hands back channels-last saved activations, and
        # a backward lowered against the joint trace's eager strides raises a
        # size assert -- or, with size asserts off, silently computes wrong
        # gradients. Layout optimization is FORCED so the restride is
        # genuinely engaged (at default heuristics this shape keeps contiguous
        # strides and the test would pass with the restride deleted), and
        # cuDNN TF32 is off so the gradients compare at default tolerance.
        with (
            torch._inductor.config.patch(force_layout_optimization=True),
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False),
        ):
            m = torch.nn.Conv2d(16, 32, 3).cuda()
            x = torch.randn(2, 16, 16, 16, device="cuda")
            for p in m.parameters():
                p.grad = None
            m(x).sum().backward()
            expected = {n: p.grad.detach().clone() for n, p in m.named_parameters()}

            gm = _capture(m, x)
            with torch.enable_grad():
                src, _cache = compile_to_python(
                    gm, _flat_inputs(m, x), grad_enabled=True
                )
            out = _exec(src)(_flat_inputs(m, x))
            out = out[0] if isinstance(out, (list, tuple)) else out
            for p in m.parameters():
                p.grad = None
            out.sum().backward()
            for name, param in m.named_parameters():
                self.assertEqual(param.grad, expected[name])


class TestAOTComposeGuards(TestCase):
    # The composer's defensive guards (which reject rather than emit a subtly-wrong module)
    # only fire if AOTAutograd's codegen drifts, so drive them directly with hand-built
    # GeneratedSource objects rather than waiting for an upstream regression.
    _ORCH_SRC = (
        "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
        "    return _compiled_fn_(args)\n"
    )
    _CHAIN_SRC = "def inner_fn(args):\n    return compiled_fn(args)\n"

    def test_orchestration_signature_guard(self):
        # The generated call invokes the orchestration positionally, so a changed signature
        # must fail loudly rather than silently pass wrong arguments.
        bad_orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(wrong, args):\n    return None\n",
            {},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "orchestration wrapper signature"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [bad_orch],
                _SENTINEL_INNER_CALL,
            )

    def test_orchestration_extra_kwonly_param_rejected(self):
        # The 4 positional params are intact but a keyword-only param is added. The standalone
        # call is purely positional, so a kw-only-with-default would be silently dropped; the
        # guard must compare the FULL signature and reject this, not just the positional list.
        kwonly_orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args, "
            "*, new_flag=None):\n    return _compiled_fn_(args)\n",
            {},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "orchestration wrapper signature"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [kwonly_orch],
                _SENTINEL_INNER_CALL,
            )

    def test_empty_capture_rejected(self):
        # The real backstop for an incomplete capture (e.g. if a future change offloaded
        # wrapper codegen to a worker thread so nothing was captured): the composer requires
        # exactly one forward orchestration wrapper and rejects an empty capture rather than
        # emitting a partial module.
        with self.assertRaisesRegex(
            NotImplementedError, "exactly one forward orchestration wrapper"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [], _SENTINEL_INNER_CALL
            )

    def test_orchestration_global_colliding_with_inner_rejected(self):
        # The inlined orchestration hoists its globals to module scope; a hoisted name that
        # shadows a top-level name the inner module already binds is rejected (rather than
        # silently rebinding it). ``aten`` is a real inner top-level binding; the resolved
        # expr (a helper) differs from the name, so it is hoisted and trips the guard.
        inner = "aten = 1\ndef call(args):\n    return args\n"
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {"aten": torch.autograd.graph.increment_version},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "collides with another top-level name in the composed"
        ):
            _compose_standalone_module(inner, [orch], _SENTINEL_INNER_CALL)

    def test_orchestration_def_name_colliding_with_inner_rejected(self):
        # Distinct from the hoisted-global collision above: the up-front _reserve loop
        # reserves every wrapper DEF name before any global is hoisted, so an inner module
        # that binds a top-level name equal to a wrapper's fn_name ("_runtime_wrapper")
        # trips that earlier guard rather than the hoist path.
        inner = "_runtime_wrapper = 1\ndef call(args):\n    return args\n"
        with self.assertRaisesRegex(
            NotImplementedError, "collides with another top-level name in the composed"
        ):
            _compose_standalone_module(inner, [self._orch()], _SENTINEL_INNER_CALL)

    def test_compose_rejects_baked_live_tensor(self):
        # The module's no-weights / no-pickle.loads promise: a wrapper closing over a live
        # tensor (with no import path) must RAISE at compose time rather than embed raw
        # bytes. This drives that rejection through the COMPOSE path (a wrapper global),
        # not just emit_value in isolation -- the _assert_composed markers only prove no
        # blob appears when nothing forces one, so this is what actually locks the promise.
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
            "    return [_baked]\n",
            {"_baked": torch.randn(4)},
            lambda: None,
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot bake a live Tensor"):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [orch], _SENTINEL_INNER_CALL
            )

    def test_rebuild_helper_spliced_and_runs_in_composed_module(self):
        # When a baked global reconstructs via the pickle-reduce-as-source path (_NewObjEx
        # emits ``_rebuild(...)``), the composer must splice the _rebuild helper into the
        # module (needs_rebuild) AND _rebuild must actually run at module-exec time to
        # reconstruct the value. The _emit_via_reduce unit tests cover the _rebuild logic in
        # isolation; this is the only coverage of the splice + in-module execution.
        baked = _NewObjEx(1, b=2)
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
            "    return [_baked]\n",
            {"_baked": baked},
            lambda: None,
        )
        src = _compose_standalone_module(
            "def call(args):\n    return args\n", [orch], _SENTINEL_INNER_CALL
        )
        self.assertIn("def _rebuild", src)  # helper spliced (needs_rebuild=True)
        self.assertIn("_rebuild(", src)
        out = _exec(src)(
            []
        )  # exec the module; _rebuild runs to rebuild the baked value
        self.assertEqual(out[0], baked)

    def test_chain_head_order_inversion_guard(self):
        # Inner-chain capture order is assumed innermost-to-outermost. Feed it OUTER-first
        # (inverted): the outer wrapper (wraps the inner wrapper) is captured before the
        # inner wrapper (which wraps the authoritative inner call ``inner_fn``), so the
        # "last with an inner-ref" head is actually wrapped by an earlier sibling.
        def inner_fn(args):
            return args

        def fn_a(args):
            return args

        def fn_b(args):
            return args

        def orch_fn():
            return None

        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {},
            orch_fn,
        )
        # outer (fn_b) wraps the inner wrapper (fn_a); inner (fn_a) wraps the dense call.
        outer = GeneratedSource(
            "dedup_wrapper", "inner_fn", self._CHAIN_SRC, {"compiled_fn": fn_a}, fn_b
        )
        inner = GeneratedSource(
            "dedup_wrapper",
            "inner_fn",
            self._CHAIN_SRC,
            {"compiled_fn": inner_fn},
            fn_a,
        )
        with self.assertRaisesRegex(NotImplementedError, "innermost-to-outermost"):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [outer, inner, orch], inner_fn
            )

    def _orch(self, origin_id=None):
        # A valid forward orchestration wrapper whose codegen'd signature matches what the
        # composer invokes positionally; the guard tests below pair it with a deliberately
        # broken sibling so the SIBLING is what trips the guard, not a missing orchestration.
        return GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {},
            lambda: None,
            origin_id,
        )

    def test_backward_wrapper_rejected(self):
        # A backward wrapper is out of scope for forward lowering, so it is rejected up
        # front (before chain wiring) even when paired with a valid forward orchestration:
        # an "in backward" artifact_name must fail loudly rather than be spliced in.
        bwd = GeneratedSource(
            "backward_subclass_wrapper",
            "wrap_fn",
            "def wrap_fn(unwrapped_outs):\n    return unwrapped_outs\n",
            {},
            lambda args: args,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "cannot yet compose these runtime"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [bwd, self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_unwired_chain_wrapper_rejected(self):
        # A chain wrapper that names the inner it wraps via a global NOT in _INNER_NAMES
        # (here "mystery_inner") is invisible to chain-head/inner-call detection, so it can
        # never be wired into the module. If _INNER_NAMES drifts out of sync with a new
        # AOTAutograd inner-ref global this is exactly the shape that arises; reject it
        # rather than silently emit a structurally-wrong module.
        def chain_fn(args):
            return args

        # mystery_inner is bound to an importable module-level function so its global
        # resolves cleanly as source -- the point under test is the UNRECOGNIZED global
        # NAME (not in _INNER_NAMES), which is what leaves the wrapper unwired.
        mystery = GeneratedSource(
            "dedup_wrapper",
            "inner_fn",
            "def inner_fn(args):\n    return mystery_inner(args)\n",
            {"mystery_inner": _make_holder},
            chain_fn,
        )
        with self.assertRaisesRegex(NotImplementedError, "could not wire"):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [mystery, self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_multiple_orchestrations_rejected(self):
        # Two orchestration wrappers sharing one origin_id (so the origin filter keeps both)
        # is an impossible capture for a single forward; the composer requires exactly one
        # and must reject the ambiguous pair rather than pick one arbitrarily.
        with self.assertRaisesRegex(
            NotImplementedError, "exactly one forward orchestration"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [self._orch(origin_id=5), self._orch(origin_id=5)],
                _SENTINEL_INNER_CALL,
            )

    def test_foreign_origin_wrapper_filtered_out(self):
        # The capture sink is duration-scoped, so a re-entrant on-thread lowering can append
        # ITS wrappers (a different origin_id) during the window. The composer filters to the
        # target origin (the last orchestration's), so a foreign-origin wrapper must be
        # dropped from the emitted source while the target orchestration still composes.
        def foreign_fn(args):
            return args

        foreign = GeneratedSource(
            "dedup_wrapper",
            "foreign_inner",
            "def foreign_inner(args):\n    return compiled_fn(args)\n",
            {"compiled_fn": lambda a: a},
            foreign_fn,
            origin_id=1,
        )
        src = _compose_standalone_module(
            "def call(args):\n    return args\n",
            [foreign, self._orch(origin_id=2)],
            _SENTINEL_INNER_CALL,
        )
        self.assertNotIn("foreign_inner", src)
        self.assertIn("_runtime_wrapper", src)

    def test_inner_call_guard_rejects_missing_call(self):
        # The module splices ``_inner_call = call``, so the inner Inductor source MUST bind a
        # module-level ``call``. An inner module that binds only ``not_call`` would surface as
        # a bare NameError at exec; the guard turns that into a clear contract error.
        with self.assertRaisesRegex(
            NotImplementedError, "does not bind a module-level 'call'"
        ):
            _compose_standalone_module(
                "def not_call(args):\n    return args\n",
                [self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_inner_call_guard_accepts_runner_assign(self):
        # The other inductor codegen form binds the entry point as ``call = runner.call``
        # (the graph_partition Runner path) rather than ``def call``. The guard must accept
        # that Assign-with-Name-target form too, so this composes without raising.
        runner_inner = """\
class _R:
    def call(self, args):
        return args
runner = _R()
call = runner.call
"""
        src = _compose_standalone_module(
            runner_inner, [self._orch()], _SENTINEL_INNER_CALL
        )
        # ``_inner_call = call`` is emitted for any successful compose, so it only proves the
        # guard did not raise; assert the runner-specific binding survived into the source to
        # pin that the Assign form (not just some ``call``) was the accepted one, then exec to
        # confirm the spliced ``_inner_call = call`` actually resolves at runtime.
        self.assertIn("call = runner.call", src)
        self.assertEqual(_exec(src)([7]), [7])


if __name__ == "__main__":
    run_tests()
