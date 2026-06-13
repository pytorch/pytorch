# Owner(s): ["module: inductor"]
import copy
import os
import tempfile
import unittest

import torch
from torch._non_strict_compile import NonStrictCompileError
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNonStrictCompile(TestCase):
    def test_plain_function(self):
        def f(x, y):
            return (x @ y).sin(), x + y

        fc = torch.non_strict_compile(f)
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        ref = f(a, b)
        out = fc.example(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])
        # Re-invocation after JIT compile should also work.
        out2 = fc(a, b)
        self.assertEqual(out2[0], ref[0])

    def test_module_params_and_buffers_are_lifted(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        ref = m(x)

        fc = torch.non_strict_compile(m)
        out = fc.example(x)
        self.assertEqual(out, ref)

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)

        def f(x):
            return x + captured

        fc = torch.non_strict_compile(f)
        with self.assertRaisesRegex(NonStrictCompileError, "hard-coded"):
            fc.example(torch.randn(3))

    def test_global_tensor_rejected_unlike_make_fx(self):
        # Vanilla make_fx silently bakes a referenced global tensor into the
        # GraphModule as a get_attr constant; non_strict_compile must instead
        # error, so such a tensor is never hard-coded into the graph.
        from torch.fx.experimental.proxy_tensor import make_fx

        global_tensor = torch.randn(4)

        def f(x):
            return x + global_tensor

        # Document the vanilla behavior we are guarding against: a get_attr node
        # whose attribute is a Tensor (the baked-in constant).
        gm = make_fx(f)(torch.randn(4))
        baked = [
            n.target
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(gm, n.target, None), torch.Tensor)
        ]
        self.assertTrue(baked, "expected vanilla make_fx to bake a tensor constant")

        fc = torch.non_strict_compile(f)
        with self.assertRaisesRegex(NonStrictCompileError, "hard-coded"):
            fc.example(torch.randn(4))

    def test_unregistered_module_tensor_attr_is_rejected(self):
        # A plain tensor attribute (not a registered parameter/buffer) is not
        # lifted, so referencing it would bake it in -- this must error.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.scale = torch.randn(4)  # plain attr, NOT a buffer/parameter

            def forward(self, x):
                return (x @ self.weight) * self.scale

        fc = torch.non_strict_compile(M().eval())
        with self.assertRaisesRegex(NonStrictCompileError, "hard-coded"):
            fc.example(torch.randn(2, 4))

    def test_export_python_and_cache_roundtrip(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        ref = m(x)

        fc = torch.non_strict_compile(m)
        fc.example(x)

        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "foo.py")
            cache = os.path.join(d, "cache.bin")
            fc.export_python(py)
            fc.export_cache(cache)

            with open(py) as fh:
                src = fh.read()
            self.assertIn("Calling-convention metadata", src)
            self.assertIn("Inductor output code", src)
            self.assertIn("def forward(", src)
            self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", src)

            loaded = torch.non_strict_compile.load(py, cache)
            out = loaded(x)
            self.assertEqual(out, ref)

    def test_cache_primes_inductor_on_reload(self):
        # The cache stores the real compiled artifact; reloading in a fresh
        # inductor cache dir primes it and hits FxGraphCache (no re-lowering),
        # which is the kernel caching the cache provides.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        m = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
        ).eval()
        x = torch.randn(3, 8)
        ref = m(x)

        fc = torch.non_strict_compile(m)
        fc.example(x)

        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "foo.py")
            cache = os.path.join(d, "cache.bin")
            fc.export_cache(cache)
            fc.export_python(py)

            with fresh_cache():
                counters.clear()
                loaded = torch.non_strict_compile.load(py, cache)
                out = loaded(x)
                self.assertEqual(out, ref)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for Triton autotuning")
    def test_cache_bundles_autotune_artifacts(self):
        # On GPU the cache bundle includes Triton autotuning artifacts, and a
        # reload in a fresh inductor cache dir restores them (so kernels are not
        # re-autotuned).
        from torch._inductor.utils import fresh_cache

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(512, 512)
                self.l2 = torch.nn.Linear(512, 512)

            def forward(self, x):
                return torch.softmax(self.l2(torch.relu(self.l1(x))), dim=-1)

        m = M().cuda().eval()
        x = torch.randn(128, 512, device="cuda")
        ref = m(x)

        fc = torch.non_strict_compile(m)
        fc.example(x)

        # The compiled artifact's cache bundle includes the autotune category.
        _, cache_info = fc._artifact._artifacts
        self.assertIn("autotune", cache_info.artifacts)

        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "foo.py")
            cache = os.path.join(d, "cache.bin")
            fc.export_cache(cache)
            fc.export_python(py)

            with fresh_cache():
                loaded = torch.non_strict_compile.load(py, cache)
                self.assertEqual(loaded(x), ref)

    def test_exported_python_is_self_contained(self):
        # The exported .py must be runnable on its own via exec: it defines a
        # `forward` that loads weights from the companion cache file.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(self.lin(x))

        m = M().eval()
        x = torch.randn(5, 4)
        ref = m(x)

        fc = torch.non_strict_compile(m)
        fc.example(x)

        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "foo.py")
            cache = os.path.join(d, "foo_cache.bin")
            # export_cache first so the baked WEIGHTS_PATH points at it.
            fc.export_cache(cache)
            fc.export_python(py)

            with open(py) as fh:
                code = fh.read()
            ns = {"__name__": "_artifact"}
            exec(compile(code, py, "exec"), ns)
            out = ns["forward"](x)
            self.assertEqual(out, ref)

    def test_dtensor_subclass_jit_and_load(self):
        # Tensor subclasses (DTensor) work through the JIT path, through direct
        # exec of the exported python, and through load(). The exported python
        # recomposes AOTAutograd's actual codegen'd subclass unwrap/rewrap and
        # prologue/epilogue wrappers, so it handles DTensor without the cache's
        # full artifact.
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = torch.nn.Linear(4, 3)

                def forward(self, x):
                    return torch.relu(self.lin(x))

            m = M().eval()
            for name, p in list(m.named_parameters()):
                mod = m
                *path, leaf = name.split(".")
                for part in path:
                    mod = getattr(mod, part)
                dt = distribute_tensor(p.detach(), mesh, [Replicate()])
                setattr(mod, leaf, torch.nn.Parameter(dt))

            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            fc = torch.non_strict_compile(m)
            jit_out = fc.example(x)
            self.assertEqual(jit_out.to_local(), ref.to_local())
            names = [r["name"] for r in fc._wrapper_records]
            self.assertIn("subclass_wrapper", names)

            with tempfile.TemporaryDirectory() as d:
                py = os.path.join(d, "foo.py")
                cache = os.path.join(d, "cache.bin")
                fc.export_cache(cache)
                fc.export_python(py)

                # Direct exec of the generated python (no load()).
                with open(py) as fh:
                    code = fh.read()
                ns = {"__name__": "_artifact"}
                exec(compile(code, py, "exec"), ns)
                direct = ns["forward"](x)
                self.assertEqual(direct.to_local(), ref.to_local())

                # Public load() API.
                loaded = torch.non_strict_compile.load(py, cache)
                out = loaded(x)
                self.assertEqual(out.to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()

    def test_custom_step_grads_match_eager(self):
        # A generic step lambda (forward + loss + autograd.grad) traces into one
        # graph; its returned grads match eager autograd exactly.
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(5, 4)
        target = torch.randn(5, 3)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def step(model, x, target):
            loss = loss_fn(model(x), target)
            grads = torch.autograd.grad(loss, list(model.parameters()))
            return loss, grads

        f_c = torch.non_strict_compile(model, step=step)
        loss, grads = f_c(x, target)
        # No autograd.Function is generated; the backward is plain graph ops.
        self.assertEqual(
            [r["name"] for r in f_c._wrapper_records],
            ["runtime_wrapper_orchestration"],
        )
        self.assertFalse(loss.requires_grad)  # loss is a graph output, not a leaf
        for g, rg in zip(grads, ref_grads):
            self.assertEqual(g, rg)

        # A multi-step loop (apply returned grads via an optimizer) reduces loss.
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        for _ in range(5):
            loss, grads = f_c(x, target)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_custom_step_export_is_self_contained(self):
        # A custom step lowers like inference, so the exported python inlines the
        # (forward+loss+backward) Inductor call and is self-contained; reload
        # keeps training.
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 3))
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(5, 4)
        target = torch.randn(5, 3)

        def step(model, x, target):
            loss = loss_fn(model(x), target)
            return loss, torch.autograd.grad(loss, list(model.parameters()))

        f_c = torch.non_strict_compile(model, step=step)
        f_c(x, target)

        with tempfile.TemporaryDirectory() as d:
            py = os.path.join(d, "foo.py")
            cache = os.path.join(d, "cache.bin")
            f_c.export_cache(cache)
            f_c.export_python(py)

            with open(py) as fh:
                src = fh.read()
            self.assertIn("call = runner.call", src)  # inlined Inductor call

            g = torch.non_strict_compile.load(py, cache)
            opt = torch.optim.SGD(g.parameters(), lr=0.1)
            losses = []
            for _ in range(3):
                loss, grads = g(x, target)
                for p, gr in zip(g.parameters(), grads):
                    p.grad = gr
                opt.step()
                losses.append(loss.item())
            self.assertLess(losses[-1], losses[0])

    def test_step_requires_module(self):
        with self.assertRaises(NonStrictCompileError):
            torch.non_strict_compile(lambda x: x, step=lambda m, x: x)

    def test_export_before_compile_errors(self):
        fc = torch.non_strict_compile(lambda x: x + 1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NonStrictCompileError):
                fc.export_python(os.path.join(d, "foo.py"))
            with self.assertRaises(NonStrictCompileError):
                fc.export_cache(os.path.join(d, "cache.bin"))


if __name__ == "__main__":
    run_tests()
