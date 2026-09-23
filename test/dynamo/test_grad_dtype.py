# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounterWithBackend
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCachePickler
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests


class _FP32Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad.float() * 1.00390625


def _projection(x):
    return _FP32Backward.apply(x)


class TestGradDtype(torch._dynamo.test_case.TestCase):
    @parametrize("backend", ["aot_eager", "inductor"])
    @parametrize("grad_dtype", [torch.bfloat16, torch.float32, None])
    def test_backward_precision(self, device, backend, grad_dtype):
        x = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        x.grad_dtype = grad_dtype
        fn = torch.compile(_projection, backend=backend, fullgraph=True)
        expected = torch.full_like(x, 1.00390625, dtype=torch.float32)
        if grad_dtype is not None:
            expected = expected.to(grad_dtype)
        for step in range(2):
            fn(x).sum().backward()
            self.assertEqual(x.grad.dtype, expected.dtype)
            self.assertEqual(x.grad, expected * (step + 1), atol=0, rtol=0)

    def test_recompile_on_grad_dtype_change(self, device):
        x = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        backend = CompileCounterWithBackend("aot_eager")
        fn = torch.compile(_projection, backend=backend, fullgraph=True)
        for grad_dtype in (torch.bfloat16, torch.float32, None):
            x.grad = None
            x.grad_dtype = grad_dtype
            fn(x).sum().backward()
            expected = torch.full_like(x, 1.00390625, dtype=torch.float32)
            if grad_dtype is not None:
                expected = expected.to(grad_dtype)
            self.assertEqual(x.grad, expected, atol=0, rtol=0)
        self.assertEqual(backend.frame_count, 3)

    def test_existing_gradient(self, device):
        x = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        x.grad_dtype = torch.float32
        x.grad = torch.ones_like(x, dtype=torch.float32)
        fn = torch.compile(_projection, backend="aot_eager", fullgraph=True)
        fn(x).sum().backward()
        self.assertEqual(x.grad, torch.full_like(x.grad, 2.00390625), atol=0, rtol=0)

    def test_leaf_to_nonleaf(self, device):
        fn = torch.compile(_projection, backend="aot_eager", fullgraph=True)
        x = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        x.grad_dtype = torch.float32
        fn(x).sum().backward()
        base = torch.ones_like(x, requires_grad=True)
        fn(base * 1).sum().backward()
        self.assertEqual(base.grad, torch.ones_like(base), atol=0, rtol=0)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_nonleaf_to_leaf(self, device, backend):
        counter = CompileCounterWithBackend(backend)
        fn = torch.compile(_projection, backend=counter, fullgraph=True)
        base = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        fn(base * 1).sum().backward()
        self.assertEqual(base.grad, torch.ones_like(base), atol=0, rtol=0)

        leaf = torch.ones_like(base, requires_grad=True)
        leaf.grad_dtype = torch.float32
        fn(leaf).sum().backward()
        self.assertEqual(leaf.grad.dtype, torch.float32)
        self.assertEqual(
            leaf.grad,
            torch.full_like(leaf, 1.00390625, dtype=torch.float32),
            atol=0,
            rtol=0,
        )
        self.assertEqual(counter.frame_count, 2)

    def test_aot_cache_key(self, device):
        x = torch.ones(32, device=device, dtype=torch.bfloat16, requires_grad=True)
        gm = torch.fx.symbolic_trace(lambda x: x.sin())
        pickler = AOTAutogradCachePickler(gm)
        keys = []
        for grad_dtype in (torch.bfloat16, torch.float32, None):
            x.grad_dtype = grad_dtype
            keys.append(pickler.dumps(x))
        self.assertEqual(len(set(keys)), 3)

    @onlyCUDA
    @parametrize("backend", ["aot_eager", "inductor"])
    def test_fp32_gemm(self, device, backend):
        class Projection(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h, w):
                ctx.save_for_backward(h, w)
                return h @ w.T

            @staticmethod
            def backward(ctx, dz):
                h, w = ctx.saved_tensors
                return dz @ w, torch.mm(dz.T.contiguous(), h, out_dtype=torch.float32)

        def projection(h, w):
            return Projection.apply(h, w)

        h = torch.full((3, 32), 1.0078125, device=device, dtype=torch.bfloat16)
        w = torch.full(
            (32, 32), 0.125, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        w.grad_dtype = torch.float32
        dz = torch.full_like(h, 1.0078125)
        expected = dz.float().T @ h.float()
        self.assertNotEqual(expected, expected.bfloat16().float())
        fn = torch.compile(projection, backend=backend, fullgraph=True)
        fn(h, w).backward(dz)
        self.assertEqual(w.grad.dtype, torch.float32)
        self.assertEqual(w.grad, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestGradDtype, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
