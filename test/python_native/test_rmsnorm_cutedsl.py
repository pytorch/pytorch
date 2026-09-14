# Owner(s): ["module: dsl-native-ops"]

import contextlib

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


@contextlib.contextmanager
def jit_only():
    with torch.backends.python_native.cutedsl.disabled():
        torch.backends.python_native.cutedsl.enable()
        try:
            yield
        finally:
            torch.backends.python_native.cutedsl.disable()


@skipIfNoCuteDSL
class TestRmsNormJit(TestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
            self.skipTest("shared RMSNorm kernels require Hopper or Blackwell")

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("n", [128, 512, 1024, 2048, 4096, 8192])
    @parametrize("m", [513])
    @parametrize("has_weight", [False, True])
    def test_shared_kernels(self, device, dtype, n, m, has_weight):
        x = torch.randn(m, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype) if has_weight else None
        dy = torch.randn_like(x)

        def run():
            y, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            grads = torch.ops.aten._fused_rms_norm_backward(
                dy, x, [n], rstd, w, [True, has_weight]
            )
            return y, rstd, grads

        with torch.backends.python_native.cutedsl.disabled():
            expected = run()
        with jit_only():
            actual = run()
        tol = {torch.float16: 3e-3, torch.bfloat16: 3e-2, torch.float32: 2e-4}[dtype]
        self.assertEqual(actual, expected, atol=tol, rtol=tol)

    @dtypes(torch.float32)
    def test_autograd_and_cuda_graph(self, device, dtype):
        x = torch.randn(37, 1024, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(1024, device=device, dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)

        def run():
            y = torch.nn.functional.rms_norm(x, [1024], w, 1e-5)
            return y, torch.autograd.grad(y, (x, w), dy)

        with torch.backends.python_native.cutedsl.disabled():
            expected = run()
        expected = (expected[0].detach(), expected[1])
        with jit_only():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            graph.replay()
        self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)


instantiate_device_type_tests(TestRmsNormJit, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
