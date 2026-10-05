# Owner(s): ["module: dsl-native-ops"]

import contextlib
from collections.abc import Iterator
from unittest import mock

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfTorchDynamo,
    TestCase,
)


@contextlib.contextmanager
def jit_only() -> Iterator[None]:
    with torch.backends.python_native.cutedsl.disabled():
        torch.backends.python_native.cutedsl.enable()
        try:
            yield
        finally:
            torch.backends.python_native.cutedsl.disable()


_TOLERANCES = {torch.float16: 3e-3, torch.bfloat16: 3e-2, torch.float32: 2e-4}


@skipIfTorchDynamo("host-side capability predicates need no dynamo compilation")
@instantiate_parametrized_tests
class TestRmsNormCapability(TestCase):
    @parametrize("kind", ["meta", "fake", "functional_fake"])
    def test_traced_inputs_decline(self, kind: str) -> None:
        from torch._native.ops.norm.rmsnorm_impl import _is_supported
        from torch._subclasses.fake_tensor import FakeTensorMode

        if kind == "meta":
            x = torch.empty(4, 128, device="meta")
        else:
            with FakeTensorMode():
                x = torch.empty(4, 128, device="cuda")
            if kind == "functional_fake":
                x = torch._to_functional_tensor(x)
        with mock.patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("queried hardware for a traced tensor"),
        ):
            self.assertFalse(_is_supported(x))


@skipIfNoCuteDSL
class TestRmsNormJit(TestCase):
    def setUp(self) -> None:
        super().setUp()
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("RMSNorm kernels require SM9x, SM10x, or SM12x")

    def test_compile_cache_operator_attribution(self, device: str) -> None:
        from torch._native.ops.norm.rmsnorm_kernels import compile_rmsnorm

        compile_rmsnorm.cache_clear()
        compiled = []
        with (
            mock.patch("torch._native.instrumentation._listening", return_value=True),
            mock.patch("torch._native.instrumentation._emit") as emit,
        ):
            for direction in ("forward", "backward"):
                args = (
                    direction,
                    torch.float32,
                    128,
                    False,
                    torch.cuda.get_device_capability(device),
                )
                first = compile_rmsnorm(*args)
                compiled.append(first)
                self.assertIs(first, compile_rmsnorm(*args))
        self.assertIsNot(*compiled)
        events = [call.args[0] for call in emit.call_args_list]
        self.assertEqual(
            [event.op for event in events],
            ["aten::_fused_rms_norm"] * 2 + ["aten::_fused_rms_norm_backward"] * 2,
        )
        self.assertFalse(events[1].compiled or events[3].compiled)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("n", [128, 512, 1024, 2048, 4096, 8192])
    @parametrize("m", [1, 513])
    @parametrize("has_weight", [False, True])
    def test_shared_kernels(
        self, device: str, dtype: torch.dtype, n: int, m: int, has_weight: bool
    ) -> None:
        x = torch.randn(m, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype) if has_weight else None
        dy = torch.randn_like(x)

        def run() -> tuple[
            torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]
        ]:
            y, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            grads = torch.ops.aten._fused_rms_norm_backward(
                dy, x, [n], rstd, w, [True, has_weight]
            )
            return y, rstd, grads

        with torch.backends.python_native.cutedsl.disabled():
            expected = run()
        with jit_only():
            actual = run()
        tol = _TOLERANCES[dtype]
        self.assertEqual(actual, expected, atol=tol, rtol=tol)

    @parametrize(
        "m,n,has_weight,dtype",
        [
            (m, n, has_weight, dtype)
            for m, n in [
                (1, 2),
                (1, 30),
                (33, 30),
                (513, 34),
                (1, 130),
                (33, 384),
                (257, 768),
                (64, 12288),
                (8, 65536),
                (8, 131072),
                (8, 262144),
            ]
            for has_weight in (False, True)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
            # Sample optional weights and bf16 at alignment and width boundaries.
            if (has_weight or (m, n) in ((1, 2), (33, 30), (257, 768), (8, 131072)))
            and (
                dtype != torch.bfloat16
                or (m, n) in ((1, 30), (33, 30), (8, 131072), (8, 262144))
            )
        ],
        name_fn=lambda m, n, has_weight, _dtype: f"m_{m}_n_{n}_has_weight_{has_weight}",
    )
    def test_general_widths(
        self, device: str, dtype: torch.dtype, m: int, n: int, has_weight: bool
    ) -> None:
        from torch._native.ops.norm.rmsnorm_impl import _fused_rms_norm_backward_cond
        from torch.profiler import profile, ProfilerActivity

        x = torch.randn(m, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype) if has_weight else None
        dy = torch.randn_like(x)
        with torch.backends.python_native.cutedsl.disabled():
            y, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            args = (dy, x, [n], rstd, w, [True, has_weight])
            if not _fused_rms_norm_backward_cond(*args):
                self.skipTest("row exceeds the device's backward kernel limits")
            grads = torch.ops.aten._fused_rms_norm_backward(*args)
        with jit_only():
            torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            torch.ops.aten._fused_rms_norm_backward(*args)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                actual_y, actual_rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
                actual_grads = torch.ops.aten._fused_rms_norm_backward(
                    dy, x, [n], actual_rstd, w, [True, has_weight]
                )
        names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        self.assertTrue(any("RMSNormBackward" in name for name in names))
        self.assertEqual(any("reduce_rows" in name for name in names), has_weight)
        tol = _TOLERANCES[dtype]
        self.assertEqual(
            (actual_y, actual_rstd, actual_grads),
            (y, rstd, grads),
            atol=tol,
            rtol=tol,
        )

    @parametrize(
        "dtype,grad_dtype,n,compute_dw",
        [
            (dtype, dtype, n, False)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
            for n in (30, 1024)
        ]
        + [
            (dtype, grad_dtype, n, compute_dw)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
            for grad_dtype in (torch.float16, torch.bfloat16, torch.float32)
            if dtype != grad_dtype
            for n, compute_dw in ((30, False), (30, True), (1024, True))
        ],
        name_fn=lambda _dtype, grad_dtype, n, compute_dw: (
            f"{str(grad_dtype).removeprefix('torch.')}_n_{n}_compute_dw_{compute_dw}"
        ),
    )
    def test_gradient_dtype(
        self,
        device: str,
        dtype: torch.dtype,
        grad_dtype: torch.dtype,
        n: int,
        compute_dw: bool,
    ) -> None:
        from torch._native.ops.norm.norms import quack_rmsnorm_bwd, quack_rmsnorm_fwd

        x = torch.randn(33, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype)
        dy = torch.randn(x.shape, device=device, dtype=grad_dtype)
        _, rstd = quack_rmsnorm_fwd(x, w, [n], 1e-5)
        actual_dx, actual_dw = quack_rmsnorm_bwd(dy, x, rstd, w, [n], compute_dw)
        xf = x.float().requires_grad_()
        wf = w.float().requires_grad_()
        y = xf * (xf.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * wf
        dx, dw = torch.autograd.grad(y, (xf, wf), dy.float())
        tol = _TOLERANCES[dtype]
        self.assertEqual(actual_dx, dx.to(dtype), atol=tol, rtol=tol)
        self.assertEqual(
            actual_dw, dw.to(dtype) if compute_dw else None, atol=tol, rtol=tol
        )

    @dtypes(torch.float32)
    @parametrize("m", [1, 33])
    @parametrize("negative", [False, True])
    def test_odd_width_views(
        self, device: str, dtype: torch.dtype, m: int, negative: bool
    ) -> None:
        from torch._native.ops.norm.norms import quack_rmsnorm_bwd, quack_rmsnorm_fwd

        n = 31
        x = torch.randn(n, m, device=device, dtype=dtype).t()
        w = torch.randn(n + 1, device=device, dtype=dtype)[1:]
        dy = torch.randn(n, m, device=device, dtype=dtype).t()
        if negative:
            x, w, dy = x._neg_view(), w._neg_view(), dy._neg_view()
        with torch.backends.python_native.cutedsl.disabled():
            y, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            grads = torch.ops.aten._fused_rms_norm_backward(
                dy, x, [n], rstd, w, [True, True]
            )
        actual_y, actual_rstd = quack_rmsnorm_fwd(x, w, [n], 1e-5)
        stats = torch.empty(m, 2, device=device, dtype=torch.float32)[:, 1:]
        stats.copy_(actual_rstd)
        actual_grads = quack_rmsnorm_bwd(dy, x, stats, w, [n])
        self.assertEqual(
            (actual_y, actual_rstd, actual_grads),
            (y, rstd, grads),
            atol=2e-4,
            rtol=2e-4,
        )

    @dtypes(torch.float32)
    @parametrize("n", [30, 1024])
    def test_autograd_and_cuda_graph(
        self, device: str, dtype: torch.dtype, n: int
    ) -> None:
        x = torch.randn(37, n, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)

        def run() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            y = torch.nn.functional.rms_norm(x, [n], w, 1e-5)
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

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("m", [4096, 16384])
    @parametrize("n", [128, 1024, 8192])
    def test_persistent_weight_gradient(
        self, device: str, dtype: torch.dtype, m: int, n: int
    ) -> None:
        x = torch.randn(m, n, device=device, dtype=dtype)
        dy = torch.randn_like(x)
        w = torch.randn(n, device=device, dtype=dtype)
        with torch.backends.python_native.cutedsl.disabled():
            _, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            expected = torch.ops.aten._fused_rms_norm_backward(
                dy, x, [n], rstd, w, [True, True]
            )
        with jit_only():
            actual = torch.ops.aten._fused_rms_norm_backward(
                dy, x, [n], rstd, w, [True, True]
            )
        tol = _TOLERANCES[dtype]
        self.assertEqual(actual, expected, atol=tol, rtol=tol)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize(
        "n,m,native_low_precision,native_fp32",
        [
            (128, 8191, False, False),
            (128, 8192, True, True),
            (512, 8192, True, True),
            (1024, 8192, True, False),
            (1024, 16384, True, True),
            (2048, 16383, False, False),
            (2048, 16384, True, True),
            (4096, 16384, False, False),
            (8192, 16384, False, False),
        ],
    )
    def test_weight_only_dispatch(
        self,
        device: str,
        dtype: torch.dtype,
        n: int,
        m: int,
        native_low_precision: bool,
        native_fp32: bool,
    ) -> None:
        from torch.profiler import profile, ProfilerActivity

        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
            self.skipTest("weight-only crossover is tuned for SM90 and SM100")
        x = torch.randn(m, n, device=device, dtype=dtype)
        w = torch.randn(n, device=device, dtype=dtype)
        dy = torch.randn_like(x)
        with torch.backends.python_native.cutedsl.disabled():
            _, rstd = torch.ops.aten._fused_rms_norm(x, [n], w, 1e-5)
            args = (dy, x, [n], rstd, w, [False, True])
            expected = torch.ops.aten._fused_rms_norm_backward(*args)
        native = native_fp32 if dtype == torch.float32 else native_low_precision
        with jit_only():
            torch.ops.aten._fused_rms_norm_backward(*args)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                actual = torch.ops.aten._fused_rms_norm_backward(*args)
        names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        self.assertEqual(any("RMSNormBackward" in name for name in names), native)
        tol = _TOLERANCES[dtype]
        self.assertEqual(actual, expected, atol=tol, rtol=tol)


instantiate_device_type_tests(TestRmsNormJit, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
