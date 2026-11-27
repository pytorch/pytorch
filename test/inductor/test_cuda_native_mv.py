# Owner(s): ["module: inductor"]
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._inductor import config, ir
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


class NoNativeCopy(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.clone.default, torch.ops.aten.copy_.default):
            raise AssertionError(
                "Runtime GEMV must use generated copy kernels: " + str(func)
            )
        if func == torch.ops.aten.resolve_conj.default and args[0].is_conj():
            raise AssertionError(
                "Runtime GEMV must materialize conjugation in a generated kernel"
            )
        if func == torch.ops.aten.resolve_neg.default and args[0].is_neg():
            raise AssertionError(
                "Runtime GEMV must materialize negation in a generated kernel"
            )
        return func(*args, **kwargs or {})


class TestCudaNativeMv(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize(
        "layout", ("row", "column", "strided", "broadcast", "negative", "conjugate")
    )
    def test_aoti_dynamic_geometry(self, device, dtype, layout):
        class Model(torch.nn.Module):
            def forward(self, x, vector):
                return torch.mv(x, vector)

        def inputs(m, k):
            x = torch.randn(m, k, device=device, dtype=dtype)
            vector = torch.randn(k, device=device, dtype=dtype)
            if layout == "column":
                x = x.t().contiguous().t()
            elif layout == "strided":
                x = torch.randn(2 * m, 2 * k, device=device, dtype=dtype)[::2, ::2]
                vector = torch.randn(2 * k, device=device, dtype=dtype)[::2]
            elif layout == "broadcast":
                vector = vector[:1].expand(k)
            elif layout == "negative":
                x = torch._neg_view(x)
                vector = torch._neg_view(vector)
            elif layout == "conjugate":
                x = x.t().contiguous().t().conj()
                vector = vector.conj()
            return x, vector

        torch.manual_seed(0)
        m = torch.export.Dim("m", min=2, max=32)
        k = torch.export.Dim("k", min=2, max=4096)
        exported = torch.export.export(
            Model(), inputs(16, 257), dynamic_shapes=({0: m, 1: k}, {0: k})
        )
        native_ops = set()
        original = ir.FallbackKernel.codegen

        def record_native(kernel, wrapper):
            native_ops.add(str(kernel.op_overload))
            return original(kernel, wrapper)

        with (
            TemporaryDirectory() as directory,
            mock.patch.object(ir.FallbackKernel, "codegen", record_native),
        ):
            package = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(directory) / "gemv.pt2"),
                inductor_configs={"numerics": "strict"},
            )
            loaded = torch._inductor.aoti_load_package(package)
            for rows, columns in ((8, 8), (16, 257), (31, 2048)):
                x, vector = inputs(rows, columns)
                expected = torch.mv(x, vector)
                with NoNativeCopy():
                    actual = loaded(x, vector)
                self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))
        self.assertNotIn("aten.mv.default", native_ops)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("axis", ("rows", "columns"))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_aoti_unbacked_geometry(self, device, dtype, axis):
        class Model(torch.nn.Module):
            def forward(self, x, mask, vector):
                if axis == "rows":
                    return torch.mv(x[mask], vector)
                return torch.mv(x[:, mask], vector[mask])

        torch.manual_seed(0)
        x = torch.randn(16, 257, device=device, dtype=dtype)
        vector = torch.randn(257, device=device, dtype=dtype)
        size = x.size(0) if axis == "rows" else vector.numel()
        mask = torch.arange(size, device=device) < size // 2
        model = Model()
        exported = torch.export.export(model, (x, mask, vector))
        with TemporaryDirectory() as directory:
            package = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(directory) / "gemv.pt2"),
                inductor_configs={"numerics": "strict"},
            )
            loaded = torch._inductor.aoti_load_package(package)
            for count in (0, 1, size // 2, size):
                mask = torch.arange(size, device=device) < count
                expected = model(x, mask, vector)
                with NoNativeCopy():
                    actual = loaded(x, mask, vector)
                self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("axis", ("rows", "columns"))
    @parametrize("cpp_wrapper", (False, True))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unbacked_backward(self, device, dtype, axis, cpp_wrapper):
        torch.manual_seed(0)
        x = torch.randn((16, 257), device=device, dtype=dtype, requires_grad=True)
        vector = torch.randn(257, device=device, dtype=dtype, requires_grad=True)

        def fn(x, mask, vector):
            if axis == "rows":
                return torch.mv(x[mask], vector)
            return torch.mv(x[:, mask], vector[mask])

        counter = CompileCounterWithBackend("inductor")
        with config.patch(
            numerics="strict", fx_graph_cache=False, cpp_wrapper=cpp_wrapper
        ):
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            size = x.size(0) if axis == "rows" else vector.numel()
            for count in (size // 2, 1, 0, size):
                mask = torch.arange(size, device=device) < count
                expected = fn(x, mask, vector)
                actual = compiled(x, mask, vector)
                grad = torch.randn_like(expected)
                expected_grads = torch.autograd.grad(expected, (x, vector), grad)
                actual_grads = torch.autograd.grad(actual, (x, vector), grad)
                self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    self.assertEqual(
                        actual_grad.view(torch.uint8), expected_grad.view(torch.uint8)
                    )
        self.assertEqual(counter.frame_count, 1)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("columns", (8, 257, 2048))
    @parametrize("axis", ("rows", "columns"))
    @parametrize("cpp_wrapper", (False, True))
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unbacked_geometry(self, device, dtype, columns, axis, cpp_wrapper):
        torch.manual_seed(0)
        x = torch.randn((64, columns), device=device, dtype=dtype)
        vector = torch.randn(columns, device=device, dtype=dtype)

        def fn(x, mask, vector):
            if axis == "rows":
                return torch.mv(x[mask], vector)
            return torch.mv(x[:, mask], vector[mask])

        counter = CompileCounterWithBackend("inductor")
        native_ops = set()
        original = ir.FallbackKernel.codegen

        def record_native(kernel, wrapper):
            native_ops.add(str(kernel.op_overload))
            return original(kernel, wrapper)

        with (
            config.patch(
                numerics="strict", fx_graph_cache=False, cpp_wrapper=cpp_wrapper
            ),
            mock.patch.object(ir.FallbackKernel, "codegen", record_native),
        ):
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            size = 64 if axis == "rows" else columns
            for count in (size // 2, 1, 0, size):
                mask = torch.arange(size, device=device) < count
                expected = fn(x, mask, vector)
                actual = compiled(x, mask, vector)
                self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))
        self.assertEqual(counter.frame_count, 1)
        self.assertNotIn("aten.mv.default", native_ops)

    @dtypes(
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    @parametrize("kind", ("matrix", "vector", "negative", "conjugate"))
    def test_generated_normalization(self, device, dtype, kind):
        from torch._inductor.runtime.cuda_native_mv import cuda_native_mv

        torch.manual_seed(0)
        x = torch.randn((16, 8), device=device, dtype=dtype)
        v = torch.randn(8, device=device, dtype=dtype)
        if kind == "matrix":
            x = torch.randn((32, 16), device=device, dtype=dtype)[::2, ::2]
        elif kind == "vector":
            v = v[:1].expand(8)
        elif kind == "negative":
            x = torch._neg_view(x)
        elif kind == "conjugate":
            x = x.t().contiguous().t().conj()
        expected = torch.mv(x, v)
        out = torch.empty_like(expected)
        with NoNativeCopy():
            cuda_native_mv(x, v, out=out)
        self.assertEqual(out.view(torch.uint8), expected.view(torch.uint8))


instantiate_device_type_tests(TestCudaNativeMv, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
