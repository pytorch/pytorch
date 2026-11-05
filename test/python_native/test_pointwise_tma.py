# Owner(s): ["module: dsl-native-ops"]

import unittest
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_cuda import IS_SM10X
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfNoNativeAot,
    subtest,
    TestCase,
)


@contextmanager
def _execution_route(route: str) -> Iterator[None]:
    from torch import _native

    controller = torch.backends.python_native.cutedsl
    jit_enabled, aot_enabled = controller.enabled, _native.aot_enabled()
    try:
        if route == "aot":
            controller.disable()
        else:
            _native.set_aot_enabled(False)
        yield
    finally:
        _native.set_aot_enabled(aot_enabled)
        if jit_enabled:
            controller.enable()


@unittest.skipUnless(IS_SM10X, "pointwise kernels require compute capability 10.x")
class TestPointwiseTma(TestCase):
    hw_classification = HardwareClassification.CUDA

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float32)
    def test_pow_float32(self, device: str, dtype: torch.dtype) -> None:
        from torch._native.ops.pointwise import kernel

        values = [-3, -2, -1, -0.5, -0.0, 0.0, 0.5, 1, 2, 3]
        values += [1.0001, 0.9999, 1e-20, float("-inf"), float("inf"), float("nan")]
        values = torch.tensor(values, device=device, dtype=dtype).repeat(2)
        x, y = values.repeat(32, 1).t(), values.repeat(32, 1)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.pow(x, y)
        with (
            patch.object(kernel, "_ENABLED", True),
            patch.object(kernel, "run", wraps=kernel.run) as launch,
        ):
            actual = torch.pow(x, y)
        launch.assert_called()
        self.assertEqual(actual, expected, atol=1e-6, rtol=1e-6)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize(
        "layout",
        ("mixed", "transposed", "rank3", "batched", "padded", "strided", "sliced"),
    )
    @parametrize(
        "route", ("jit", subtest("aot", decorators=[skipIfNoNativeAot("add")]))
    )
    def test_tma_pointwise(
        self, device: str, dtype: torch.dtype, layout: str, route: str
    ) -> None:
        from torch._native.ops.pointwise import kernel

        if layout == "rank3":
            x = torch.randn(32, 32, 64, device=device, dtype=dtype).permute(2, 0, 1)
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout == "batched":
            x = torch.randn(2, 32, 32, 64, device=device, dtype=dtype).permute(
                0, 3, 1, 2
            )
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout == "padded":
            x = torch.randn(1024, 4, 64, device=device, dtype=dtype)[..., :32]
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout == "strided":
            x = torch.randn(256, 128, device=device, dtype=dtype)[::2, :]
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout == "sliced":
            x = torch.randn(160, 160, device=device, dtype=dtype)[32:, :128].t()
            y = torch.randn(x.shape, device=device, dtype=dtype)
        else:
            x = torch.randn(256, 128, device=device, dtype=dtype).t()
            y = torch.randn(x.shape, device=device, dtype=dtype)
            if layout == "transposed":
                y = torch.randn(256, 128, device=device, dtype=dtype).t()
        for alpha in (0.5, 1.5):
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.add(x, y, alpha=alpha)
            profiler = profile(activities=[ProfilerActivity.CUDA])
            with (
                _execution_route(route),
                profiler if route == "aot" else nullcontext(),
                patch.object(kernel, "_ENABLED", True),
                patch.object(kernel, "run", wraps=kernel.run) as launch,
            ):
                if alpha == 0.5:
                    actual = torch.add(x, y, alpha=alpha)
                else:
                    actual = torch.empty(x.numel() + 32, device=device, dtype=dtype)[
                        32:
                    ].view(x.shape)
                    torch.add(x, y, alpha=alpha, out=actual)
            if route == "jit":
                launch.assert_called()
            else:
                launch.assert_not_called()
                self.assertEqual(
                    any(
                        "ElementwiseTma2D" in event.name for event in profiler.events()
                    ),
                    x.ndim == 2,
                )
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    @parametrize(
        "route", ("jit", subtest("aot", decorators=[skipIfNoNativeAot("gt.Tensor")]))
    )
    def test_tma_bool_output_and_capture(
        self, device: str, dtype: torch.dtype, route: str
    ) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(256, 128, device=device, dtype=dtype).t()
        y = torch.randn(x.shape, device=device, dtype=dtype)
        x[0, :3] = torch.tensor(
            (float("nan"), float("inf"), -float("inf")), device=device, dtype=dtype
        )
        out = torch.empty(x.shape, device=device, dtype=torch.bool)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.gt(x, y)
        profiler = profile(activities=[ProfilerActivity.CUDA])
        with (
            _execution_route(route),
            profiler if route == "aot" else nullcontext(),
            patch.object(kernel, "_ENABLED", True),
            patch.object(kernel, "run", wraps=kernel.run) as launch,
        ):
            # Exercise Boolean output during graph capture; warm up compilation
            # and launch state before starting capture.
            torch.gt(x, y, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.gt(x, y, out=out)
            graph.replay()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(out, expected)
        if route == "jit":
            launch.assert_called()
        else:
            launch.assert_not_called()
            self.assertTrue(
                any("ElementwiseTma2D" in event.name for event in profiler.events())
            )

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize(
        "layout",
        (
            "contiguous",
            "inner_stride",
            "broadcast",
            "scalar",
            "tail",
            "alias",
            "misaligned",
            "float64",
        ),
    )
    @parametrize(
        "route", ("jit", subtest("aot", decorators=[skipIfNoNativeAot("add")]))
    )
    def test_aten_fallback(self, device: str, layout: str, route: str) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(64, 65 if layout == "tail" else 64, device=device)
        if layout == "inner_stride":
            x = x[:, ::2]
        elif layout == "misaligned":
            x = torch.randn(4097, device=device)[1:].view(64, 64).t()
        elif layout == "float64":
            x = x.double().t()
        elif layout in ("broadcast", "scalar", "tail", "alias"):
            x = x.t()
        shape = (x.shape[-1],) if layout == "broadcast" else x.shape
        y = (
            0.5
            if layout == "scalar"
            else torch.randn(shape, device=device, dtype=x.dtype)
        )
        with torch.backends.python_native.cutedsl.disabled():
            expected = x + y
        if layout == "alias":
            # Warm an identical layout with independent output storage first.
            with _execution_route(route):
                torch.add(x, y, out=torch.empty_like(y))
        profiler = profile(activities=[ProfilerActivity.CUDA])
        with (
            _execution_route(route),
            profiler if route == "aot" else nullcontext(),
            patch.object(kernel, "run", wraps=kernel.run) as launch,
        ):
            actual = torch.add(x, y, out=y) if layout == "alias" else x + y
        launch.assert_not_called()
        if route == "aot":
            self.assertFalse(
                any("ElementwiseTma2D" in event.name for event in profiler.events())
            )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestPointwiseTma, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
