# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest.mock import patch

import torch
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
    TestCase,
)


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
    def test_tma_pointwise(self, device: str, dtype: torch.dtype, layout: str) -> None:
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
            with (
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
            launch.assert_called()
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.bfloat16, torch.float32)
    def test_tma_bool_output_and_capture(self, device: str, dtype: torch.dtype) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(256, 128, device=device, dtype=dtype).t()
        y = torch.randn(x.shape, device=device, dtype=dtype)
        x[0, :3] = torch.tensor(
            (float("nan"), float("inf"), -float("inf")), device=device, dtype=dtype
        )
        out = torch.empty(x.shape, device=device, dtype=torch.bool)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.gt(x, y)
        with (
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
        launch.assert_called()

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize(
        "layout",
        ("contiguous", "inner_stride", "broadcast", "scalar", "misaligned", "float64"),
    )
    def test_aten_fallback(self, device: str, layout: str) -> None:
        from torch._native.ops.pointwise import kernel

        x = torch.randn(64, 64, device=device)
        if layout == "inner_stride":
            x = x[:, ::2]
        elif layout == "misaligned":
            x = torch.randn(4097, device=device)[1:].view(64, 64).t()
        elif layout == "float64":
            x = x.double().t()
        elif layout in ("broadcast", "scalar"):
            x = x.t()
        shape = (x.shape[-1],) if layout == "broadcast" else x.shape
        y = (
            0.5
            if layout == "scalar"
            else torch.randn(shape, device=device, dtype=x.dtype)
        )
        with torch.backends.python_native.cutedsl.disabled():
            expected = x + y
        with patch.object(kernel, "run", wraps=kernel.run) as launch:
            actual = x + y
        launch.assert_not_called()
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestPointwiseTma, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
