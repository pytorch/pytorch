# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._native.cutedsl_utils import POINTWISE_ARCH_MAJORS
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


if TYPE_CHECKING:
    from collections.abc import Sequence


class TestPointwiseTma(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self) -> None:
        super().setUp()
        if torch.cuda.get_device_capability()[0] not in POINTWISE_ARCH_MAJORS:
            self.skipTest("TMA requires a supported CUDA architecture")

    def _run(
        self,
        inputs: Sequence[torch.Tensor],
        *,
        fn: str = "_add",
        dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        result = inputs[0].dtype if dtype is None else dtype
        compute = (
            torch.float32
            if result in (torch.bool, torch.float16, torch.bfloat16)
            else result
        )
        ct = torch2cute[compute]
        consts = (ct(1),) if fn == "_add" else ()
        (actual,) = kernel.run(
            ops.get_fn(fn),
            (fn, result, tuple((t.dtype, tuple(t.shape), t.stride()) for t in inputs)),
            len(inputs),
            1,
            consts,
            ct,
            inputs,
            (result,),
            load="tma",
            out=[out] if out is not None else None,
        )
        self.assertEqual(
            actual.shape, torch.broadcast_shapes(*(t.shape for t in inputs))
        )
        self.assertEqual(actual.dtype, result)
        if out is not None:
            self.assertIs(actual, out)
        return actual

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize(
        "layout",
        (
            "dense",
            "transpose",
            "both_transposed",
            "padded",
            "mixed_pitches",
            "batch",
            "grouped",
        ),
    )
    def test_arithmetic(self, device: str, dtype: torch.dtype, layout: str) -> None:
        if layout == "grouped":
            x = torch.randn(2, 64, 3, 32, device=device, dtype=dtype).permute(
                0, 2, 3, 1
            )
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout == "batch":
            x = torch.randn(2, 96, 64, device=device, dtype=dtype).transpose(1, 2)
            y = torch.randn(x.shape, device=device, dtype=dtype)
        elif layout in ("transpose", "both_transposed"):
            x = torch.randn(96, 64, device=device, dtype=dtype).t()
            y = (
                torch.randn(96, 64, device=device, dtype=dtype).t()
                if layout == "both_transposed"
                else torch.randn(x.shape, device=device, dtype=dtype)
            )
        elif layout in ("padded", "mixed_pitches"):
            x = torch.randn(64, 112, device=device, dtype=dtype)[:, :96]
            pitch = 128 if layout == "mixed_pitches" else 112
            y = torch.randn(64, pitch, device=device, dtype=dtype)[:, :96]
        else:
            x = torch.randn(64, 96, device=device, dtype=dtype)
            y = torch.randn_like(x)
        with torch.backends.python_native.cutedsl.disabled():
            expected = x + y
        actual = self._run([x, y])
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)
        if dtype is torch.float32 and layout == "transpose":
            actual = self._run([x, y], fn="_gt", dtype=torch.bool)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, x > y)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("zero_dim", (True, False))
    def test_cuda_scalar(self, device: str, dtype: torch.dtype, zero_dim: bool) -> None:
        x = torch.randn(96, 64, device=device, dtype=dtype).t()
        y = torch.tensor(1.003, device=device)
        if not zero_dim:
            y = y.reshape(1, 1)
        result = dtype if zero_dim else torch.float32
        with torch.backends.python_native.cutedsl.disabled():
            expected = x + y
        actual = self._run([x, y], dtype=result)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_graph_replays_changed_inputs(self, device: str) -> None:
        x = torch.randn(96, 64, device=device).t()
        y = torch.tensor(1.0, device=device)
        actual = torch.empty(x.shape, device=device)
        self._run([x, y], out=actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run([x, y], out=actual)
        for a, b in ((2.0, 3.0), (-4.0, 0.5)):
            x.fill_(a)
            y.fill_(b)
            graph.replay()
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, x + y)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_integer_scalar_float_promotion(self, device: str) -> None:
        x = torch.ones(96, 64, device=device, dtype=torch.int32).t()
        y = torch.tensor(2**32 + 1, device=device, dtype=torch.int64)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.div(x, y)
        actual = self._run([x, y], fn="_div", dtype=torch.float32)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(torch.div(x, y), expected, atol=0, rtol=0)


instantiate_device_type_tests(TestPointwiseTma, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
