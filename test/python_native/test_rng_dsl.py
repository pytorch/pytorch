# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

import re
import unittest
from contextlib import nullcontext

import torch
from torch._native.cutedsl.config import POINTWISE_ARCH_MAJORS
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


# Exact values and generator advancement detect draw-order and reservation errors.


def _disabled() -> AbstractContextManager[None]:
    return torch.backends.python_native.cutedsl.disabled()


class _DrawCase(NamedTuple):
    kind: str
    bounds: tuple[float, float] = (0.0, 1.0)
    sizes: tuple[int, ...] = (4096,)
    entrypoint: str = "registered"
    seed: int = 7
    view_offset: int | None = None
    offset: int = 0
    capture: bool = False
    draws: int = 1
    warmup: int = 3


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLRng(TestCase):
    gen: torch.Generator
    hw_classification = HardwareClassification.CUDA

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.init()
        if torch.version.hip is not None:
            self.skipTest("RNG overrides require CUDA")
        if torch.cuda.get_device_capability()[0] not in POINTWISE_ARCH_MAJORS:
            self.skipTest("RNG overrides require a supported architecture")
        self.gen = torch.cuda.default_generators[torch.cuda.current_device()]

    def _draw_sequence(
        self, device: str, n: int, case: _DrawCase, native: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _disabled() if native else nullcontext():
            self.gen.manual_seed(case.seed)
            out = torch.empty(n + (case.view_offset or 0), device=device)
            if case.view_offset is not None:
                out = out[case.view_offset :]
            first, second = case.bounds
            scale = second - first if case.kind == "uniform" else second
            upper = second if case.kind == "uniform" else 0.0

            def fill() -> torch.Tensor:
                if not native and case.entrypoint == "kernel":
                    from torch._native.ops.rng import cutedsl_kernels

                    return cutedsl_kernels.fill_random(
                        out, case.kind, scale, first, upper
                    )
                result = getattr(out, case.kind + "_")(*case.bounds)
                self.assertIs(result, out)
                return result

            graph = None
            if case.capture:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(case.warmup):
                        fill()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fill()
            if case.offset or not case.capture:
                self.gen.manual_seed(case.seed)
                self.gen.set_offset(case.offset)
            values = []
            for _ in range(case.draws):
                if graph is None:
                    fill()
                else:
                    graph.replay()
                values.append(out.clone())
            return torch.stack(values), self.gen.get_state()

    @parametrize(
        "case",
        [
            _DrawCase(kind, bounds, sizes, entrypoint)
            for entrypoint, sizes in (
                ("registered", (4, 256, 1000, 4096, 100_000, 1 << 20, (1 << 20) + 7)),
                ("kernel", (1 << 10, 1 << 16, 1 << 20, 1 << 24, 1 << 10)),
            )
            for kind, bounds in (
                ("uniform", (0.0, 1.0)),
                ("uniform", (-2.0, 2.0)),
                ("normal", (0.0, 1.0)),
                ("normal", (10.0, 2.5)),
            )
        ]
        + [
            _DrawCase("uniform", bounds, (64,), seed=17)
            for bounds in ((0.0, 1.0), (2.0, 2.0))
        ]
        + [
            _DrawCase(kind, view_offset=offset)
            for kind in ("uniform", "normal")
            for offset in range(1, 4)
        ]
        + [
            _DrawCase("uniform", bounds)
            for bounds in ((10000.0005, 10000.00146), (1e8 + 3, 1e8 + 17))
        ]
        + [
            _DrawCase(
                kind,
                sizes=(16,),
                entrypoint="kernel",
                offset=offset,
                capture=capture,
                draws=2,
            )
            for kind in ("uniform", "normal")
            for offset in ((1 << 63) - 4, 1 << 63)
            for capture in (False, True)
        ]
        + [_DrawCase("uniform", entrypoint="kernel", seed=99, capture=True, draws=3)],
    )
    def test_native_draws_and_generator_state(
        self, device: str, case: _DrawCase
    ) -> None:
        for n in case.sizes:
            self._check_draws(device, n, case)

    def _check_draws(self, device: str, n: int, case: _DrawCase) -> None:
        actual, state = self._draw_sequence(device, n, case, False)
        expected, native_state = self._draw_sequence(device, n, case, True)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(state, native_state)
        if case.capture and case.offset == 0:
            self.assertNotEqual(actual[0], actual[1])
        if case.kind == "uniform" and case.bounds[0] == case.bounds[1]:
            self.assertEqual(
                actual, torch.full_like(actual, case.bounds[0]), atol=0, rtol=0
            )

    @parametrize("case", ("float64", "float16", "generator", "transposed", "empty"))
    def test_unsupported_inputs_preserve_native_behavior(
        self, device: str, case: str
    ) -> None:
        gen = torch.Generator(device=device).manual_seed(5)
        calls = {
            "float64": lambda: torch.empty(
                64, device=device, dtype=torch.float64
            ).uniform_(),
            "float16": lambda: torch.empty(
                64, device=device, dtype=torch.float16
            ).normal_(),
            "generator": lambda: torch.empty(64, device=device).uniform_(generator=gen),
            "transposed": lambda: torch.empty(8, 8, device=device).t().uniform_(),
            "empty": lambda: torch.empty(0, device=device).uniform_(),
        }
        fn = calls[case]
        default_state, explicit_state = self.gen.get_state(), gen.get_state()
        actual = fn()
        default_after, explicit_after = self.gen.get_state(), gen.get_state()
        self.gen.set_state(default_state)
        gen.set_state(explicit_state)
        with _disabled():
            expected = fn()
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(default_after, self.gen.get_state())
        self.assertEqual(explicit_after, gen.get_state())
        if case == "generator":
            self.assertEqual(default_after, default_state)
            self.assertNotEqual(explicit_after, explicit_state)

    def _check_invalid_parameters(self, device: str, case: str) -> None:
        out = torch.full((64,), 7.0, device=device)
        initial = out.clone()
        state = self.gen.get_state()
        calls = {
            "reversed": lambda: out.uniform_(2.0, 1.0),
            "nan_from": lambda: out.uniform_(float("nan"), 1.0),
            "nan_to": lambda: out.uniform_(0.0, float("nan")),
            "overflow": lambda: out.uniform_(-3e38, 3e38),
            "nan_std": lambda: out.normal_(std=float("nan")),
            "negative_std": lambda: out.normal_(std=-1),
        }
        with _disabled(), self.assertRaises(RuntimeError) as native_error:
            calls[case]()
        with self.assertRaisesRegex(
            RuntimeError, re.escape(str(native_error.exception))
        ):
            calls[case]()
        self.assertEqual(out, initial)
        self.assertEqual(self.gen.get_state(), state)

    @parametrize("case", ("reversed", "nan_from", "nan_to", "overflow", "nan_std"))
    def test_invalid_parameters_preserve_output_and_generator(
        self, device: str, case: str
    ) -> None:
        self._check_invalid_parameters(device, case)

    @largeTensorTest("3 GB")
    @parametrize("n", (1 << 29, (1 << 29) + 1))
    @parametrize("distribution", ("uniform_", "normal_"))
    def test_byte_index_boundary_preserves_values_and_state(
        self, device: str, n: int, distribution: str
    ) -> None:
        out = torch.empty(n, device=device)
        samples = []
        offsets = []
        for disabled in (True, False):
            self.gen.manual_seed(7)
            with _disabled() if disabled else nullcontext():
                getattr(out, distribution)()
            samples.append([out[i : i + 16].cpu() for i in (0, n // 2, n - 16)])
            offsets.append(self.gen.get_offset())
        self.assertEqual(samples[0], samples[1], atol=0, rtol=0)
        self.assertEqual(offsets[0], offsets[1])

    def test_kernel_preparation_error_preserves_state(self, device: str) -> None:
        from torch._native.ops.rng.cutedsl_kernels import fill_random

        out = torch.full((32,), 7 + 3j, device=device, dtype=torch.complex64)
        before = out.clone()
        state = self.gen.get_state()
        with self.assertRaisesRegex(KeyError, "complex64"):
            fill_random(out, "uniform", 1.0, 0.0)
        self.assertEqual(out, before)
        self.assertEqual(self.gen.get_state(), state)


instantiate_device_type_tests(TestCuTeDSLRng, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
