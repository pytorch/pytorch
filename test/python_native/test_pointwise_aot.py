# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

import os
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    shell,
    TestCase,
)


def _probe(case: str, device: str) -> None:
    from python_native.test_pointwise_cutedsl import (
        _packed_transpose_reference,
        _quantized_storage_bytes,
        _storage_bytes,
    )

    check = TestCase()
    disabled = torch.backends.python_native.cutedsl.disabled
    if case.startswith(("dense", "tma_arithmetic")):
        dtype = (
            torch.bfloat16 if case == "tma_arithmetic_specialized" else torch.float32
        )
        if case.startswith("dense"):
            dtype = getattr(torch, case.removeprefix("dense_").removesuffix("_large"))
        if case.startswith("tma_arithmetic"):
            rows = 1024 if case == "tma_arithmetic_specialized" else 64
            cols = (
                rows
                if case in ("tma_arithmetic_square", "tma_arithmetic_specialized")
                else 96
            )
            x = torch.randn(cols, rows, device=device, dtype=dtype).t()
            if case in ("tma_arithmetic_both", "tma_arithmetic_specialized"):
                y = torch.randn(cols, rows + 16, device=device, dtype=dtype)[
                    :, :rows
                ].t()
            elif case == "tma_arithmetic_padded":
                y = torch.randn(64, cols + 32, device=device)[:, :cols]
            else:
                y = torch.randn(x.shape, device=device)
        else:
            size = 4 * 1024**2 if case.endswith("_large") else 4096
            x = torch.randn(size, device=device, dtype=dtype)
            y = torch.randn_like(x)
        target = torch.full(x.shape, -127.0, device=device, dtype=dtype)
        with disabled():
            expected = torch.add(x, y, alpha=0.375)

        def fn() -> torch.Tensor:
            return torch.add(x, y, alpha=0.375, out=target)

    elif case.startswith("copy"):
        if case == "copy_tma":
            base = torch.randn(2, 2048, 1024, device=device, dtype=torch.bfloat16)
            x = base.transpose(1, 2)
        elif case == "copy_payload":
            x = torch.randn(2, 7, 11, 13, device=device).transpose(0, 1)
        elif case == "copy_complex":
            x = torch.randn(35, 31, device=device, dtype=torch.complex128).t()
        else:
            raise ValueError(case)
        target = torch.full(x.shape, 7, device=device, dtype=x.dtype)
        with disabled():
            expected = x.contiguous()

        def fn() -> torch.Tensor:
            return target.copy_(x)

    elif case.startswith("packed"):
        pack = int(case[-1])
        dtype = torch.quint4x2 if pack == 2 else torch.quint2x4
        base = torch._empty_affine_quantized(
            (3, 5), scale=0.125, zero_point=3, dtype=dtype, device=device
        )
        _quantized_storage_bytes(base).random_(256)
        x = base.t()
        target = torch._empty_affine_quantized(
            x.shape, scale=0.125, zero_point=3, dtype=dtype, device=device
        )
        _quantized_storage_bytes(target).fill_(0xA5)
        expected = _packed_transpose_reference(base, (0, 1), 8 // pack, pack)
        mask = (1 << ((target.numel() % pack) * (8 // pack))) - 1
        expected[-1] = (expected[-1] & mask) | (0xA5 & (255 ^ mask))

        def fn() -> torch.Tensor:
            return target.copy_(x)

    else:
        raise ValueError(case)
    actual = fn()
    torch.cuda.synchronize()
    check.assertIs(actual, target)
    if case.startswith("packed"):
        check.assertEqual(_quantized_storage_bytes(actual), expected)
        check.assertEqual(actual.q_scale(), x.q_scale())
        check.assertEqual(actual.q_zero_point(), x.q_zero_point())
    else:
        check.assertEqual(actual, expected)
        check.assertEqual(actual.dtype, expected.dtype)
    if case.startswith("copy"):
        check.assertEqual(_storage_bytes(actual), _storage_bytes(expected))


def _fallback(case: str, device: str) -> None:
    from contextlib import nullcontext

    check = TestCase()
    disabled = torch.backends.python_native.cutedsl.disabled
    x = torch.ones((32, 32), device=device)
    generator = torch.Generator(device=device).manual_seed(13)
    state = generator.get_state()

    def draw() -> torch.Tensor:
        generator.set_state(state)
        return x.uniform_(generator=generator)

    calls = {
        "broadcast": lambda: x + torch.ones((32,), device=device),
        "mixed_dtype": lambda: x + torch.ones_like(x, dtype=torch.float64),
        "copy_cast": lambda: x.copy_(torch.full_like(x, 2.0, dtype=torch.float64)),
        "generator": draw,
        "disabled": lambda: torch.add(x, x),
    }
    operation = calls[case]

    with disabled():
        expected = operation().clone()
        expected_state = generator.get_state() if case == "generator" else None
        if case in ("copy_cast", "generator"):
            x.fill_(-17.0)
    with disabled() if case == "disabled" else nullcontext():
        actual = operation()
        torch.cuda.synchronize()
    check.assertEqual(actual, expected, atol=0, rtol=0)
    if case in ("copy_cast", "generator"):
        check.assertIs(actual, x)
    if case == "generator":
        check.assertEqual(generator.get_state(), expected_state)


def _default_aot_probe(case: str, device: str) -> None:
    from torch._native.ops.pointwise import transpose_copy

    count = len(transpose_copy._PLANS)
    _probe(case, device)
    TestCase().assertEqual(len(transpose_copy._PLANS), count)


def _invalid_rng(case: str, device: str) -> None:
    from python_native.test_rng_dsl import TestCuTeDSLRngCUDA

    check = TestCuTeDSLRngCUDA()
    index = torch._utils._get_device_index(device, optional=True)
    check.gen = torch.cuda.default_generators[index]
    check._check_invalid_parameters(
        device, "negative_std" if case == "std" else "reversed"
    )


def _rng_state(distribution: str, capture: bool, device: str) -> None:
    from python_native.test_rng_dsl import _DrawCase, TestCuTeDSLRngCUDA

    check = TestCuTeDSLRngCUDA()
    check.gen = torch.cuda.default_generators[
        torch.device(device).index or torch.cuda.current_device()
    ]
    case = _DrawCase(
        distribution.removesuffix("_"),
        seed=13 if capture else 7,
        capture=capture,
        draws=2 if capture else 1,
        view_offset=None if capture else 1,
        warmup=1,
    )
    check._check_draws(device, 4096, case)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestPointwiseAot(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _require_coverage(self, op: str, device: str) -> None:
        from torch._native import aot_manifest

        coverage = aot_manifest.get_coverage(op, "CUDA")
        if coverage is None or not coverage.is_available(torch.device(device)):
            self.skipTest("AOT artifacts do not cover this CUDA device")

    def _run_probe(self, name: str, *args: object, jit_enabled: bool = False) -> None:
        source = f"from python_native.test_pointwise_aot import {name}; {name}{args!r}"
        env = dict(
            os.environ,
            TORCH_DISABLE_NATIVE_JIT="0" if jit_enabled else "1",
            TORCH_DISABLE_NATIVE_AOT="0",
        )
        code = shell(
            [sys.executable, "-c", source],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            timeout=300,
        )
        self.assertEqual(code, 0)

    @parametrize(
        "case",
        (
            "dense_float16",
            "dense_bfloat16",
            "dense_float32",
            "dense_float64_large",
            "tma_arithmetic",
            "tma_arithmetic_square",
            "tma_arithmetic_both",
            "tma_arithmetic_padded",
            "tma_arithmetic_specialized",
            "copy_tma",
            "copy_payload",
            "copy_complex",
            "packed2",
            "packed4",
        ),
    )
    def test_values_and_aliases(self, device: str, case: str) -> None:
        op = "copy_" if case.startswith(("copy", "packed")) else "add.Tensor"
        self._require_coverage(op, device)
        self._run_probe("_probe", case, device)

    @parametrize("case", ("copy_payload", "packed2", "packed4"))
    def test_default_aot_copy_does_not_compile_jit(
        self, device: str, case: str
    ) -> None:
        self._require_coverage("copy_", device)
        self._run_probe("_default_aot_probe", case, device, jit_enabled=True)

    @parametrize("distribution", ("uniform_", "normal_"))
    @parametrize("capture", (False, True))
    def test_rng_state(self, device: str, distribution: str, capture: bool) -> None:
        self._require_coverage(distribution, device)
        self._run_probe("_rng_state", distribution, capture, device)

    @parametrize(
        "case", ("broadcast", "mixed_dtype", "copy_cast", "generator", "disabled")
    )
    def test_fallback_preserves_native_behavior(self, device: str, case: str) -> None:
        op = {"copy_cast": "copy_", "generator": "uniform_"}.get(case, "add.Tensor")
        self._require_coverage(op, device)
        self._run_probe("_fallback", case, device)

    @parametrize("case,op", (("std", "normal_"), ("bounds", "uniform_")))
    def test_invalid_rng_preserves_state(self, device: str, case: str, op: str) -> None:
        self._require_coverage(op, device)
        self._run_probe("_invalid_rng", case, device)


instantiate_device_type_tests(TestPointwiseAot, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
