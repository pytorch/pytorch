# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

import math
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


_MATH_EDGES: dict[str, tuple[float, ...]] = {
    "expm1": (-1e-8, 1e-8),
    "log1p": (-1e-8, 1e-8),
    "erfc": (5.0, 9.0),
    "asinh": (-1e-8, 1e-8),
    "acosh": (1.0000001,),
    "atanh": (-1e-8, 1e-8),
    "sinh": (-710.0, -89.0, -1e-8, 89.0, 710.0),
    "cosh": (-710.0, -89.0, 89.0, 710.0),
    "round": (-4503599627370497.0, -8388609.0, 8388609.0, 4503599627370497.0),
}


def _disabled() -> AbstractContextManager[None]:
    return torch.backends.python_native.cutedsl.disabled()


def _random_bits(
    shape: tuple[int, ...], dtype: torch.dtype, *, device: str, offset: int = 0
) -> torch.Tensor:
    numel = math.prod(shape)
    storage = torch.empty(numel + offset, device=device, dtype=dtype)
    storage.as_strided((storage.numel(),), (1,)).view(torch.uint8).random_(256)
    if offset:
        storage = storage[offset:]
    return storage.view(shape)


def _storage_bytes(value: torch.Tensor) -> torch.Tensor:
    return value.as_strided((value.numel(),), (1,)).view(torch.uint8)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLPointwise(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("normalize", (False, True))
    def test_transpose_payload_block_ownership(
        self, device: str, normalize: bool
    ) -> None:
        import cutlass
        import cutlass.cute as cute

        from torch._native.cutedsl import launch
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise.transpose_copy import (
            _PAYLOAD_THREADS,
            _PayloadTranspose,
        )

        dtype = torch.int8 if normalize else torch.float32
        src = torch.arange(516, device=device).to(dtype).view(-1, 1)
        dst = torch.full_like(src, -7)
        op = _PayloadTranspose(torch2cute[dtype], 1, normalize, False)

        @cute.jit
        def one_block(src: Any, dst: Any, stream: Any) -> None:
            divisors = tuple(
                cute.FastDivmodDivisorV2(cutlass.Int32(v)) for v in (2, 1, 2, 129, 1)
            )
            op.kernel(
                src,
                dst,
                divisors,
                cutlass.Int32(129),
                cutlass.Int32(4),
                cutlass.Int32(1),
            ).launch(grid=[1, 1, 1], block=[_PAYLOAD_THREADS, 1, 1], stream=stream)

        fake_src = launch.fake_compact(torch2cute[dtype], (516, 1), align=16)
        fake_dst = launch.fake_compact(torch2cute[dtype], (516, 1), align=16)
        compiled = launch.compile_kernel(
            one_block, fake_src, fake_dst, cute.runtime.make_fake_stream()
        )
        compiled(launch.read_only(src), dst, launch.stream())
        with _disabled():
            expected = torch.full_like(dst, -7)
            row = src[:129]
            expected[:129] = (row != 0).to(dtype) if normalize else row
            self.assertEqual(dst, expected, atol=0, rtol=0)

    def test_transpose_copy_tma_graph_replays(self, device: str) -> None:
        dtype, shape = torch.uint16, (2, 96, 3, 80)
        if torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("TMA transpose requires Blackwell")
        from torch._native.cutedsl import launch
        from torch._native.ops.pointwise import transpose_copy
        from torch._native.ops.pointwise.transpose_tma import compile_tma

        base = _random_bits(shape, dtype, device=device)
        dims = (1, len(shape) - 1)
        src = base.transpose(*dims)
        dst = torch.empty_like(src, memory_format=torch.contiguous_format)
        outer, a, b = shape[0], shape[1], shape[-1]
        middle = math.prod(shape[2:-1])
        spec = (outer, b, middle, a, 1, dtype.itemsize)
        fn = compile_tma(dtype.itemsize)
        source = launch.read_only(
            transpose_copy._tiled_view(src, spec, dtype.itemsize, source=True)
        )
        target = transpose_copy._tiled_view(dst, spec, dtype.itemsize, source=False)
        fn(source, target, launch.stream())
        with _disabled():
            expected = (
                _storage_bytes(base)
                .view(*shape, dtype.itemsize)
                .transpose(*dims)
                .contiguous()
                .view(-1)
            )
        self.assertEqual(_storage_bytes(dst), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(source, target, launch.stream())
        for mask in (0x5A, 0x33):
            _storage_bytes(base).bitwise_xor_(mask)
            dst.zero_()
            graph.replay()
            with _disabled():
                expected = (
                    _storage_bytes(base)
                    .view(*shape, dtype.itemsize)
                    .transpose(*dims)
                    .contiguous()
                    .view(-1)
                )
            self.assertEqual(_storage_bytes(dst), expected)

    @parametrize(
        "dtype,shapes",
        (
            (torch.float32, ((2, 5, 2, 3, 7), (2, 9, 2, 7, 11))),
            (torch.uint16, ((2, 512, 2, 2048, 1), (2, 1024, 2, 1024, 1))),
        ),
    )
    def test_transpose_copy_reuses_runtime_geometry(
        self, device: str, dtype: torch.dtype, shapes: tuple[tuple[int, ...], ...]
    ) -> None:
        from torch._native.ops.pointwise import transpose_copy

        cached = None
        for shape in shapes:
            outer, a, middle, b, inner = shape
            with _disabled():
                src = _random_bits(shape, dtype, device=device).transpose(1, 3)
                expected = src.contiguous()
                target = torch.empty_like(expected)
            transpose_copy.run(
                src, target, (outer, b, middle, a, inner, dtype.itemsize)
            )
            self.assertEqual(target, expected, atol=0, rtol=0)
            if cached is None:
                cached = len(transpose_copy._PLANS)
            self.assertEqual(len(transpose_copy._PLANS), cached)


instantiate_device_type_tests(TestCuTeDSLPointwise, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
