# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from torch.testing._internal.opinfo.core import OpInfo

import math
import re
import unittest
import warnings

import torch
from torch._native.cutedsl.config import POINTWISE_ARCH_MAJORS
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    set_default_dtype,
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


def _quantized_storage_bytes(value: torch.Tensor) -> torch.Tensor:
    pack = 1
    if value.dtype is torch.quint4x2:
        pack = 2
    elif value.dtype is torch.quint2x4:
        pack = 4
    offset = value.storage_offset() * value.element_size() // pack
    size = (value.numel() * value.element_size() + pack - 1) // pack
    view = torch.empty(0, device=value.device, dtype=torch.uint8)
    return view.set_(value.untyped_storage(), offset, (size,), (1,))


def _unpack_quantized(value: torch.Tensor, bits: int, pack: int) -> torch.Tensor:
    raw = _quantized_storage_bytes(value).cpu().tolist()
    values = [
        (raw[i // pack] >> ((i % pack) * bits)) & ((1 << bits) - 1)
        for i in range(value.numel())
    ]
    return torch.tensor(values, dtype=torch.uint8).view(value.shape)


def _packed_transpose_reference(
    base: torch.Tensor, dims: tuple[int, int], bits: int, pack: int
) -> torch.Tensor:
    values = _unpack_quantized(base, bits, pack)
    values = values.transpose(*dims).contiguous().view(-1).tolist()
    result = [0] * ((len(values) + pack - 1) // pack)
    for i, value in enumerate(values):
        result[i // pack] |= value << ((i % pack) * bits)
    return torch.tensor(result, device=base.device, dtype=torch.uint8)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLPointwise(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize(
        "name",
        (
            "gt",
            "eq",
            "logical_not",
            "isnan",
            "isinf",
            "signbit",
            "isposinf",
            "isneginf",
        ),
    )
    @parametrize("size", (32, 257, 4096))
    def test_boolean_storage_and_tails(
        self, device: str, dtype: torch.dtype, name: str, size: int
    ) -> None:
        operator = getattr(torch, name)
        with _disabled():
            pattern = torch.tensor(
                [-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan, -math.nan],
                device=device,
                dtype=dtype,
            )
            x = pattern.repeat((size + 7) // 8)[:size]
            args = (x, torch.roll(x, 1)) if name in ("gt", "eq") else (x,)
            expected = operator(*args)
        actual = operator(*args)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(actual.view(torch.uint8), expected.to(torch.uint8))
        if name in ("gt", "eq", "logical_not"):
            raw = torch.full((size + 32,), 255, device=device, dtype=torch.uint8)
            target = raw[16 : 16 + size].view(torch.bool)
            self.assertIs(operator(*args, out=target), target)
            self.assertEqual(target, expected, atol=0, rtol=0)
            self.assertEqual(raw[16 : 16 + size], expected.to(torch.uint8))
            self.assertEqual(raw[:16], torch.full_like(raw[:16], 255))
            self.assertEqual(raw[16 + size :], torch.full_like(raw[16 + size :], 255))

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("name", ("pow", "hypot"))
    def test_cuda_math_extremes(
        self, device: str, dtype: torch.dtype, name: str
    ) -> None:
        with _disabled():
            limits = torch.finfo(dtype)
            edges = (0.0, -0.0, 1.0, -1.0, 2.0, -2.0, math.inf, -math.inf, math.nan)
            values = torch.tensor(
                (*edges, limits.tiny, limits.max), device=device, dtype=dtype
            )
            zero = torch.zeros(1, device=device, dtype=dtype)
            values = torch.cat((values, torch.nextafter(zero, torch.ones_like(zero))))
            other = (
                torch.tensor(
                    (*range(-3, 4), -0.5, 0.5, -0.0, -math.inf, math.inf, math.nan),
                    device=device,
                    dtype=dtype,
                )
                if name == "pow"
                else values
            )
            x, y = torch.meshgrid(values, other, indexing="ij")
            x, y = x.flatten().repeat(16), y.flatten().repeat(16)
            expected = getattr(torch, name)(x, y)
        actual = getattr(torch, name)(x, y)
        self.assertEqual(
            actual,
            expected,
            atol=0,
            rtol=min(default_tolerances(actual, expected)[0], 1e-5),
        )
        with _disabled():
            zeros = expected == 0
            self.assertEqual(
                torch.signbit(actual[zeros]), torch.signbit(expected[zeros])
            )

    @dtypes(torch.int32, torch.int64)
    @parametrize("name", ("fmod", "remainder"))
    @parametrize("api", ("functional", "out", "inplace"))
    def test_integer_remainder_boundaries(
        self, device: str, dtype: torch.dtype, name: str, api: str
    ) -> None:
        with _disabled():
            limits = torch.iinfo(dtype)
            values = torch.tensor(
                [limits.min, limits.min + 1, limits.max],
                device=device,
                dtype=dtype,
            )
            divisors = torch.tensor(
                [-7, -3, -1, 0, 1, 3, 7], device=device, dtype=dtype
            )
            x, y = torch.meshgrid(values, divisors, indexing="ij")
            x, y = x.flatten().repeat(32), y.flatten().repeat(32)
            expected = getattr(torch, name)(x, y)
        if api == "functional":
            actual = getattr(torch, name)(x, y)
        elif api == "out":
            actual = torch.empty_like(x)
            self.assertIs(getattr(torch, name)(x, y, out=actual), actual)
        else:
            actual = x.clone()
            self.assertIs(getattr(actual, name + "_")(y), actual)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    def test_boolean_graph_replays(self, device: str) -> None:
        x = torch.ones(4096, device=device)
        y = torch.zeros_like(x)
        args = (x, y)
        operator = torch.gt
        operator(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = operator(*args)
        for value in (0.0, -1.0, math.nan):
            with _disabled():
                x.fill_(value)
                expected = operator(*args)
            graph.replay()
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("api", ("builtin", "ops"))
    def test_resize_warning(self, device: str, api: str) -> None:
        x = torch.ones(4, device=device)
        operator = torch.neg if api == "builtin" else torch.ops.aten.neg.out

        def invoke(target: torch.Tensor) -> torch.Tensor:
            return operator(x, out=target)

        target = torch.empty(2, device=device)
        with _disabled(), warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always")
            expected = invoke(torch.empty_like(target))
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            self.assertIs(invoke(target), target)
        with _disabled():
            self.assertEqual(target, expected, atol=0, rtol=0)
        self.assertEqual(len(actual_warnings), len(native_warnings))
        for got, ref in zip(actual_warnings, native_warnings):
            message = str(ref.message).split(" (Triggered internally at ")[0]
            self.assertEqual(str(got.message), message)
            self.assertEqual(got.filename, __file__)
            self.assertEqual(got.lineno, invoke.__code__.co_firstlineno + 1)

    def test_integer_ldexp_fallback(self, device: str) -> None:
        x = torch.tensor([1, 2], device=device, dtype=torch.int32)
        self.assertEqual(torch.ldexp(x, x), torch.tensor([2.0, 8.0], device=device))

    def test_default_dtype_promotion(self, device: str) -> None:
        x = torch.arange(1, 33, device=device, dtype=torch.int32)
        for dtype in (torch.float32, torch.float64, torch.float32):
            with set_default_dtype(dtype):
                self._assert_native_result(lambda: torch.sin(x))

    def setUp(self) -> None:
        super().setUp()
        if torch.version.hip is not None:
            self.skipTest("pointwise overrides require CUDA")
        if torch.cuda.get_device_capability()[0] not in POINTWISE_ARCH_MAJORS:
            self.skipTest("pointwise overrides require a supported architecture")

    def _assert_transpose(
        self, got: torch.Tensor, src: torch.Tensor, expected: torch.Tensor
    ) -> None:
        self.assertEqual(got.shape, src.shape)
        self.assertEqual(got.dtype, src.dtype)
        self.assertEqual(got.device, src.device)
        self.assertTrue(got.is_contiguous())
        if src.is_quantized:
            self.assertEqual(got.qscheme(), src.qscheme())
            self.assertEqual(got.q_scale(), src.q_scale())
            self.assertEqual(got.q_zero_point(), src.q_zero_point())
            actual = _quantized_storage_bytes(got)
            if src.dtype in (torch.quint4x2, torch.quint2x4):
                pack = 2 if src.dtype is torch.quint4x2 else 4
                if src.numel() % pack:
                    mask = (1 << (src.numel() % pack * (8 // pack))) - 1
                    actual, expected = actual.clone(), expected.clone()
                    actual[-1] &= mask
                    expected[-1] &= mask
            self.assertEqual(actual, expected)
        else:
            self.assertEqual(_storage_bytes(got), expected)

    @parametrize(
        "dtype,shape,dims,offset",
        (
            (torch.int1, (5, 7, 9), (0, 2), 0),
            (torch.bool, (5, 7, 9), (0, 1), 0),
            (torch.complex128, (35, 31), (0, 1), 1),
        ),
    )
    def test_transpose_copy_bitwise(
        self,
        device: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        dims: tuple[int, int],
        offset: int,
    ) -> None:
        base = _random_bits(shape, dtype, device=device, offset=offset)
        src = base.transpose(*dims)
        with _disabled():
            if dtype is torch.int1:
                ref = (
                    _storage_bytes(base)
                    .view(*shape, dtype.itemsize)
                    .transpose(*dims)
                    .contiguous()
                    .view(-1)
                )
            else:
                ref = _storage_bytes(src.contiguous())
        self._assert_transpose(src.contiguous(), src, ref)

    def test_transpose_copy_zero_tensor(self, device: str) -> None:
        dtype = torch.complex64
        src = torch._efficientzerotensor((5, 7), device=device, dtype=dtype).t()
        target = torch.full(src.shape, 3.0, device=device, dtype=dtype)
        with _disabled():
            expected = torch.empty_like(target).copy_(src)
        self.assertIs(target.copy_(src), target)
        self.assertEqual(target, expected)

    @parametrize("memory_format", (torch.preserve_format, torch.contiguous_format))
    def test_conversion_preserves_singleton_strides(
        self, device: str, memory_format: torch.memory_format
    ) -> None:
        src = torch.empty_strided((4, 1), (1, 37), device=device).fill_(2.5)
        kwargs = {"memory_format": memory_format}
        with _disabled():
            expected = src.to(torch.float64, **kwargs)
        actual = src.to(torch.float64, **kwargs)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual, expected)

    @parametrize(
        "dtype,bits,pack",
        ((torch.quint4x2, 4, 2), (torch.quint2x4, 2, 4)),
    )
    def test_transpose_copy_packed_preserves_backing_elements(
        self, device: str, dtype: torch.dtype, bits: int, pack: int
    ) -> None:
        offset = pack
        backing = torch._empty_affine_quantized(
            (20 + offset,), scale=0.125, zero_point=3, dtype=dtype, device=device
        )
        base = torch._empty_affine_quantized(
            (5, 3), scale=0.125, zero_point=3, dtype=dtype, device=device
        )
        _quantized_storage_bytes(backing).fill_(0xA5)
        _quantized_storage_bytes(base).fill_(0x3C)
        expected = _unpack_quantized(backing, bits, pack)
        expected[offset : offset + 15] = (
            _unpack_quantized(base, bits, pack).t().reshape(-1)
        )
        target = backing[offset : offset + 15].view(3, 5)
        self.assertIs(target.copy_(base.t()), target)
        self.assertEqual(_unpack_quantized(backing, bits, pack), expected)

    @parametrize("normalize", (False, True))
    def test_transpose_payload_block_ownership(
        self, device: str, normalize: bool
    ) -> None:
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
        op = _PayloadTranspose(torch2cute[dtype], 2, 1, 2, 129, 1, normalize)

        @cute.jit
        def one_block(src: Any, dst: Any, stream: Any) -> None:
            op.kernel(src, dst).launch(
                grid=[1, 1, 1], block=[_PAYLOAD_THREADS, 1, 1], stream=stream
            )

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
        fn = compile_tma(dtype.itemsize, outer, middle, a, b)
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

    @parametrize("api", ("copy", "contiguous"))
    def test_transpose_copy_quantized_bitwise(
        self,
        device: str,
        api: str,
    ) -> None:
        base = torch.quantize_per_tensor(
            torch.randn(5, 7, 9), scale=0.125, zero_point=3, dtype=torch.quint8
        ).to(device)
        _quantized_storage_bytes(base).random_(256)
        src = base.transpose(0, 2)
        with _disabled():
            ref = _quantized_storage_bytes(src.contiguous())
        got = (
            src.contiguous()
            if api == "contiguous"
            else torch.empty_like(src, memory_format=torch.contiguous_format).copy_(src)
        )
        self._assert_transpose(got, src, ref)

    @parametrize("dtype,bits,pack", ((torch.quint4x2, 4, 2), (torch.quint2x4, 2, 4)))
    @parametrize("shape", ((5, 7, 8), (3, 5)))
    def test_transpose_copy_packed_quantized_bitwise(
        self,
        device: str,
        dtype: torch.dtype,
        bits: int,
        pack: int,
        shape: tuple[int, ...],
    ) -> None:
        base = torch._empty_affine_quantized(
            shape, scale=0.125, zero_point=3, dtype=dtype, device=device
        )
        _quantized_storage_bytes(base).random_(256)
        src = base.transpose(0, 1)
        ref = _packed_transpose_reference(base, (0, 1), bits, pack)
        got = torch.empty_like(src, memory_format=torch.contiguous_format).copy_(src)
        self._assert_transpose(got, src, ref)

    @ops(
        [op for op in op_db if op.name in _MATH_EDGES and not op.variant_test_name],
        allowed_dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64),
    )
    def test_math_edges(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        values = _MATH_EDGES[op.name]
        fn = op
        with _disabled():
            value = torch.tensor(values, device=device, dtype=torch.float64).to(dtype)
            expected = fn(value)
        got = fn(value)
        if op.name == "round":
            self.assertEqual(got, expected, atol=0, rtol=0)
        else:
            rtol, _ = default_tolerances(got, expected)
            self.assertEqual(got, expected, atol=0, rtol=min(rtol, 1e-5))
        with _disabled():
            zeros = expected == 0
            self.assertEqual(torch.signbit(got[zeros]), torch.signbit(expected[zeros]))

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    def test_frexp_powers_and_subnormals(self, device: str, dtype: torch.dtype) -> None:
        with _disabled():
            powers = torch.tensor(
                [1.0, 16.0, 2.0**20], device=device, dtype=torch.float64
            ).to(dtype)
            below = torch.nextafter(powers, torch.zeros_like(powers))
            tiny = torch.nextafter(
                torch.zeros(1, device=device, dtype=dtype),
                torch.ones(1, device=device, dtype=dtype),
            )
            values = torch.cat(
                (
                    below,
                    tiny,
                    -tiny,
                    torch.tensor([float("nan")], device=device, dtype=dtype),
                )
            )
            expected = torch.frexp(values)
        got = torch.frexp(values)
        self.assertEqual(got.mantissa, expected.mantissa, atol=0, rtol=0)
        self.assertEqual(got.exponent, expected.exponent)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
    @parametrize(
        "op",
        (
            "relu",
            "relu6",
            "hardtanh",
            "hardsigmoid",
            "hardswish",
            "hardshrink",
            "softshrink",
            "threshold",
        ),
    )
    def test_activation_edges(self, device: str, dtype: torch.dtype, op: str) -> None:
        fn = getattr(torch.nn.functional, op)
        kwargs = {"threshold": 0.0, "value": -1.0} if op == "threshold" else {}
        values = torch.tensor([float("nan"), -0.0, 0.0], device=device, dtype=dtype)
        self._assert_native_result(lambda: fn(values, **kwargs), atol=0, rtol=0)

    @parametrize(
        "dtype,op",
        tuple(
            (dtype, op)
            for dtype in (torch.float16, torch.bfloat16)
            for op in ("hardshrink", "softshrink", "threshold", "clamp")
        ),
    )
    def test_scalar_bounds_round_to_storage_dtype(
        self, device: str, dtype: torch.dtype, op: str
    ) -> None:
        values = torch.full((64,), 0.3, device=device, dtype=dtype)
        calls = {
            "hardshrink": lambda: torch.nn.functional.hardshrink(values, 0.3),
            "softshrink": lambda: torch.nn.functional.softshrink(values, 0.3),
            "threshold": lambda: torch.nn.functional.threshold(values, 0.3, -0.3),
            "clamp": lambda: torch.clamp(values, min=0.3, max=0.3),
        }
        self._assert_native_result(calls[op], atol=0, rtol=0)

    @parametrize("dtype", (torch.float32, torch.float64))
    @parametrize("variant", ("functional", "inplace", "out"))
    def test_zero_dimensional_scalar_opmath(
        self, device: str, dtype: torch.dtype, variant: str
    ) -> None:
        value = torch.tensor(1e8, device=device, dtype=dtype)
        with _disabled():
            expected = value + -99999999.0
        out = torch.empty_like(value)

        def fn() -> torch.Tensor:
            if variant == "inplace":
                return value.add_(-99999999.0)
            if variant == "out":
                return torch.add(value, -99999999.0, out=out)
            return value + -99999999.0

        got = fn()
        self.assertEqual(got, expected, atol=0, rtol=0)
        if variant != "functional":
            self.assertIs(got, value if variant == "inplace" else out)

    @parametrize("case", ("celu_zero", "softshrink_nan"))
    def test_parameter_errors(self, device: str, case: str) -> None:
        x = torch.ones(64, device=device)
        self._assert_native_error(
            lambda: torch.nn.functional.celu(x, alpha=0.0)
            if case == "celu_zero"
            else torch.nn.functional.softshrink(x, math.nan)
        )

    def test_linspace_endpoint_precision(self, device: str) -> None:
        out = torch.empty(3, device=device)
        with _disabled():
            expected = torch.linspace(
                1.00000001, 1.00000009, out.numel(), device=device
            )
        self.assertIs(torch.linspace(1.00000001, 1.00000009, out.numel(), out=out), out)
        self.assertEqual(out, expected, atol=0, rtol=0)

    def test_large_scalar_arg_compiles_and_is_not_baked(self, device: str) -> None:
        x = torch.randn(1024, device=device)
        for value in (1e20, 3.4e38, -1e20, 1e-20):
            self._assert_native_result(
                lambda: torch.nn.functional.leaky_relu(x, negative_slope=value)
            )

    def test_misaligned_view_shares_no_plan_with_aligned(self, device: str) -> None:
        base = torch.arange(1024, device=device, dtype=torch.float32)
        for offset, shape in (
            (0, (512,)),
            (1, (512,)),
            (0, (64, 8)),
            (1, (64, 8)),
            (0, (512,)),
        ):
            x = base[offset : offset + math.prod(shape)].view(shape)
            self._assert_native_result(lambda: -x, atol=0, rtol=0)

    def test_empty_copy_destination_falls_back(self, device: str) -> None:
        out = torch.empty(0, 2, device=device, dtype=torch.float64)
        source = torch.ones(1, device=device, dtype=torch.float64)
        with _disabled():
            expected = torch.empty_like(out).copy_(source)
        self.assertIs(out.copy_(source), out)
        self.assertEqual(out, expected)

    @parametrize(
        "layout",
        ("unaligned", "transposed", "strided", "scalar", "double"),
    )
    def test_nullary_fill_values_across_layouts(self, device: str, layout: str) -> None:
        factories = {
            "unaligned": lambda: torch.empty(4096, device=device)[1:1025],
            "transposed": lambda: torch.empty(32, 32, device=device).t(),
            "strided": lambda: torch.empty(2048, device=device)[::2],
            "scalar": lambda: torch.empty((), device=device),
            "double": lambda: torch.empty(8, device=device, dtype=torch.float64),
        }
        out = factories[layout]()
        value = 1e20 if layout == "double" else 2.5
        strides = out.stride()
        with _disabled():
            expected = torch.empty_like(out).fill_(value)
        self.assertIs(out.fill_(value), out)
        self.assertEqual(out.stride(), strides)
        self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize("eps", (0.0, 1e-3))
    def test_logit_zero_eps_and_nan(self, device: str, eps: float) -> None:
        x = torch.tensor([0.0, 1.0, -0.5, 1.5, math.nan], device=device)
        self._assert_native_result(lambda: torch.logit(x, eps), equal_nan=True)

    @ops(
        [
            op
            for op in op_db
            if op.name in ("logaddexp", "logaddexp2", "special.xlog1py")
        ],
        allowed_dtypes=(torch.float32, torch.float64),
    )
    @toleranceOverride(
        {torch.float32: tol(atol=0, rtol=1e-6), torch.float64: tol(atol=0, rtol=1e-14)}
    )
    def test_logarithmic_small_terms(
        self, device: str, dtype: torch.dtype, op: OpInfo
    ) -> None:
        if op.name == "special.xlog1py":
            x = torch.tensor([1.0, 1e30, math.inf], device=device, dtype=dtype)
            y = torch.full_like(x, 1e-8)
        else:
            x = torch.zeros(1, device=device, dtype=dtype)
            y = torch.full_like(x, -30.0 if op.name == "logaddexp2" else -20.0)
        self._assert_native_result(lambda: op(x, y))

    @parametrize("factory", ("arange", "linspace"))
    def test_range_overlapping_output(self, device: str, factory: str) -> None:
        out = torch.zeros(1, device=device).expand(2, 3)
        with self.assertRaisesRegex(RuntimeError, "more than one element"):
            torch.arange(6.0, out=out) if factory == "arange" else torch.linspace(
                0, 5, 6, out=out
            )
        with _disabled():
            self.assertEqual(out, torch.zeros_like(out))

    @parametrize("dtype", (torch.float16, torch.float32))
    def test_large_linspace_backward_distance(
        self, device: str, dtype: torch.dtype
    ) -> None:
        n = 2**24 + 2
        with _disabled():
            expected = torch.linspace(1, 0, n, device=device, dtype=dtype)[-3:].clone()
        out = torch.empty(n, device=device, dtype=dtype)
        torch.linspace(1, 0, n, out=out)
        with _disabled():
            self.assertEqual(out[-3:], expected, atol=0, rtol=0)

    @parametrize("inplace", (False, True))
    def test_lerp_input_dtype_error(self, device: str, inplace: bool) -> None:
        x = torch.ones(2, device=device, dtype=torch.float16)
        y = torch.ones(2, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "expected dtype.*end"):
            x.lerp_(y, 0.5) if inplace else torch.lerp(x, y, 0.5)
        self.assertEqual(x, torch.ones_like(x))

    def test_nan_to_num_output_dtype_error(self, device: str) -> None:
        x = torch.ones(2, device=device)
        out = torch.zeros(2, device=device, dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "dtype of out"):
            torch.nan_to_num(x, out=out)
        self.assertEqual(out, torch.zeros_like(out))

    @parametrize("op", ("add", "leaky_relu"))
    def test_finite_scalar_overflow(self, device: str, op: str) -> None:
        x = torch.ones(2, device=device)
        with self.assertRaisesRegex(RuntimeError, "cannot be converted.*overflow"):
            torch.add(
                x, x, alpha=1e100
            ) if op == "add" else torch.nn.functional.leaky_relu(
                x, negative_slope=1e100
            )

    @parametrize(
        "kind", ("channels_last", "resize_channels_last", "resize_transpose", "rsub")
    )
    def test_output_layout_preserved(self, device: str, kind: str) -> None:
        if "channels_last" in kind:
            x = torch.randn(2, 4, 4, 4, device=device).contiguous(
                memory_format=torch.channels_last
            )
        else:
            x = torch.randn(64, 64, device=device)

        def call() -> torch.Tensor:
            if kind == "channels_last":
                return torch.relu(x)
            if kind == "rsub":
                return torch.rsub(x, x.t())
            source = x.t() if kind == "resize_transpose" else x
            return torch.neg(source, out=torch.empty(0, device=device))

        with _disabled():
            expected = call()
        actual = call()
        with _disabled():
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual, expected)

    @parametrize("op", ("add", "gt", "complex_add"))
    def test_weak_scalar_computation_precision(self, device: str, op: str) -> None:
        x = torch.tensor([16777217], device=device, dtype=torch.int32)
        if op == "complex_add":
            x = torch.tensor([1e8], device=device)

        def fn() -> torch.Tensor:
            if op == "add":
                return x + 0.5
            if op == "gt":
                return x > 16777216.0
            return x + complex(-99999999.0, 0.0)

        self._assert_native_result(fn, atol=0, rtol=0)

    @parametrize("dtype,scalar", ((torch.uint16, 70000), (torch.float32, 2**63 + 1)))
    def test_large_unsigned_scalar_division_fallback(
        self, device: str, dtype: torch.dtype, scalar: int
    ) -> None:
        x = torch.tensor(7, device=device, dtype=dtype)
        self._assert_native_result(lambda: x / scalar, atol=0, rtol=0)

    def test_frexp_preserves_transposed_output_layout(self, device: str) -> None:
        x = torch.randn(96, 64, device=device).t()
        with _disabled():
            expected = torch.frexp(x)
        actual = torch.frexp(x)
        for got, ref in zip(actual, expected):
            self.assertEqual(got.stride(), ref.stride())
            with _disabled():
                self.assertEqual(got, ref)

    @parametrize("op", ("lerp", "signbit"))
    def test_output_dtype_errors_preserve_target(self, device: str, op: str) -> None:
        with _disabled():
            x = torch.ones(64, device=device, dtype=torch.float16)
            y, weight = x * 2, x * 0.5
            out = torch.full((64,), 7.0, device=device)
            before = out.clone()

        def call() -> torch.Tensor:
            return (
                torch.lerp(x, y, weight, out=out)
                if op == "lerp"
                else torch.signbit(x, out=out)
            )

        self._assert_native_error(call)
        with _disabled():
            self.assertEqual(out, before)

    @parametrize("weight_kind", ("scalar", "zero_dim"))
    def test_lerp_scalar_wide_output(self, device: str, weight_kind: str) -> None:
        x = torch.ones(64, device=device, dtype=torch.float16)
        weight = 0.5 if weight_kind == "scalar" else torch.tensor(0.5, device=device)
        out = torch.empty(64, device=device)
        with _disabled():
            expected = torch.lerp(x, x * 2, weight, out=torch.empty_like(out))
        self.assertIs(torch.lerp(x, x * 2, weight, out=out), out)
        with _disabled():
            self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize("dtype", (torch.float32, torch.float64))
    def test_elu_coefficient_overflow(self, device: str, dtype: torch.dtype) -> None:
        x = torch.tensor([-1e-30, 0.0, 1.0], device=device, dtype=dtype)
        self._assert_native_result(
            lambda: torch.ops.aten.elu.default(x, 1e30, 1e30, 1.0), atol=0, rtol=0
        )

    def test_boxed_out_alias(self, device: str) -> None:
        x = torch.ones(64, device=device, dtype=torch.float16)
        out = torch.empty(64, device=device)
        keys = torch._C._dispatch_keys(x)
        with _disabled():
            expected = torch.empty_like(out)
            native = torch.library.get_kernel("aten::add.out", "CUDA")
            native.call_boxed(keys, x, 0.5, out=expected)
        kernel = torch.library.get_kernel("aten::add.out", "CUDA")
        self.assertIs(kernel.call_boxed(keys, x, 0.5, out=out), out)
        self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize(
        "dtype,value",
        (
            (torch.float16, 70000.0),
            (torch.float32, 1e100),
            (torch.float16, 70000),
            (torch.int32, 2**40),
            (torch.int64, 2**63),
        ),
    )
    def test_factory_overflow_preserves_native_error(
        self, device: str, dtype: torch.dtype, value: int | float
    ) -> None:
        out = torch.full((64,), 7, device=device, dtype=dtype)
        before = out.clone()
        self._assert_native_error(lambda: out.fill_(value))
        self.assertEqual(out, before)

    def test_linspace_intermediate_overflow_matches_native(self, device: str) -> None:
        self._assert_native_result(
            lambda: torch.linspace(-3e38, 3e38, 9, device=device), atol=0, rtol=0
        )

    def _assert_native_result(self, call: Callable[[], Any], **kwargs: Any) -> None:
        with _disabled():
            expected = call()
        actual = call()
        with _disabled():
            self.assertEqual(actual, expected, **kwargs)

    def _assert_native_error(self, call: Callable[[], torch.Tensor]) -> None:
        with _disabled(), self.assertRaises(RuntimeError) as native_error:
            call()
        with self.assertRaisesRegex(
            RuntimeError, re.escape(str(native_error.exception))
        ):
            call()

    @parametrize(
        "name",
        (
            "trunc",
            "sign",
            "relu",
            "abs",
            "round",
            "frac",
            "sgn",
            "bitwise_not",
            "silu",
            "elu",
            "celu",
            "mish",
            "softplus",
            "hardtanh",
            "hardsigmoid",
            "hardshrink",
            "softshrink",
            "hardswish",
            "leaky_relu",
            "gelu",
            "clamp",
            "signbit",
            "isposinf",
            "isneginf",
        ),
    )
    def test_native_output_dtype_policies(self, device: str, name: str) -> None:
        dtype = torch.int32 if name == "bitwise_not" else torch.float32
        with _disabled():
            x = torch.ones(64, device=device, dtype=dtype)
            out = torch.full(
                (64,),
                7,
                device=device,
                dtype=torch.int64 if dtype is torch.int32 else torch.float64,
            )
            before = out.clone()
        op = getattr(torch.ops.aten, name).out
        kwargs = {"min": -1, "max": 1} if name == "clamp" else {}
        self._assert_native_error(lambda: op(x, out=out, **kwargs))
        with _disabled():
            self.assertEqual(out, before)

    @parametrize(
        "name",
        (
            "sin",
            "square",
            "threshold",
            "tensor_clamp",
            "logaddexp",
            "logaddexp2",
            "bitwise_left_shift",
            "bitwise_right_shift",
        ),
    )
    def test_native_output_casts_remain_supported(self, device: str, name: str) -> None:
        x = torch.tensor([-0.5, 0.0, 0.5], device=device)
        out = torch.empty(x.shape, device=device, dtype=torch.float64)
        if name in ("threshold", "logaddexp", "logaddexp2"):
            x = x.half()
        if name.startswith("bitwise_"):
            dtype = torch.int32 if name == "bitwise_left_shift" else torch.int64
            x = torch.full(
                (1,), 1 if dtype is torch.int32 else 2**40, device=device, dtype=dtype
            )
            out = torch.empty_like(
                x, dtype=torch.int64 if dtype is torch.int32 else torch.int32
            )

        def call(target: torch.Tensor) -> torch.Tensor:
            if name == "threshold":
                return torch.ops.aten.threshold.out(x, 0.1, 0.3, out=target)
            if name in ("logaddexp", "logaddexp2"):
                return getattr(torch, name)(x, x, out=target)
            if name.startswith("bitwise_"):
                shift = 31 if name == "bitwise_left_shift" else 40
                return getattr(torch, name)(x, torch.full_like(x, shift), out=target)
            if name == "tensor_clamp":
                return torch.clamp(
                    x,
                    min=torch.tensor(-1.0, device=device),
                    max=torch.tensor(1.0, device=device),
                    out=target,
                )
            return getattr(torch, name)(x, out=target)

        with _disabled():
            expected = call(torch.empty_like(out))
        self.assertIs(call(out), out)
        with _disabled():
            self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize(
        "name,slot,allowed",
        (
            ("addcdiv", 0, False),
            ("addcdiv", 1, False),
            ("addcdiv", 2, False),
            ("lerp", 0, False),
            ("lerp", 1, False),
            ("lerp", 2, True),
        ),
    )
    def test_cpu_scalar_input_policies(
        self, device: str, name: str, slot: int, allowed: bool
    ) -> None:
        inputs = [torch.full((64,), 1.0 + i, device=device) for i in range(3)]
        inputs[slot] = torch.tensor(2.0, device="cpu")
        op = getattr(torch, name)
        if not allowed:
            self._assert_native_error(lambda: op(*inputs))
            return
        with _disabled():
            expected = op(*inputs)
        actual = op(*inputs)
        with _disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("dtype", (torch.float32, torch.float64))
    @parametrize("coefficient", (1.0, 1e30))
    def test_addcmul_native_grouping(
        self, device: str, dtype: torch.dtype, coefficient: float
    ) -> None:
        x = torch.full(
            (64,), -1.0 if coefficient == 1 else 0.0, device=device, dtype=dtype
        )
        delta = 2**-52 if dtype is torch.float64 else 2**-23
        a = torch.full_like(x, 1 + delta if coefficient == 1 else 1e30)
        b = torch.full_like(x, 1 - delta if coefficient == 1 else 1e-30)
        self._assert_native_result(
            lambda: torch.addcmul(x, a, b, value=coefficient), atol=0, rtol=0
        )

    def test_celu_native_reciprocal(self, device: str) -> None:
        x = torch.ones(64, device=device)
        self._assert_native_error(lambda: torch.nn.functional.celu(x, alpha=1e-40))
        x = torch.zeros(64, device=device, dtype=torch.float64)
        self._assert_native_result(
            lambda: torch.nn.functional.celu(x, alpha=1e-320), atol=0, rtol=0
        )

    @parametrize(
        "dtype,value,scalar", ((torch.float16, 1.0, 1.0001), (torch.int32, 0, 2**32))
    )
    @parametrize("zero_dim", (True, False))
    def test_boxed_scalar_promotion_tiers(
        self,
        device: str,
        dtype: torch.dtype,
        value: float,
        scalar: int | float,
        zero_dim: bool,
    ) -> None:
        x = torch.tensor(value, device=device, dtype=dtype)
        if not zero_dim:
            x = x.reshape(1)
        keyset = torch._C._dispatch_keys(x)
        with _disabled():
            native = torch.library.get_kernel("aten::eq.Tensor", "CUDA")
            expected = native.call_boxed(keyset, x, scalar)
        kernel = torch.library.get_kernel("aten::eq.Tensor", "CUDA")
        actual = kernel.call_boxed(keyset, x, scalar)
        with _disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize(
        "dtype,start,end,step,size",
        (
            (torch.float32, 1e40, 2e40, 1e40, 1),
            (torch.float32, 0.0, 1e100, 1e99, 10),
            (torch.int64, 2**63 - 2, 2**63, 1, 2),
        ),
    )
    def test_arange_accumulator_overflow(
        self,
        device: str,
        dtype: torch.dtype,
        start: int | float,
        end: int | float,
        step: int | float,
        size: int,
    ) -> None:
        out = torch.full((size,), 7, device=device, dtype=dtype)
        before = out.clone()
        self._assert_native_error(lambda: torch.arange(start, end, step, out=out))
        with _disabled():
            self.assertEqual(out, before)

    @parametrize("dtype,start", ((torch.int32, 2**40), (torch.float16, 70000)))
    def test_arange_accumulator_preserves_valid_values(
        self, device: str, dtype: torch.dtype, start: int
    ) -> None:
        out = torch.empty(64, device=device, dtype=dtype)
        with _disabled():
            expected = torch.arange(start, start + 64, out=torch.empty_like(out))
        self.assertIs(torch.arange(start, start + 64, out=out), out)
        with _disabled():
            self.assertEqual(out, expected, atol=0, rtol=0)

    @parametrize(
        "start,end,step,size",
        (
            (-(2**63), 2**63 - 1, 2**62, 4),
            (2**63 - 1, -(2**63), -(2**62), 4),
            (0, 2**63 - 1, 2**63 - 1, 1),
            (0, -(2**63), -(2**63), 1),
        ),
    )
    def test_arange_int64_count_overflow(
        self, device: str, start: int, end: int, step: int, size: int
    ) -> None:
        out = torch.empty(size, device=device, dtype=torch.int64)
        with _disabled(), warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always")
            expected = torch.arange(start, end, step, out=torch.empty_like(out))
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            self.assertIs(torch.arange(start, end, step, out=out), out)
        with _disabled():
            self.assertEqual(out, expected, atol=0, rtol=0)
        self.assertEqual(
            [str(w.message) for w in actual_warnings],
            [str(w.message) for w in native_warnings],
        )

    @parametrize(
        "kind",
        ("mul", "cpu_scalar", "coefficient", "clamp", "fill", "arange", "linspace"),
    )
    def test_integer_scalar_direct_rounding(self, device: str, kind: str) -> None:
        value = (1 << 62) + (1 << 38) + 1
        x = torch.ones(64, device=device)

        def call() -> torch.Tensor:
            if kind == "coefficient":
                return torch.addcmul(torch.zeros_like(x), x, x, value=value)
            if kind == "clamp":
                return torch.clamp(x * (1 << 63), max=value)
            if kind == "fill":
                return torch.empty_like(x).fill_(value)
            if kind == "arange":
                return torch.arange(
                    value, value + 64 * (1 << 39), 1 << 39, out=torch.empty_like(x)
                )
            if kind == "linspace":
                return torch.linspace(
                    value, value + 63 * (1 << 39), 64, out=torch.empty_like(x)
                )
            scalar = (
                torch.tensor(value, dtype=torch.int64)
                if kind == "cpu_scalar"
                else value
            )
            return x * scalar

        self._assert_native_result(call, atol=0, rtol=0)


instantiate_device_type_tests(TestCuTeDSLPointwise, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
