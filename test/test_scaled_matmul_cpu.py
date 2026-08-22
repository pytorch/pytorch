# Owner(s): ["module: linear algebra"]

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import _floatx_unpacked_to_f32
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


BLOCK_SIZE = 32
MXFP_RECIPE = [torch._C._ScalingType.BlockWise1x32.value]
NO_SWIZZLE = torch._C._SwizzleType.NO_SWIZZLE.value


def _e8m0(data: torch.Tensor) -> torch.Tensor:
    return data.to(torch.uint8).view(torch.float8_e8m0fnu)


def _unpack_mxfp4(data: torch.Tensor) -> torch.Tensor:
    packed = data.view(torch.uint8)
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    return _floatx_unpacked_to_f32(unpacked, 2, 1)


def _dequantize_mxfp(
    data: torch.Tensor, scale: torch.Tensor, *, transposed: bool = False
) -> torch.Tensor:
    if transposed:
        data = data.t().contiguous()
    values = data.float() if data.dtype == torch.float8_e4m3fn else _unpack_mxfp4(data)
    scales = scale.float().repeat_interleave(BLOCK_SIZE, dim=1)
    result = values * scales[..., : values.size(1)]
    return result.t() if transposed else result


def _make_mxfp_inputs(
    mx_format: str,
    device: torch.device,
    *,
    m: int = 5,
    n: int = 7,
    k: int = 48,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(42)
    if mx_format == "mxfp8":
        a_raw = torch.randint(
            0, 256, (m, k), dtype=torch.uint8, device=device, generator=generator
        )
        b_raw = torch.randint(
            0, 256, (n, k), dtype=torch.uint8, device=device, generator=generator
        )
        a_raw[(a_raw & 0x7F) == 0x7F] = 0
        b_raw[(b_raw & 0x7F) == 0x7F] = 0
        a = a_raw.view(torch.float8_e4m3fn)
        b = b_raw.view(torch.float8_e4m3fn).t()
    else:
        a_raw = torch.randint(
            0,
            256,
            (m, k // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        b_raw = torch.randint(
            0,
            256,
            (n, k // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        a = a_raw.view(torch.float4_e2m1fn_x2)
        b = b_raw.view(torch.float4_e2m1fn_x2).t()

    groups = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_a = _e8m0(
        torch.randint(120, 135, (m, groups), device=device, generator=generator)
    )
    scale_b = _e8m0(
        torch.randint(120, 135, (n, groups), device=device, generator=generator)
    )
    return a, b, scale_a, scale_b


def _scaled_mm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    recipe: list[int] | None = None,
    swizzle: list[int] | None = None,
    contraction_dim: list[int] | None = None,
) -> torch.Tensor:
    recipe = MXFP_RECIPE if recipe is None else recipe
    swizzle = [] if swizzle is None else swizzle
    contraction_dim = [] if contraction_dim is None else contraction_dim
    return torch.ops.aten._scaled_mm_v2.default(
        a,
        b,
        [scale_a],
        recipe,
        swizzle,
        [scale_b],
        recipe,
        swizzle,
        bias,
        out_dtype,
        contraction_dim,
    )


class TestScaledMatmulCPU(TestCase):
    @parametrize("mx_format", ("mxfp4", "mxfp8"))
    @parametrize("out_dtype", (torch.bfloat16, torch.float32))
    @parametrize("bias_dtype", (torch.float16, torch.bfloat16, torch.float32))
    def test_mxfp_numerics(self, device, mx_format, out_dtype, bias_dtype):
        a, b, scale_a, scale_b = _make_mxfp_inputs(mx_format, device)
        bias = torch.randn(b.size(1), device=device, dtype=bias_dtype)
        expected = (
            _dequantize_mxfp(a, scale_a) @ _dequantize_mxfp(b, scale_b, transposed=True)
            + bias.float()
        ).to(out_dtype)

        v1 = torch._scaled_mm(a, b, scale_a, scale_b, bias=bias, out_dtype=out_dtype)
        v2 = _scaled_mm_v2(a, b, scale_a, scale_b, bias, out_dtype)
        v2_explicit = _scaled_mm_v2(
            a,
            b,
            scale_a,
            scale_b,
            bias,
            out_dtype,
            swizzle=[NO_SWIZZLE],
            contraction_dim=[1, 0],
        )
        if out_dtype == torch.float32:
            self.assertEqual(v1, expected, atol=2e-5, rtol=2e-5)
        else:
            self.assertEqual(v1, expected)
        self.assertEqual(v2, v1)
        self.assertEqual(v2_explicit, v1)

    def test_mxfp_nan_accumulation(self, device):
        m, n, k = 2, 5, 64
        a_storage = torch.full((m, k), 0x38, dtype=torch.uint8, device=device)
        b_storage = torch.full((n, k), 0x38, dtype=torch.uint8, device=device)
        a_storage[0, 40] = 0x7F
        b_storage[0, 8] = 0x7F
        b_storage[4, 8] = 0x7F
        a = a_storage.view(torch.float8_e4m3fn)
        b = b_storage.view(torch.float8_e4m3fn).t()
        scale_a = _e8m0(torch.full((m, 2), 127, device=device))
        scale_b = _e8m0(torch.full((n, 2), 127, device=device))
        expected = torch.full((m, n), 64.0, device=device)
        expected[0, :] = torch.nan
        expected[:, 0] = torch.nan
        expected[:, 4] = torch.nan

        v1 = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
        v2 = _scaled_mm_v2(a, b, scale_a, scale_b, None, torch.float32)
        self.assertEqual(v1, expected)
        self.assertEqual(v2, expected)

    def test_mxfp_encodings(self, device):
        fp8_encodings = torch.arange(256, dtype=torch.uint8, device=device)
        a_storage = torch.zeros((256, BLOCK_SIZE), dtype=torch.uint8, device=device)
        a_storage[:, 0] = fp8_encodings
        a = a_storage.view(torch.float8_e4m3fn)
        b_storage = torch.zeros((1, BLOCK_SIZE), dtype=torch.uint8, device=device)
        b_storage[:, 0] = 0x38
        b = b_storage.view(torch.float8_e4m3fn).t()
        scale_a = _e8m0(torch.full((256, 1), 127, device=device))
        scale_b = _e8m0(torch.full((1, 1), 127, device=device))
        actual = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
        expected = fp8_encodings.view(torch.float8_e4m3fn).float()
        self.assertEqual(actual[:, 0], expected)

        fp4_encodings = torch.arange(16, dtype=torch.uint8, device=device)
        a_storage = torch.zeros((16, BLOCK_SIZE // 2), dtype=torch.uint8, device=device)
        a_storage[:, 0] = fp4_encodings | (fp4_encodings << 4)
        a = a_storage.view(torch.float4_e2m1fn_x2)
        b_storage = torch.zeros((2, BLOCK_SIZE // 2), dtype=torch.uint8, device=device)
        b_storage[0, 0] = 0x02
        b_storage[1, 0] = 0x20
        b = b_storage.view(torch.float4_e2m1fn_x2).t()
        scale_a = _e8m0(torch.full((16, 1), 127, device=device))
        scale_b = _e8m0(torch.full((2, 1), 127, device=device))
        actual = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
        expected = _floatx_unpacked_to_f32(fp4_encodings, 2, 1)
        self.assertEqual(actual, expected[:, None].expand(16, 2))

        scale_encodings = torch.tensor(
            [0, 1, 126, 127, 128, 253, 254, 255],
            dtype=torch.uint8,
            device=device,
        )
        shape = (scale_encodings.numel(), BLOCK_SIZE)
        a = torch.ones(shape, dtype=torch.float8_e4m3fn, device=device)
        b = torch.ones(shape, dtype=torch.float8_e4m3fn, device=device).t()
        scale_a = _e8m0(scale_encodings[:, None])
        scale_b = _e8m0(scale_encodings[:, None])
        actual = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
        exponents = (
            scale_encodings.int()[:, None] + scale_encodings.int()[None, :] - 254
        )
        expected = torch.ldexp(torch.full_like(actual, 32.0), exponents)
        expected[-1, :] = torch.nan
        expected[:, -1] = torch.nan
        self.assertEqual(actual, expected)

    @parametrize("mx_format", ("mxfp4", "mxfp8"))
    @parametrize("m,n,k", ((0, 5, 48), (3, 0, 48), (3, 5, 0)))
    def test_mxfp_empty_dimensions(self, device, mx_format, m, n, k):
        a, b, scale_a, scale_b = _make_mxfp_inputs(mx_format, device, m=m, n=n, k=k)
        bias = torch.randn(n, dtype=torch.bfloat16, device=device)
        expected = (
            _dequantize_mxfp(a, scale_a) @ _dequantize_mxfp(b, scale_b, transposed=True)
            + bias.float()
        ).to(torch.bfloat16)
        v1 = torch._scaled_mm(
            a, b, scale_a, scale_b, bias=bias, out_dtype=torch.bfloat16
        )
        v2 = _scaled_mm_v2(a, b, scale_a, scale_b, bias, torch.bfloat16)
        self.assertEqual(v1, expected)
        self.assertEqual(v2, expected)

    @parametrize("mx_format", ("mxfp4", "mxfp8"))
    @parametrize("degenerate_dim", ("k", "n"))
    def test_mxfp_degenerate_mat_b_layout(self, device, mx_format, degenerate_dim):
        n = 1 if degenerate_dim == "n" else 5
        k = (2 if mx_format == "mxfp4" else 1) if degenerate_dim == "k" else 32
        a, b, scale_a, scale_b = _make_mxfp_inputs(mx_format, device, m=3, n=n, k=k)
        b_contiguous = torch.empty_like(b, memory_format=torch.contiguous_format)
        b_contiguous.copy_(b)
        expected = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
        v1 = torch._scaled_mm(
            a, b_contiguous, scale_a, scale_b, out_dtype=torch.float32
        )
        v2 = _scaled_mm_v2(a, b_contiguous, scale_a, scale_b, None, torch.float32)
        self.assertEqual(v1, expected)
        self.assertEqual(v2, expected)

    @parametrize(
        "case,error_type,error_regex",
        (
            ("v1_scale_b", ValueError, "MXFP scale_b must have shape"),
            ("v2_scale_b", ValueError, "MXFP scale_b must have shape"),
            ("layout", ValueError, "MXFP mat_b must be column-major"),
            ("recipe", ValueError, "requires one BlockWise1x32 scale per operand"),
            ("bias_shape", ValueError, "Bias must be size"),
            ("bias_dtype", ValueError, "CPU MXFP bias must have dtype"),
            ("swizzle", NotImplementedError, "CPU MXFP only supports NO_SWIZZLE"),
            ("contraction", NotImplementedError, "only supports contraction_dim"),
        ),
    )
    def test_mxfp_validation_parity(self, device, case, error_type, error_regex):
        a, b, scale_a, scale_b = _make_mxfp_inputs("mxfp8", device, m=3, n=5, k=32)

        def invoke(a, b, scale_a, scale_b):
            if case == "v1_scale_b":
                return torch._scaled_mm(
                    a,
                    b,
                    scale_a,
                    scale_b.t().contiguous(),
                    out_dtype=torch.float32,
                )
            if case == "v2_scale_b":
                return torch.ops.aten._scaled_mm_v2.default(
                    a,
                    b,
                    [scale_a],
                    MXFP_RECIPE,
                    [],
                    [scale_b.t().contiguous()],
                    MXFP_RECIPE,
                    [],
                    None,
                    torch.float32,
                    [],
                )
            if case == "layout":
                return _scaled_mm_v2(
                    a,
                    b.contiguous(),
                    scale_a,
                    scale_b,
                    None,
                    torch.float32,
                )
            if case == "recipe":
                invalid_recipe = [torch._C._ScalingType.TensorWise.value]
                return _scaled_mm_v2(
                    a,
                    b,
                    scale_a,
                    scale_b,
                    None,
                    torch.float32,
                    recipe=invalid_recipe,
                )
            if case == "bias_shape":
                bias = torch.zeros(b.size(1) - 1, dtype=torch.float32, device=device)
                return _scaled_mm_v2(a, b, scale_a, scale_b, bias, torch.float32)
            if case == "bias_dtype":
                bias = torch.zeros(b.size(1), dtype=torch.float64, device=device)
                return _scaled_mm_v2(a, b, scale_a, scale_b, bias, torch.float32)
            if case == "swizzle":
                return _scaled_mm_v2(
                    a,
                    b,
                    scale_a,
                    scale_b,
                    None,
                    torch.float32,
                    swizzle=[1],
                )
            return _scaled_mm_v2(
                a,
                b,
                scale_a,
                scale_b,
                None,
                torch.float32,
                contraction_dim=[0, 1],
            )

        expected_errors = (error_type, torch._dynamo.exc.TorchRuntimeError)
        with self.assertRaisesRegex(expected_errors, error_regex):
            invoke(a, b, scale_a, scale_b)

        mode = FakeTensorMode()
        fake_inputs = tuple(mode.from_tensor(t) for t in (a, b, scale_a, scale_b))
        with mode, self.assertRaisesRegex(expected_errors, error_regex):
            invoke(*fake_inputs)

    @skipIfTorchDynamo(
        "explicit CPU MXFP .out calls do not support FakeTensor/Dynamo tracing"
    )
    @parametrize("api", ("v1", "v2"))
    def test_mxfp_out(self, device, api):
        a, b, scale_a, scale_b = _make_mxfp_inputs("mxfp8", device, m=3, n=5, k=32)
        expected = (
            torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32)
            if api == "v1"
            else _scaled_mm_v2(a, b, scale_a, scale_b, None, torch.float32)
        )
        out = torch.empty(0, dtype=torch.float32, device=device)

        def call_out(out, out_dtype):
            if api == "v1":
                return torch.ops.aten._scaled_mm.out(
                    a,
                    b,
                    scale_a,
                    scale_b,
                    out_dtype=out_dtype,
                    out=out,
                )
            return torch.ops.aten._scaled_mm_v2.out(
                a,
                b,
                [scale_a],
                MXFP_RECIPE,
                [],
                [scale_b],
                MXFP_RECIPE,
                [],
                None,
                out_dtype,
                [],
                out=out,
            )

        result = call_out(out, torch.float32)
        self.assertIs(result, out)
        self.assertEqual(out, expected)

        invalid_dtype = torch.empty_like(out, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "CPU MXFP output must have dtype"):
            call_out(invalid_dtype, torch.float16)

        noncontiguous = torch.empty(
            out.size(1), out.size(0), dtype=out.dtype, device=device
        ).t()
        with self.assertRaisesRegex(ValueError, "CPU MXFP output must be contiguous"):
            call_out(noncontiguous, torch.float32)


instantiate_device_type_tests(TestScaledMatmulCPU, globals(), only_for=("cpu",))


if __name__ == "__main__":
    run_tests()
