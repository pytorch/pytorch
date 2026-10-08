# Owner(s): ["module: linear algebra"]

import inspect

import torch
import torch.func._random as prng
from torch.nn import functional as F
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import (
    _f32_to_e8m0_rceil,
    _f32_to_fp8_nvidia_sr_with_words,
    compute_error,
    from_blocked,
    from_blocked_format,
    mxfp8_32x32_swizzle_f,
    to_mxfp as to_mxfp8_reference,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    subtest,
    TestCase,
)
from torch.testing._internal.mxfp8_test_utils import (
    assert_mxfp8_semantics,
    make_f32_to_e8m0_rceil_cases,
    make_mxfp8_semantic_cases,
)


def _quantize_mxfp8_reference(
    input: torch.Tensor,
    *,
    qdata_dtype: torch.dtype,
    scaling_algorithm: F.ScalingAlgorithm,
    scaling_type: F.ScalingType,
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
):
    if (
        qdata_dtype != torch.float8_e4m3fn
        or scaling_algorithm != F.ScalingAlgorithm.MXFP_E8M0_RU
        or scaling_type != F.ScalingType.BlockWise1x32
    ):
        raise ValueError("unsupported MXFP8 reference recipe")
    scales, qdata = to_mxfp8_reference(input, format="mxfp8", swizzle_type=swizzle_type)
    return qdata, scales


_NVIDIA_SM100_OR_LATER = torch.version.rocm is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference, name="reference"),
    subtest(F.quantize_tensor, name="public", decorators=[skipIfNoCuteDSL]),
)
_MXFP8_KWARGS = {
    "qdata_dtype": torch.float8_e4m3fn,
    "scaling_algorithm": F.ScalingAlgorithm.MXFP_E8M0_RU,
    "scaling_type": F.ScalingType.BlockWise1x32,
}
_MXFP8_NO_SWIZZLE_KWARGS = {
    **_MXFP8_KWARGS,
    "swizzle_type": SwizzleType.NO_SWIZZLE,
}


def _make_mxfp8_sr_gold_input(device):
    M, K = 96, 160
    values = (torch.arange(M * K, device=device, dtype=torch.int32) % 63).float()
    input = ((values - 31) / 32).to(torch.bfloat16).reshape(M, K)
    input[0, :8] = torch.tensor(
        [448, -448, 0, -0.0, 2**-9, -(2**-9), 2**-10, -(2**-10)],
        device=device,
        dtype=torch.bfloat16,
    )
    input[1, 1] = float("nan")
    input[2, 2] = float("inf")
    input[3, 3] = -float("inf")
    return input


class TestMXFP8ReferenceNumerics(TestCase):
    def test_f32_to_e8m0_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L113
        values, expected = make_f32_to_e8m0_rceil_cases(device=device)
        self.assertEqual(_f32_to_e8m0_rceil(values), expected.to(device))

    @parametrize("quantize_fn", _MXFP8_IMPLEMENTATIONS)
    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_mxfp8_corner_case_bytes(self, quantize_fn, input_dtype, device):
        if quantize_fn is F.quantize_tensor and not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, device=device)
        qdata, scales = quantize_fn(cases.inputs, **_MXFP8_NO_SWIZZLE_KWARGS)
        assert_mxfp8_semantics(qdata, scales, cases)

    @parametrize(
        "swizzle_type",
        (SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4),
        name_fn=lambda swizzle_type: swizzle_type.name,
    )
    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        (
            (1, 32),
            (31, 96),
            (128, 64),
            (128, 128),
            (129, 160),
            (256, 256),
            (1024, 1152),
            (2080, 2048),
            (2048, 4224),
        ),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_randn_sqnr(self, swizzle_type, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        qdata, scales = F.quantize_tensor(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(qdata.shape, data_hp.shape)
        expected_scale_shape = (
            ((shape[0] + 127) // 128, (shape[1] + 127) // 128, 32, 16)
            if swizzled
            else (shape[0], shape[1] // 32)
        )
        self.assertEqual(tuple(scales.shape), expected_scale_shape)
        if swizzled:
            scale_bytes = scales.view(torch.uint8).flatten()
            self.assertEqual(scale_bytes, scales_ref.view(torch.uint8))
            scales = from_blocked(qdata, scales.flatten(), 32)
        else:
            self.assertEqual(scales.view(torch.uint8), scales_ref.view(torch.uint8))
        dequantized = from_blocked_format(qdata, scales).float()
        sqnr = compute_error(data_hp.float(), dequantized)
        self.assertGreater(sqnr.item(), 18.0)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 16), (64, 128), (160, 144), (2048, 2080)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_m(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata, scales = F.quantize_tensor(
            data_hp.t(), **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(tuple(qdata.shape), (shape[1], shape[0]))
        self.assertEqual(
            tuple(scales.shape),
            ((shape[1] + 127) // 128, (shape[0] + 127) // 128, 32, 16),
        )
        scale_bytes = scales.view(torch.uint8).flatten()
        self.assertEqual(scale_bytes, scales_ref.view(torch.uint8))

    @parametrize("input_dtype", (torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32896, 128), (11008, 384)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_m_large_k_tile_boundary(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(*shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata, scales = F.quantize_tensor(
            data_hp.t(), **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 32), (64, 128), (160, 160), (2048, 2080)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_km(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata_k.view(torch.uint8), qdata_k_ref.view(torch.uint8))
        self.assertEqual(qdata_m.view(torch.uint8), qdata_m_ref.view(torch.uint8))
        scale_k_bytes = scales_k.view(torch.uint8).flatten()
        scale_m_bytes = scales_m.view(torch.uint8).flatten()
        self.assertEqual(scale_k_bytes, scales_k_ref.view(torch.uint8))
        self.assertEqual(scale_m_bytes, scales_m_ref.view(torch.uint8))

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 32), (64, 128), (160, 160), (2048, 4224)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            tuple(scales.shape),
            ((shape[0] + 127) // 128, (shape[1] + 127) // 128, 32, 16),
        )
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square_nan(self, input_dtype, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 64, device=device, dtype=input_dtype)
        data_hp[0, 0] = float("nan")
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )
        expected_nan = torch.zeros((64, 64), device=device, dtype=torch.bool)
        expected_nan[:32, :32] = True
        self.assertEqual(torch.isnan(qdata), expected_nan)

        scale_bytes = from_blocked(qdata, scales.flatten(), 32).view(torch.uint8)
        expected_nan_scale = torch.zeros((64, 2), device=device, dtype=torch.bool)
        expected_nan_scale[:32, 0] = True
        self.assertEqual(scale_bytes == 255, expected_nan_scale)

    @parametrize(
        "quant_orientation,is_square_scaling,is_scale_swizzled",
        (
            subtest(("dim_k", False, False), name="dim_k_compact"),
            subtest(("dim_k", False, True), name="dim_k_swizzled"),
            subtest(("dim_k", True, True), name="dim_k_square"),
            subtest(("dim_m", False, True), name="dim_m"),
        ),
    )
    @parametrize(
        "shape",
        ((0, 32), (32, 0), (0, 0)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_empty(
        self, quant_orientation, is_square_scaling, is_scale_swizzled, shape, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape
        scale_k_shape = (
            ((M + 127) // 128, (K + 127) // 128, 32, 16)
            if is_scale_swizzled
            else (M, K // 32)
        )
        scale_m_shape = ((K + 127) // 128, (M + 127) // 128, 32, 16)
        expected_shapes = {
            "dim_k": ((M, K), scale_k_shape),
            "dim_m": ((K, M), scale_m_shape),
        }[quant_orientation]
        api_input = data_hp.t() if quant_orientation == "dim_m" else data_hp
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
        )
        outputs = F.quantize_tensor(
            api_input,
            **_MXFP8_KWARGS,
            swizzle_type=swizzle_type,
            scaling_type_use_square_block_size=is_square_scaling,
        )
        self.assertEqual(len(outputs), len(expected_shapes))
        for index, (output, expected_shape) in enumerate(
            zip(outputs, expected_shapes, strict=True)
        ):
            self.assertEqual(tuple(output.shape), expected_shape)
            expected_dtype = (
                torch.float8_e4m3fn if index % 2 == 0 else torch.float8_e8m0fnu
            )
            self.assertEqual(output.dtype, expected_dtype)
            self.assertEqual(output.device, data_hp.device)
            self.assertEqual(output.numel(), 0)

    @parametrize(
        "shape",
        ((0, 32), (32, 0), (0, 0)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_km_empty(self, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        M, K = shape
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        outputs = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        expected_shapes = (
            (M, K),
            ((M + 127) // 128, (K + 127) // 128, 32, 16),
            (K, M),
            ((K + 127) // 128, (M + 127) // 128, 32, 16),
        )
        for index, (output, expected_shape) in enumerate(
            zip(outputs, expected_shapes, strict=True)
        ):
            self.assertEqual(tuple(output.shape), expected_shape)
            self.assertEqual(
                output.dtype,
                torch.float8_e4m3fn if index % 2 == 0 else torch.float8_e8m0fnu,
            )
            self.assertEqual(output.numel(), 0)

    @skipIfNoCuteDSL
    def test_quantize_tensor_invalid_configuration(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.ones((64, 64), dtype=torch.bfloat16, device=device)
        with self.assertRaisesRegex(ValueError, "SWIZZLE_32_4_4"):
            F.quantize_tensor(data.t(), **_MXFP8_NO_SWIZZLE_KWARGS)
        with self.assertRaisesRegex(ValueError, "32x32 MXFP8 scaling"):
            F.quantize_tensor(
                data,
                **_MXFP8_NO_SWIZZLE_KWARGS,
                scaling_type_use_square_block_size=True,
            )
        with self.assertRaisesRegex(ValueError, "32x32 MXFP8 scaling"):
            F.quantize_tensor(
                data.t(),
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                scaling_type_use_square_block_size=True,
            )
        with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
            F.quantize_tensor(
                data, **(_MXFP8_NO_SWIZZLE_KWARGS | {"qdata_dtype": torch.float16})
            )
        with self.assertRaisesRegex(ValueError, "MXFP_E8M0_RU"):
            F.quantize_tensor(
                data, **(_MXFP8_NO_SWIZZLE_KWARGS | {"scaling_algorithm": 1})
            )
        with self.assertRaisesRegex(ValueError, "BlockWise1x32"):
            F.quantize_tensor(
                data,
                **(
                    _MXFP8_NO_SWIZZLE_KWARGS
                    | {"scaling_type": F.ScalingType.BlockWise1x16}
                ),
            )
        with self.assertRaisesRegex(ValueError, "transpose of contiguous"):
            F.quantize_tensor(data[:, ::2], **_MXFP8_NO_SWIZZLE_KWARGS)
        with self.assertRaisesRegex(ValueError, "columns divisible by 32"):
            F.quantize_tensor(data.new_ones(64, 48), **_MXFP8_NO_SWIZZLE_KWARGS)

    @skipIfNoCuteDSL
    def test_quantize_tensor_requires_grad(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.ones(
            (64, 64), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        qdata, _ = F.quantize_tensor(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertTrue(qdata.requires_grad)
        with self.assertRaisesRegex(
            RuntimeError, "derivative for aten::_quantize_tensor is not implemented"
        ):
            qdata.float().sum().backward()
        with torch.no_grad():
            qdata, scales = F.quantize_tensor(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertFalse(qdata.requires_grad)
        self.assertFalse(scales.requires_grad)

    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_invalid_configuration(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.ones((64, 96), dtype=torch.bfloat16, device=device)
        kwargs = {
            **_MXFP8_KWARGS,
            "swizzle_type": SwizzleType.SWIZZLE_32_4_4,
        }
        with self.assertRaisesRegex(ValueError, "SWIZZLE_32_4_4"):
            F.quantize_tensor_dual(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        with self.assertRaisesRegex(ValueError, "32x32 MXFP8 scaling"):
            F.quantize_tensor_dual(
                data, **kwargs, scaling_type_use_square_block_size=True
            )
        with self.assertRaisesRegex(ValueError, "contiguous input"):
            F.quantize_tensor_dual(data.t(), **kwargs)
        with self.assertRaisesRegex(ValueError, "both dimensions divisible by 32"):
            F.quantize_tensor_dual(data[:, :48], **kwargs)
        with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
            F.quantize_tensor_dual(data, **(kwargs | {"qdata_dtype": torch.float16}))

    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_requires_grad(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.ones(
            (64, 96), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        kwargs = {
            **_MXFP8_KWARGS,
            "swizzle_type": SwizzleType.SWIZZLE_32_4_4,
        }
        outputs = F.quantize_tensor_dual(data, **kwargs)
        self.assertTrue(outputs[0].requires_grad)
        with self.assertRaisesRegex(
            RuntimeError,
            "derivative for aten::_quantize_tensor_dual is not implemented",
        ):
            outputs[0].float().sum().backward()
        with torch.no_grad():
            outputs = F.quantize_tensor_dual(data, **kwargs)
        for output in outputs:
            self.assertFalse(output.requires_grad)

    @skipIfNoCuteDSL
    def test_quantize_tensor_dispatch_transposed(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((64, 96), dtype=torch.float16, device=device).t()
        qdata, scales = torch._quantize_tensor(  # pyrefly: ignore[missing-attribute]
            input,
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        qdata_ref, scales_ref = F.quantize_tensor(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales.view(torch.uint8), scales_ref.view(torch.uint8))

    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_dispatch(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((64, 96), dtype=torch.float16, device=device)
        actual = torch.ops.aten._quantize_tensor_dual.default(
            input,
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        expected = F.quantize_tensor_dual(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))

    @parametrize("transposed,square", ((False, False), (True, False), (False, True)))
    @skipIfNoCuteDSL
    def test_quantize_tensor_compile(self, transposed, square, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.randn((160, 160), dtype=torch.bfloat16, device=device)
        api_input = data.t() if transposed else data

        def fn(x):
            return F.quantize_tensor(
                x,
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                scaling_type_use_square_block_size=square,
            )

        expected = fn(api_input)
        actual = torch.compile(fn, fullgraph=True)(api_input)
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))

    @skipIfNoCuteDSL
    def test_quantize_tensor_scaled_mm(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        mat_a = torch.randn((128, 256), dtype=torch.bfloat16, device=device)
        mat_b = torch.randn((256, 192), dtype=torch.bfloat16, device=device)
        swizzle = SwizzleType.SWIZZLE_32_4_4
        qa, sa = F.quantize_tensor(mat_a, **_MXFP8_KWARGS, swizzle_type=swizzle)
        qb, sb = F.quantize_tensor(mat_b.t(), **_MXFP8_KWARGS, swizzle_type=swizzle)
        out = F.scaled_mm(
            qa,
            qb.t(),
            sa,
            F.ScalingType.BlockWise1x32,
            sb,
            F.ScalingType.BlockWise1x32,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
        )
        self.assertEqual(tuple(out.shape), (128, 192))
        self.assertGreater(
            compute_error(mat_a.float() @ mat_b.float(), out.float()).item(), 15.0
        )

    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_compile(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((160, 192), dtype=torch.bfloat16, device=device)

        def fn(x):
            return F.quantize_tensor_dual(
                x, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
            )

        expected = fn(input)
        actual = torch.compile(fn, fullgraph=True)(input)
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))


@instantiate_parametrized_tests
class TestQuantizeTensorMeta(TestCase):
    @parametrize(
        "quantize_fn,kwargs",
        (
            subtest((F.quantize_tensor, _MXFP8_NO_SWIZZLE_KWARGS), name="single"),
            subtest(
                (
                    F.quantize_tensor_dual,
                    {**_MXFP8_KWARGS, "swizzle_type": SwizzleType.SWIZZLE_32_4_4},
                ),
                name="dual",
            ),
        ),
    )
    def test_meta_validates_arguments(self, quantize_fn, kwargs):
        input = torch.empty((32, 32), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "MXFP_E8M0_RU"):
            quantize_fn(input, **(kwargs | {"scaling_algorithm": 1}))
        with self.assertRaisesRegex(ValueError, "2D"):
            quantize_fn(input.flatten(), **kwargs)
        input.requires_grad_()
        outputs = quantize_fn(input, **kwargs)
        self.assertTrue(all(output.requires_grad for output in outputs))

    def test_quantize_tensor_argument_names(self):
        self.assertEqual(F.ScalingAlgorithm.MXFP_E8M0_RU.value, 0)
        public_names = tuple(inspect.signature(F.quantize_tensor).parameters)
        dual_names = tuple(inspect.signature(F.quantize_tensor_dual).parameters)
        native = torch.ops.aten._quantize_tensor.default._schema.arguments
        native_dual = torch.ops.aten._quantize_tensor_dual.default._schema.arguments
        self.assertEqual(
            public_names,
            (
                "input",
                "scaling_type",
                "qdata_dtype",
                "scaling_algorithm",
                "swizzle_type",
                "scaling_type_use_square_block_size",
            ),
        )
        self.assertEqual(public_names, dual_names)
        native_names = tuple(arg.name for arg in native)
        self.assertEqual(native_names, tuple(arg.name for arg in native_dual))
        self.assertEqual(public_names, native_names)
        for fn in (F.quantize_tensor, F.quantize_tensor_dual):
            params = tuple(inspect.signature(fn).parameters.values())
            self.assertEqual(params[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for param in params[1:]:
                self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(params[4].default, inspect.Parameter.empty)
        for schema_args in (native, native_dual):
            self.assertEqual(str(schema_args[3].type), "int")
            self.assertFalse(schema_args[0].kwarg_only)
            for arg in schema_args[1:]:
                self.assertTrue(arg.kwarg_only)
            self.assertFalse(schema_args[4].has_default_value())

    @parametrize(
        "shape,transposed,square,swizzle",
        (
            ((160, 160), False, False, SwizzleType.NO_SWIZZLE),
            ((160, 160), False, False, SwizzleType.SWIZZLE_32_4_4),
            ((160, 160), True, False, SwizzleType.SWIZZLE_32_4_4),
            ((160, 160), False, True, SwizzleType.SWIZZLE_32_4_4),
            ((0, 32), False, False, SwizzleType.NO_SWIZZLE),
        ),
    )
    def test_meta_shapes(self, shape, transposed, square, swizzle):
        source = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        api_input = source.t() if transposed else source
        qdata, scales = F.quantize_tensor(
            api_input,
            **_MXFP8_KWARGS,
            swizzle_type=swizzle,
            scaling_type_use_square_block_size=square,
        )
        rows, cols = api_input.shape
        expected_scale_shape = (
            ((rows + 127) // 128, (cols + 127) // 128, 32, 16)
            if swizzle == SwizzleType.SWIZZLE_32_4_4
            else (rows, cols // 32)
        )
        self.assertEqual(tuple(qdata.shape), (rows, cols))
        self.assertEqual(tuple(scales.shape), expected_scale_shape)
        self.assertEqual(qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)

    def test_fake_cuda_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        if not torch.backends.cuda.is_built():
            self.skipTest("requires a CUDA build")
        with FakeTensorMode():
            source = torch.empty((160, 160), dtype=torch.float16, device="cuda")
            qdata, scales = F.quantize_tensor(
                source.t(),
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            )
            self.assertEqual(tuple(qdata.shape), (160, 160))
            self.assertEqual(tuple(scales.shape), (2, 2, 32, 16))
            dual_input = torch.empty((160, 288), dtype=torch.float16, device="cuda")
            qk, sk, qm, sm = F.quantize_tensor_dual(
                dual_input,
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            )
            self.assertEqual(tuple(qk.shape), (160, 288))
            self.assertEqual(tuple(sk.shape), (2, 3, 32, 16))
            self.assertEqual(tuple(qm.shape), (288, 160))
            self.assertEqual(tuple(sm.shape), (3, 2, 32, 16))

    def test_dispatch_transposed_shape(self):
        input = torch.empty((32, 160), dtype=torch.float16, device="meta").t()
        qdata, scales = torch._quantize_tensor(  # pyrefly: ignore[missing-attribute]
            input,
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        self.assertEqual(tuple(qdata.shape), (160, 32))
        self.assertEqual(tuple(scales.shape), (2, 1, 32, 16))

    def test_dual_meta_shapes(self):
        input = torch.empty((160, 288), dtype=torch.bfloat16, device="meta")
        outputs = F.quantize_tensor_dual(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(
            tuple(tuple(output.shape) for output in outputs),
            ((160, 288), (2, 3, 32, 16), (288, 160), (3, 2, 32, 16)),
        )
        self.assertEqual(
            tuple(output.dtype for output in outputs),
            (
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
            ),
        )


class TestMXFP8StochasticReferenceNumerics(TestCase):
    @parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("swizzle_type", (SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4))
    def test_stateless_round_trip(self, input_dtype, swizzle_type, device):
        input = torch.randn((96, 160), device=device, dtype=input_dtype)
        key = prng.key(7, device=device)
        scales, qdata = to_mxfp8_reference(
            input, swizzle_type=swizzle_type, rounding_mode="stochastic", random_key=key
        )
        unswizzled = (
            from_blocked(qdata, scales, 32)
            if swizzle_type == SwizzleType.SWIZZLE_32_4_4
            else scales
        )
        reconstructed = from_blocked_format(qdata, unswizzled)
        self.assertGreater(
            compute_error(input.float(), reconstructed.float()).item(), 15.0
        )

    def test_stateless_nonfinite_groups(self, device):
        input = torch.zeros((32, 32), device=device, dtype=torch.bfloat16)
        input[0, 0] = float("nan")
        input[1, 1] = float("inf")
        input[2, 2] = -float("inf")
        scales, qdata = to_mxfp8_reference(
            input, rounding_mode="stochastic", random_key=prng.key(7, device=device)
        )
        qbytes = qdata.view(torch.uint8)
        self.assertEqual(
            qbytes[:3], torch.full((3, 32), 0x7F, device=device, dtype=torch.uint8)
        )
        self.assertEqual(
            qbytes[3:], torch.zeros((29, 32), device=device, dtype=torch.uint8)
        )
        self.assertEqual(
            scales.view(torch.uint8)[:3],
            torch.full((3, 1), 0xFF, device=device, dtype=torch.uint8),
        )

    @parametrize("num_ops", (1, 2))
    def test_stateful_cuda_graph(self, num_ops, device):
        # verifies that:
        # * cuda graph capture + replay matches eager
        # * ^ holds for chains of 1 to 2 to_mxfp8_reference ops
        if torch.device(device).type != "cuda" or torch.version.rocm is not None:
            self.skipTest("stateful NVIDIA Philox rounding requires CUDA")
        input = torch.randn((96, 160), device=device, dtype=torch.bfloat16)
        generator = torch.cuda.default_generators[input.get_device()]
        with torch.random.fork_rng(devices=[input.get_device()]):
            torch.manual_seed(123)
            expected = [
                to_mxfp8_reference(input, rounding_mode="stochastic")
                for _ in range(2 * num_ops)
            ]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = [
                    to_mxfp8_reference(input, rounding_mode="stochastic")
                    for _ in range(num_ops)
                ]

            torch.manual_seed(123)
            replays = []
            for replay in range(2):
                offset = generator.get_offset()
                graph.replay()
                # verify offset advanced correctly
                self.assertEqual(generator.get_offset(), offset + 4 * num_ops)
                outputs = []
                for op, (scales, qdata) in enumerate(captured):
                    # verify results from cuda graph match eager
                    scale_bytes = scales.view(torch.uint8).clone()
                    qdata_bytes = qdata.view(torch.uint8).clone()
                    expected_scales, expected_qdata = expected[replay * num_ops + op]
                    self.assertEqual(scale_bytes, expected_scales.view(torch.uint8))
                    self.assertEqual(qdata_bytes, expected_qdata.view(torch.uint8))
                    reconstructed = from_blocked_format(qdata, scales)
                    self.assertGreater(
                        compute_error(input.float(), reconstructed.float()).item(), 15.0
                    )
                    outputs.append((scale_bytes, qdata_bytes))
                replays.append(outputs)

            for op in range(num_ops):
                self.assertEqual(replays[0][op][0], replays[1][op][0])
                self.assertFalse(torch.equal(replays[0][op][1], replays[1][op][1]))
            if num_ops == 2:
                self.assertFalse(torch.equal(replays[0][0][1], replays[0][1][1]))

    def test_stateless_cuda_graph(self, device):
        # verifies that:
        # * cuda graph replay of to_mxfp8_reference with unchanged random_key
        #   leads to results bitwise equivalent to original
        # * cuda graph replay of to_mxfp8_reference with changed random_key
        #   leads to a fresh random draw + different (and still valid) results

        if torch.device(device).type != "cuda" or torch.version.rocm is not None:
            self.skipTest("NVIDIA Philox rounding requires CUDA")
        input = torch.randn((96, 160), device=device, dtype=torch.bfloat16)
        original_key = prng.key(7, device=device)
        changed_key = prng.fold_in(original_key, 1)
        expected = [
            to_mxfp8_reference(input, rounding_mode="stochastic", random_key=trial_key)
            for trial_key in (original_key, changed_key)
        ]

        # record the CUDA graph
        key_for_cuda_graph = original_key.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scales, qdata = to_mxfp8_reference(
                input, rounding_mode="stochastic", random_key=key_for_cuda_graph
            )

        # fetch global RNG (to make sure later that it does not change)
        generator = torch.cuda.default_generators[input.get_device()]
        offset = generator.get_offset()

        replays = []
        for trial_key, (expected_scales, expected_qdata) in (
            (original_key, expected[0]),
            (original_key, expected[0]),
            (changed_key, expected[1]),
            (original_key, expected[0]),
        ):
            key_for_cuda_graph.copy_(trial_key)
            graph.replay()
            # verify RNG state did not change
            self.assertEqual(generator.get_offset(), offset)
            # verify results from cuda graph match expected result from eager mode
            scale_bytes = scales.view(torch.uint8).clone()
            qdata_bytes = qdata.view(torch.uint8).clone()
            self.assertEqual(scale_bytes, expected_scales.view(torch.uint8))
            self.assertEqual(qdata_bytes, expected_qdata.view(torch.uint8))
            reconstructed = from_blocked_format(qdata, scales)
            self.assertGreater(
                compute_error(input.float(), reconstructed.float()).item(), 15.0
            )
            replays.append((scale_bytes, qdata_bytes))

        self.assertEqual(replays[0], replays[1])
        self.assertFalse(torch.equal(replays[0][1], replays[2][1]))
        self.assertEqual(replays[0], replays[3])


class TestFP8StochasticRounding(TestCase):
    def test_exact(self, device):
        key = prng.fold_in(prng.key(7, device=device), 12345)
        words = prng.bits(key, 4, dtype=torch.uint32)

        # verify that SR does not adjust values that are exactly representable
        # in float8_e4m3fn
        exact = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1.125,
                -1.125,
                2**-9,
                -(2**-9),
                2**-6,
                -(2**-6),
                1.875,
                2.0,
                416.0,
                448.0,
                -448.0,
                0.0,
            ],
            device=device,
        )
        rounded = _f32_to_fp8_nvidia_sr_with_words(exact, words)
        expected_exact = exact.to(torch.float8_e4m3fn).view(torch.uint8)
        self.assertEqual(rounded.view(torch.uint8), expected_exact)
        self.assertEqual(
            rounded[:2].view(torch.uint8),
            torch.tensor([0x00, 0x80], device=device, dtype=torch.uint8),
        )

    def test_neighbors(self, device):
        # verify that tensor values between the representable values of
        # float8_e4m3fn get rounded to a neighboring representable value
        intervals = torch.tensor(
            [
                [1.0, 1.125],
                [-1.125, -1.0],
                [1.875, 2.0],
                [416.0, 448.0],
                [0.0, 2**-9],
                [7 * 2**-9, 2**-6],
                [-(2**-9), -0.0],
                [-(2**-6), -(7 * 2**-9)],
            ],
            device=device,
        )

        key = prng.fold_in(prng.key(7, device=device), 12345)
        for i in range(10):
            key = prng.fold_in(key, i)
            words = prng.bits(key, 4, dtype=torch.uint32)
            ratio = torch.rand_like(intervals[:, 0])
            between = intervals[:, 0] + ratio * (intervals[:, 1] - intervals[:, 0])
            rounded = _f32_to_fp8_nvidia_sr_with_words(between, words[:2])
            rounded = rounded.view(torch.uint8)
            endpoints = intervals.to(torch.float8_e4m3fn).view(torch.uint8)
            neighbors = (rounded == endpoints[:, 0]) | (rounded == endpoints[:, 1])
            self.assertEqual(neighbors, torch.ones_like(neighbors))

    def test_special_values(self, device):
        key = prng.fold_in(prng.key(7, device=device), 12345)
        words = prng.bits(key, 4, dtype=torch.uint32)

        # verify that out-of-range values saturate, and NaN gets converted to NaN
        inf = float("inf")
        nan = float("nan")
        nonfinite_or_overflow = torch.tensor(
            [449.0, -449.0, 1e6, -1e6, inf, -inf, nan, -nan],
            device=device,
        )
        rounded = _f32_to_fp8_nvidia_sr_with_words(nonfinite_or_overflow, words[:2])
        expected_finite = torch.tensor([448.0, -448.0] * 3, device=device)
        self.assertEqual(rounded[:6].float(), expected_finite)
        self.assertEqual(
            rounded[6:].view(torch.uint8),
            torch.tensor([0x7F, 0x7F], device=device, dtype=torch.uint8),
        )

    @parametrize(
        "lower,upper",
        (
            # hand chosen pairs of neighboring values exactly representable in
            # float8_e4m3fn
            (1.0, 1.125),
            (-1.125, -1.0),
            (1.875, 2.0),
            (416.0, 448.0),
            (0.0, 2**-9),
            (-(2**-9), -0.0),
            (7 * 2**-9, 2**-6),
        ),
    )
    def test_round_up_probability(self, lower, upper, device):
        # given a value x between A and B, verifies that SR rounds
        # x up to B with probability `P = (x - A) / (B - A)`, and rounds
        # x down to A with probability `1 - P`

        probabilities = torch.tensor((0.125, 0.5, 0.875), device=device)
        values = lower + (upper - lower) * probabilities
        samples = values[:, None].expand(-1, 8192).contiguous()
        key = prng.fold_in(prng.key(7, device=device), 12345)
        words = prng.bits(key, samples.numel() // 4, dtype=torch.uint32)
        rounded = _f32_to_fp8_nvidia_sr_with_words(samples, words).float()
        neighbors = (rounded == lower) | (rounded == upper)
        self.assertEqual(neighbors, torch.ones_like(neighbors))

        observed = (rounded == upper).float().mean(dim=1)
        tolerance = 6 * torch.sqrt(probabilities * (1 - probabilities) / 8192)
        for probability, frequency, bound in zip(probabilities, observed, tolerance):
            self.assertLessEqual(
                abs(frequency.item() - probability.item()),
                bound.item(),
                f"[{lower}, {upper}]: expected round-up probability {probability.item()}, got {frequency.item()}",
            )

    def test_mean_preservation(self, device):
        # test that E(SR(X)) ~= X

        generator = torch.Generator(device=device).manual_seed(1234)
        values = torch.randn(256, device=device, generator=generator)
        key = prng.fold_in(prng.key(7, device=device), 12345)
        rounded_trials = []
        for trial in range(256):
            words = prng.bits(
                prng.fold_in(key, trial), values.numel() // 4, dtype=torch.uint32
            )
            rounded_trials.append(
                _f32_to_fp8_nvidia_sr_with_words(values, words).float()
            )
        self.assertEqual(
            torch.stack(rounded_trials).mean(dim=0), values, rtol=0.03, atol=0.001
        )


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")
instantiate_device_type_tests(
    TestMXFP8StochasticReferenceNumerics, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestFP8StochasticRounding, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
