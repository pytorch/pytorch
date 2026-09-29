# Owner(s): ["module: linear algebra"]

import inspect
from functools import cache

import torch
from torch.nn import functional as F
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import (
    _f32_to_e8m0_rceil,
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
    inner_scale_calc: F.InnerScaleCalc,
    scaling_type: F.ScalingType,
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
):
    if (
        qdata_dtype != torch.float8_e4m3fn
        or inner_scale_calc != F.InnerScaleCalc.RCEIL_E8M0
        or scaling_type != F.ScalingType.BlockWise1x32
    ):
        raise ValueError("unsupported MXFP8 reference recipe")
    scales, qdata = to_mxfp8_reference(input, format="mxfp8", swizzle_type=swizzle_type)
    return qdata, scales


@cache
def _nvidia_sm100_or_newer(device: str) -> bool:
    if torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(device) >= (10, 0)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference, name="reference"),
    subtest(F.quantize_tensor, name="public"),
)
_MXFP8_KWARGS = {
    "qdata_dtype": torch.float8_e4m3fn,
    "inner_scale_calc": F.InnerScaleCalc.RCEIL_E8M0,
    "scaling_type": F.ScalingType.BlockWise1x32,
}
_MXFP8_NO_SWIZZLE_KWARGS = {
    **_MXFP8_KWARGS,
    "swizzle_type": SwizzleType.NO_SWIZZLE,
}


class TestMXFP8ReferenceNumerics(TestCase):
    def test_f32_to_e8m0_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L113
        values, expected = make_f32_to_e8m0_rceil_cases(device=device)
        self.assertEqual(_f32_to_e8m0_rceil(values), expected.to(device))

    @parametrize("quantize_fn", _MXFP8_IMPLEMENTATIONS)
    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_mxfp8_corner_case_bytes(self, quantize_fn, input_dtype, device):
        if quantize_fn is F.quantize_tensor and not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, "rceil", device=device)
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
    def test_to_mx_rceil_randn_sqnr(self, swizzle_type, input_dtype, shape, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        original = data_hp.float()
        ref_scales = from_blocked(qdata_ref, scales_ref, 32) if swizzled else scales_ref
        dequantized_ref = from_blocked_format(qdata_ref, ref_scales).float()
        self.assertGreater(compute_error(original, dequantized_ref).item(), 18.0)
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
        sqnr = compute_error(original, dequantized)
        self.assertGreater(sqnr.item(), 18.0)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 16), (64, 128), (160, 144), (2048, 2080)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    def test_to_mx_rceil_dim_m(self, input_dtype, shape, device):
        if not _nvidia_sm100_or_newer(device):
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
    def test_to_mx_rceil_dim_m_large_k_tile_boundary(self, input_dtype, shape, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.ones(shape, device=device, dtype=input_dtype)
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
    def test_to_mx_rceil_dim_km(self, input_dtype, shape, device):
        if not _nvidia_sm100_or_newer(device):
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
    def test_to_mx_rceil_dim_k_square(self, input_dtype, shape, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_square_block_and_expand=True,
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
    def test_to_mx_rceil_dim_k_square_nan(self, input_dtype, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.ones((32, 32), device=device, dtype=input_dtype)
        data_hp[0, 0] = float("nan")
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        self.assertTrue(torch.isnan(qdata_ref).all())
        self.assertEqual(scales_ref.view(torch.uint8)[0], 255)

        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_square_block_and_expand=True,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )

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
    def test_to_mx_rceil_empty(
        self, quant_orientation, is_square_scaling, is_scale_swizzled, shape, device
    ):
        if not _nvidia_sm100_or_newer(device):
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
            scaling_type_square_block_and_expand=is_square_scaling,
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
    def test_to_mx_rceil_dim_km_empty(self, shape, device):
        if not _nvidia_sm100_or_newer(device):
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

    def test_quantize_tensor_invalid_configuration(self, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.ones((64, 64), dtype=torch.bfloat16, device=device)
        with self.assertRaisesRegex(ValueError, "SWIZZLE_32_4_4"):
            F.quantize_tensor(data.t(), **_MXFP8_NO_SWIZZLE_KWARGS)
        with self.assertRaisesRegex(ValueError, "32x32 MXFP8 scaling"):
            F.quantize_tensor(
                data,
                **_MXFP8_NO_SWIZZLE_KWARGS,
                scaling_type_square_block_and_expand=True,
            )
        with self.assertRaisesRegex(ValueError, "32x32 MXFP8 scaling"):
            F.quantize_tensor(
                data.t(),
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                scaling_type_square_block_and_expand=True,
            )
        with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
            F.quantize_tensor(
                data, **(_MXFP8_NO_SWIZZLE_KWARGS | {"qdata_dtype": torch.float16})
            )
        with self.assertRaisesRegex(ValueError, "RCEIL_E8M0"):
            F.quantize_tensor(
                data, **(_MXFP8_NO_SWIZZLE_KWARGS | {"inner_scale_calc": 1})
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

    def test_quantize_tensor_requires_grad(self, device):
        data = torch.ones(
            (64, 64), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, "does not support autograd"):
            F.quantize_tensor(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        if _nvidia_sm100_or_newer(device):
            with torch.no_grad():
                qdata, scales = F.quantize_tensor(data, **_MXFP8_NO_SWIZZLE_KWARGS)
            self.assertFalse(qdata.requires_grad)
            self.assertFalse(scales.requires_grad)

    def test_quantize_tensor_dual_invalid_configuration(self, device):
        if not _nvidia_sm100_or_newer(device):
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
                data, **kwargs, scaling_type_square_block_and_expand=True
            )
        with self.assertRaisesRegex(ValueError, "contiguous input"):
            F.quantize_tensor_dual(data.t(), **kwargs)
        with self.assertRaisesRegex(ValueError, "both dimensions divisible by 32"):
            F.quantize_tensor_dual(data[:, :48], **kwargs)
        with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
            F.quantize_tensor_dual(data, **(kwargs | {"qdata_dtype": torch.float16}))

    def test_quantize_tensor_dual_requires_grad(self, device):
        data = torch.ones(
            (64, 96), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        kwargs = {
            **_MXFP8_KWARGS,
            "swizzle_type": SwizzleType.SWIZZLE_32_4_4,
        }
        with self.assertRaisesRegex(RuntimeError, "does not support autograd"):
            F.quantize_tensor_dual(data, **kwargs)
        if _nvidia_sm100_or_newer(device):
            with torch.no_grad():
                outputs = F.quantize_tensor_dual(data, **kwargs)
            for output in outputs:
                self.assertFalse(output.requires_grad)

    def test_quantize_tensor_dispatch_transposed(self, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((64, 96), dtype=torch.float16, device=device).t()
        qdata, scales = torch._quantize_tensor(  # pyrefly: ignore[missing-attribute]
            input,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=F.InnerScaleCalc.RCEIL_E8M0.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        qdata_ref, scales_ref = F.quantize_tensor(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales.view(torch.uint8), scales_ref.view(torch.uint8))

    def test_quantize_tensor_dual_dispatch(self, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((64, 96), dtype=torch.float16, device=device)
        actual = torch.ops.aten._quantize_tensor_dual.default(
            input,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=F.InnerScaleCalc.RCEIL_E8M0.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        expected = F.quantize_tensor_dual(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))

    @parametrize("transposed,square", ((False, False), (True, False), (False, True)))
    def test_quantize_tensor_compile(self, transposed, square, device):
        if not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.randn((160, 160), dtype=torch.bfloat16, device=device)
        api_input = data.t() if transposed else data

        def fn(x):
            return F.quantize_tensor(
                x,
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                scaling_type_square_block_and_expand=square,
            )

        expected = fn(api_input)
        actual = torch.compile(fn, fullgraph=True)(api_input)
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))

    def test_quantize_tensor_scaled_mm(self, device):
        if not _nvidia_sm100_or_newer(device):
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

    def test_quantize_tensor_dual_compile(self, device):
        if not _nvidia_sm100_or_newer(device):
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

    @parametrize("quantize_fn", _MXFP8_IMPLEMENTATIONS)
    def test_to_mx_rceil(self, quantize_fn, device):
        if quantize_fn is F.quantize_tensor and not _nvidia_sm100_or_newer(device):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L276
        # TODO(future PR): refactor below to make it look more like
        # `test_mxfp8_corner_case_bytes`

        # nan
        # fmt: off
        data_hp = torch.tensor(
            [
            2143289344, 1054459450, 1060527345, 1045656552, 1058239340, 1045057552, 1061158006, 1049626606,
            1052757568, 1032293288, 1056992320, 1064929425, 1061036255, 1047450552, 1057077424, 1055125012,
            1036491424, 1063542041, 1057099838, 1058731224, 1050189482, 1049114228, 1058347802, 1060065968,
            1058846156, 1048878912, 1065109089, 1054494928, 1044803976, 1049117692, 1065222528, 1056965012,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)

        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertTrue(torch.isnan(scales))
        # When any element in block is NaN, entire quantized block becomes NaN
        self.assertTrue(torch.all(torch.isnan(qdata)))
        # fp32 denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            6142315, 5096174, 3345704, 6178415, 5728750, 419002, 1716691, 4335089,
            5785800, 6234845, 1697524, 33075, 3975816, 3714822, 5411407, 3040844,
            7400945, 4474166, 7257182, 1273750, 5872176, 4694081, 2096530, 6273621,
            67028, 7585260, 4532315, 4599275, 6133942, 4542483, 5992199, 6862780,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # E8M0 byte 0 is 2^-127, so these FP32 subnormals remain representable
        # after scaling instead of being flushed to zero.
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            60, 58, 53, 60, 59, 29, 45, 56,
            59, 60, 45, 4, 55, 54, 58, 52,
            62, 57, 62, 42, 59, 57, 48, 60,
            8, 62, 57, 57, 60, 57, 59, 61,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # bf16 denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            101, 3, 47, 54, 36, 19, 70, 79,
            35, 95, 28, 120, 84, 94, 20, 92,
            18, 42, 98, 58, 3, 26, 64, 86,
            60, 86, 52, 23, 61, 70, 59, 74,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            61, 20, 52, 54, 49, 42, 57, 58,
            49, 60, 46, 63, 58, 60, 42, 60,
            41, 50, 60, 54, 20, 45, 56, 59,
            55, 59, 53, 44, 55, 57, 55, 57,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # fp32 some denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            8388608, 1063716449, 1064039365, 1063568877, 1051091338, 1062185569, 1034449408, 1060813641,
            1054893736, 1034907680, 1036660744, 1023639888, 1058536559, 1050896496, 1049237634, 1064950601,
            1051852994, 1059794063, 1054011102, 1062023602, 1059467900, 1062276774, 1059155029, 1053287574,
            1064378711, 1055768540, 1045266076, 1059575077, 1054928758, 1040468200, 1058061961, 1053066436,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            0, 118, 119, 118, 106, 117, 91, 116,
            110, 91, 93, 80, 113, 106, 105, 120,
            107, 115, 109, 117, 114, 117, 114, 108,
            119, 111, 101, 114, 110, 96, 113, 108,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # bf16 some denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            128, 16118, 16143, 16074, 16187, 16002, 16193, 16217,
            15680, 16183, 16092, 16158, 16251, 15876, 15896, 16194,
            16135, 16214, 16205, 16110, 16122, 15960, 15824, 16106,
            16220, 16230, 15952, 15896, 16000, 16144, 16232, 16157,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            0, 111, 113, 109, 116, 104, 116, 118,
            84, 115, 110, 114, 120, 96, 98, 116,
            112, 117, 117, 111, 112, 102, 93, 111,
            118, 118, 101, 98, 104, 113, 118, 114,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # zero
        data_hp = torch.tensor([0] * 32, dtype=torch.uint32).view(torch.float32)
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        ground_truth_fp8 = torch.tensor([0] * 32, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # fp32 normal
        # fmt: off
        data_hp = torch.tensor(
            [
            1037408064, 1058534842, 1053630662, 1063310394, 994704128, 1057245441, 1060663708, 1058053571,
            1052395648, 1064831570, 1038427336, 1064777688, 1059248393, 1060959028, 1062878286, 1057799482,
            1057854101, 1053562724, 1027482352, 1060498324, 1063238522, 1060472055, 1054346794, 1029092912,
            1056687298, 1059146141, 1037992128, 1064097772, 1056522806, 1059255744, 1064364912, 1060606252,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            93, 113, 109, 118, 53, 112, 116, 113,
            108, 120, 94, 119, 114, 116, 118, 113,
            113, 109, 84, 115, 118, 115, 110, 85,
            112, 114, 94, 119, 112, 114, 119, 115,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))
        # bf16 normal
        # fmt: off
        data_hp = torch.tensor(
            [
            15752, 16143, 16182, 15896, 16195, 16186, 16048, 16223,
            15988, 16231, 16140, 16088, 16032, 16240, 16228, 16133,
            16210, 16024, 16248, 16187, 16050, 15696, 16060, 15956,
            16131, 16251, 15896, 16014, 15808, 16024, 16159, 16186,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            88, 113, 115, 98, 116, 116, 107, 118,
            103, 118, 113, 110, 106, 119, 118, 112,
            117, 106, 120, 116, 107, 85, 108, 101,
            112, 120, 98, 105, 92, 106, 114, 116,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        data = data_hp.to(device).view(1, -1)
        qdata, scales = quantize_fn(data, **_MXFP8_NO_SWIZZLE_KWARGS)
        self.assertEqual(scales, ground_truth_scale.to(device).view(1, -1))
        self.assertEqual(qdata, ground_truth_fp8.to(device).view(1, -1))


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
        with self.assertRaisesRegex(ValueError, "RCEIL_E8M0"):
            quantize_fn(input, **(kwargs | {"inner_scale_calc": 1}))
        with self.assertRaisesRegex(ValueError, "2D"):
            quantize_fn(input.flatten(), **kwargs)

    def test_quantize_tensor_argument_names(self):
        self.assertEqual(F.InnerScaleCalc.RCEIL_E8M0.value, 0)
        public_names = tuple(inspect.signature(F.quantize_tensor).parameters)
        dual_names = tuple(inspect.signature(F.quantize_tensor_dual).parameters)
        native = torch.ops.aten._quantize_tensor.default._schema.arguments
        native_dual = torch.ops.aten._quantize_tensor_dual.default._schema.arguments
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
            self.assertEqual(str(schema_args[2].type), "int")
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
            scaling_type_square_block_and_expand=square,
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
            inner_scale_calc=F.InnerScaleCalc.RCEIL_E8M0.value,
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


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
