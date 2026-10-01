# Owner(s): ["module: linear algebra"]

import torch
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_cuda import SM100OrLater
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
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    if quant_orientation != "dim_k" or is_square_scaling:
        raise ValueError("unsupported MXFP8 reference configuration")
    swizzle_type = (
        SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
    )
    scales, qdata = to_mxfp8_reference(input, format="mxfp8", swizzle_type=swizzle_type)
    return qdata, scales


def _quantize_mxfp8_tma(
    input: torch.Tensor,
    quant_orientation: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
        _blockscaled_tma_impl,
    )

    return _blockscaled_tma_impl(
        input, quant_orientation, is_square_scaling, is_scale_swizzled
    )


_NVIDIA_SM100_OR_LATER = torch.version.hip is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference, name="reference"),
    subtest(_quantize_mxfp8_tma, name="tma", decorators=[skipIfNoCuteDSL]),
)


class TestMXFP8ReferenceNumerics(TestCase):
    def test_f32_to_e8m0_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L113
        values, expected = make_f32_to_e8m0_rceil_cases(device=device)
        self.assertEqual(_f32_to_e8m0_rceil(values), expected.to(device))

    @parametrize("quantize_fn", _MXFP8_IMPLEMENTATIONS)
    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_mxfp8_corner_case_bytes(self, quantize_fn, input_dtype, device):
        if quantize_fn is _quantize_mxfp8_tma and not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, device=device)
        qdata, scales = quantize_fn(cases.inputs, "dim_k", False, False)
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
            data_hp, "dim_k", False, swizzled
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, "dim_k", False, swizzled)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
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
            data_hp.t().contiguous(), "dim_k", False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, "dim_m", False, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
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
            data_hp.t().contiguous(), "dim_k", False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, "dim_m", False, True)
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
            data_hp, "dim_k", False, True
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(), "dim_k", False, True
        )
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma(
            data_hp, "dim_km", False, True
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
        qdata, scales = _quantize_mxfp8_tma(data_hp, "dim_k", True, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
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
        qdata, scales = _quantize_mxfp8_tma(data_hp, "dim_k", True, True)
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
            subtest(("dim_km", False, True), name="dim_km"),
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
            "dim_km": ((M, K), scale_k_shape, (K, M), scale_m_shape),
        }[quant_orientation]
        outputs = _quantize_mxfp8_tma(
            data_hp, quant_orientation, is_square_scaling, is_scale_swizzled
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


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
