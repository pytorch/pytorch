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
    HardwareClassification,
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
    dim: int,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    if input.ndim != 2 or dim not in (-2, -1, 0, 1) or is_square_scaling:
        raise ValueError("unsupported MXFP8 reference configuration")
    dim %= 2
    swizzle = (
        SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
    )
    source = (input if dim == 1 else input.t()).contiguous()
    scales, qdata = to_mxfp8_reference(source, format="mxfp8", swizzle_type=swizzle)
    if dim == 0:
        qdata = qdata.t()
        # For packed scales, the singleton axis makes this a metadata-only transpose.
        scales = scales.t()
    return qdata, scales


def _quantize_mxfp8_tma(
    input: torch.Tensor,
    dim: tuple[int, int],
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
        _blockscaled_tma_impl,
    )

    return _blockscaled_tma_impl(input, dim, is_square_scaling, is_scale_swizzled)


def _quantize_mxfp8_tma_dual(input: torch.Tensor):
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
        _blockscaled_tma_impl_dual,
    )

    return _blockscaled_tma_impl_dual(input)


_NVIDIA_SM100_OR_LATER = torch.version.rocm is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference, name="reference"),
    subtest(_quantize_mxfp8_tma, name="tma", decorators=[skipIfNoCuteDSL]),
)


class TestMXFP8ReferenceNumerics(TestCase):
    hw_classification = HardwareClassification.CUDA

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
        dim = -1 if quantize_fn is _quantize_mxfp8_reference else (-2, -1)
        qdata, scales = quantize_fn(cases.inputs, dim, False, False)
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
        qdata_ref, scales_ref = _quantize_mxfp8_reference(data_hp, -1, False, swizzled)
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-2, -1), False, swizzled)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales.shape, scales_ref.shape)
        self.assertEqual(scales, scales_ref)
        if swizzled:
            scales = from_blocked(qdata, scales.flatten(), 32)
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
        qdata_ref, scales_ref = _quantize_mxfp8_reference(data_hp, -2, False, True)
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-1, -2), False, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(qdata.shape, shape)
        self.assertEqual(qdata.stride(), (1, shape[0]))
        self.assertEqual(scales.shape, (scales.numel(), 1))
        self.assertEqual(scales, scales_ref)

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
        qdata_ref, scales_ref = _quantize_mxfp8_reference(data_hp, -2, False, True)
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-1, -2), False, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales, scales_ref)

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
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference(data_hp, -1, False, True)
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference(data_hp, -2, False, True)
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(data_hp)
        self.assertEqual(qdata_k.view(torch.uint8), qdata_k_ref.view(torch.uint8))
        self.assertEqual(qdata_m.view(torch.uint8), qdata_m_ref.view(torch.uint8))
        self.assertEqual(qdata_m.shape, shape)
        self.assertEqual(qdata_m.stride(), (1, shape[0]))
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(scales_m.shape, (scales_m.numel(), 1))
        self.assertEqual(scales_m, scales_m_ref)

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
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-2, -1), True, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales, scales_ref)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square_nan(self, input_dtype, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 64, device=device, dtype=input_dtype)
        data_hp[0, 0] = float("nan")
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-2, -1), True, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales, scales_ref)
        expected_nan = torch.zeros((64, 64), device=device, dtype=torch.bool)
        expected_nan[:32, :32] = True
        self.assertEqual(torch.isnan(qdata), expected_nan)

        scale_bytes = from_blocked(qdata, scales.flatten(), 32).view(torch.uint8)
        expected_nan_scale = torch.zeros((64, 2), device=device, dtype=torch.bool)
        expected_nan_scale[:32, 0] = True
        self.assertEqual(scale_bytes == 255, expected_nan_scale)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "dim,is_scale_swizzled",
        (
            subtest(((-2, -1), True), name="dim_k_swizzled"),
            subtest(((-1, -2), True), name="dim_m_swizzled"),
            subtest(((-1, -2), False), name="dim_m_compact"),
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_transposed_input(
        self, input_dtype, dim, is_scale_swizzled, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 128, dtype=input_dtype, device=device).t()
        axis = dim[-1] % 2
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp, axis, False, is_scale_swizzled
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, dim, False, is_scale_swizzled)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(qdata.shape, data_hp.shape)
        expected_stride = (data_hp.shape[1], 1) if axis == 1 else (1, data_hp.shape[0])
        self.assertEqual(qdata.stride(), expected_stride)
        if is_scale_swizzled:
            expected_scale_shape = (
                (1, scales.numel()) if axis == 1 else (scales.numel(), 1)
            )
            self.assertEqual(scales.shape, expected_scale_shape)
        else:
            self.assertEqual(scales.shape, (data_hp.shape[0] // 32, data_hp.shape[1]))
            self.assertEqual(scales.stride(), (1, data_hp.shape[0] // 32))

    @skipIfNoCuteDSL
    def test_to_mx_rceil_transposed_input_dual(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 128, dtype=torch.bfloat16, device=device).t()
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference(data_hp, -1, False, True)
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference(data_hp, -2, False, True)
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(data_hp)
        self.assertEqual(qdata_k, qdata_k_ref)
        self.assertEqual(qdata_m, qdata_m_ref)
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(scales_m, scales_m_ref)
        self.assertEqual(qdata_k.shape, data_hp.shape)
        self.assertEqual(qdata_m.shape, data_hp.shape)
        self.assertEqual(qdata_k.stride(), (data_hp.shape[1], 1))
        self.assertEqual(qdata_m.stride(), (1, data_hp.shape[0]))
        self.assertEqual(scales_k.shape, (1, scales_k.numel()))
        self.assertEqual(scales_m.shape, (scales_m.numel(), 1))

    @skipIfNoCuteDSL
    def test_to_mx_rceil_transposed_input_square(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 128, dtype=torch.bfloat16, device=device).t()
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp.t())
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-1, -2), True, True)
        self.assertEqual(qdata, qdata_ref.t())
        self.assertEqual(scales, scales_ref.t())
        self.assertEqual(scales.shape, (scales.numel(), 1))
        self.assertEqual(qdata.shape, data_hp.shape)
        self.assertEqual(qdata.stride(), (1, data_hp.shape[0]))

    @skipIfNoCuteDSL
    def test_to_mx_rceil_strided_input_rejected(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 256, dtype=torch.bfloat16, device=device)[:, ::2]
        with self.assertRaisesRegex(ValueError, "contiguous or transpose-contiguous"):
            _quantize_mxfp8_tma(data_hp, (-2, -1), False, True)

    @parametrize(
        "dim,is_square_scaling,is_scale_swizzled",
        (
            subtest(((-2, -1), False, False), name="dim_k_compact"),
            subtest(((-2, -1), False, True), name="dim_k_swizzled"),
            subtest(((-2, -1), True, True), name="dim_k_square"),
            subtest(((-1, -2), False, True), name="dim_m"),
        ),
    )
    @parametrize(
        "shape",
        ((0, 32), (32, 0), (0, 0)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_empty(
        self, dim, is_square_scaling, is_scale_swizzled, shape, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape
        scale_k_shape = (
            (1, ((M + 127) // 128) * ((K + 127) // 128) * 32 * 16)
            if is_scale_swizzled
            else (M, K // 32)
        )
        scale_m_shape = (((K + 127) // 128) * ((M + 127) // 128) * 32 * 16, 1)
        expected_shapes = (
            (M, K),
            scale_k_shape if dim[-1] == -1 else scale_m_shape,
        )
        outputs = _quantize_mxfp8_tma(
            data_hp, dim, is_square_scaling, is_scale_swizzled
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
    def test_to_mx_rceil_dual_empty(self, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(data_hp)
        self.assertEqual(qdata_k.shape, (M, K))
        self.assertEqual(qdata_m.shape, (M, K))
        scale_bytes = ((M + 127) // 128) * ((K + 127) // 128) * 512
        self.assertEqual(scales_k.shape, (1, scale_bytes))
        self.assertEqual(scales_m.shape, (scale_bytes, 1))
        for output in (qdata_k, scales_k, qdata_m, scales_m):
            self.assertEqual(output.device, data_hp.device)
            self.assertEqual(output.numel(), 0)


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
