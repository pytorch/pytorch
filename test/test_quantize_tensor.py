# Owner(s): ["module: linear algebra"]

import torch
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import (
    _f32_to_e8m0_rceil,
    compute_error,
    from_blocked,
    from_blocked_3d,
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


def _quantize_mxfp8_reference_inner(
    input: torch.Tensor,
    dim: int,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    if input.ndim != 2 or dim not in (-2, -1, 0, 1):
        raise ValueError("unsupported MXFP8 reference configuration")
    dim %= 2
    if is_square_scaling and not is_scale_swizzled:
        raise ValueError("square MXFP8 reference requires swizzled scales")
    swizzle = (
        SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
    )
    source = (input if dim == 1 else input.t()).contiguous()
    if is_square_scaling:
        qdata, scales = mxfp8_32x32_swizzle_f(source)
    else:
        scales, qdata = to_mxfp8_reference(source, format="mxfp8", swizzle_type=swizzle)
    if dim == 0:
        qdata = qdata.t()
        # For packed scales, the singleton axis makes this a metadata-only transpose.
        scales = scales.t()
    return qdata, scales


def _quantize_mxfp8_reference_nd(
    input: torch.Tensor,
    dim: tuple[int, int],
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    r"""_quantize_mxfp8_reference_nd(input, dim, is_square_scaling, is_scale_swizzled) -> tuple[Tensor, Tensor]

    Args:
        input (Tensor): Tensor containing the matrices to quantize.
        dim (tuple[int, int]): Row and quantized axes, in that order.
        is_square_scaling (bool): Compute scales per square block before expansion.
          Requires ``is_scale_swizzled=True``.
        is_scale_swizzled (bool): Return packed, swizzled scales when true.

    Returns:
        tuple[Tensor, Tensor]: MXFP8 qdata and E8M0 scales. Each matrix is
        stored with its quantized axis contiguous, including for batched dim-M.

    Examples::

        # Shape @ stride; is_square_scaling=False in every example.
        # Compact means is_scale_swizzled=False; swizzled means True.
        # L = 512 * ceil(M / 128) * ceil(K / 128).
        input (M, K), dim=(-2, -1):
            qdata             (M, K) @ (K, 1)
            scales compact    (M, K//32) @ (K//32, 1)
            scales swizzled   (1, L) @ (L, 1)

        input (M, K), dim=(-1, -2):
            qdata             (M, K) @ (1, M)
            scales compact    (M//32, K) @ (1, M//32)
            scales swizzled   (L, 1) @ (1, L)

        input (B, M, K), dim=(-2, -1):
            qdata             (B, M, K) @ (M*K, K, 1)
            scales compact    (B, M, K//32) @ (M*(K//32), K//32, 1)
            scales swizzled   (B, 1, L) @ (L, L, 1)

        input (B, M, K), dim=(-1, -2):
            qdata             (B, M, K) @ (M*K, 1, M)
            scales compact    (B, M//32, K) @ ((M//32)*K, 1, M//32)
            scales swizzled   (B, L, 1) @ (L, 1, L)

        input (B0, M, B1, K, B2), dim=(1, 3):
            qdata             (B0, M, B1, K, B2)
                              @ (B1*B2*M*K, K, B2*M*K, 1, M*K)
            scales compact    (B0, M, B1, K//32, B2)
                              @ (B1*B2*M*(K//32), K//32, B2*M*(K//32), 1,
                                 M*(K//32))
            scales swizzled   (B0, 1, B1, L, B2)
                              @ (B1*B2*L, L, B2*L, 1, L)
    """
    # canonicalize quantization axes to positive
    # for example, if ndim == 3 then dim = (-2, -1) is canonicalized to dim = (1, 2)
    axes = tuple(axis % input.ndim for axis in dim)
    # sort the quantization axes
    matrix_axes = tuple(sorted(axes))
    # calculate batch axes that do not participate in quantization
    batch_axes = tuple(axis for axis in range(input.ndim) if axis not in matrix_axes)
    # canonicalize axis order into (*B, M, K)
    permutation = (*batch_axes, *matrix_axes)
    # find the permutation to go from canonicalized axis order back to original axis order
    inverse = tuple(permutation.index(axis) for axis in range(input.ndim))
    batch_shape = tuple(input.shape[axis] for axis in batch_axes)
    # make the canonicalized order contiguous, to keep the reference simple
    canonical = input.permute(permutation).contiguous()
    matrices = canonical.reshape(-1, *canonical.shape[-2:])
    # the line below selects dim-k vs dim-m in the 2d reference
    matrix_dim = -1 if axes[-1] == matrix_axes[-1] else -2
    # loop over 2d reference for each flattened batch
    outputs = [
        _quantize_mxfp8_reference_inner(
            matrix, matrix_dim, is_square_scaling, is_scale_swizzled
        )
        for matrix in matrices
    ]
    if batch_shape:
        if matrix_dim == -2:
            qdata = torch.stack([result[0].t() for result in outputs]).transpose(-2, -1)
            scales = torch.stack(
                [result[1].t().view(torch.uint8) for result in outputs]
            ).transpose(-2, -1)
        else:
            qdata = torch.stack([result[0] for result in outputs])
            scales = torch.stack([result[1].view(torch.uint8) for result in outputs])
        if len(batch_shape) > 1:
            qdata = qdata.unflatten(0, batch_shape)
            scales = scales.unflatten(0, batch_shape)
        scales = scales.view(torch.float8_e8m0fnu)
    else:
        qdata, scales = outputs[0]
    # undo the axis order permutation
    return qdata.permute(inverse), scales.permute(inverse)


def _quantize_mxfp8_tma(
    input: torch.Tensor,
    dim: tuple[int, int] | None,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
        _blockscaled_tma_impl,
    )

    return _blockscaled_tma_impl(input, dim, is_square_scaling, is_scale_swizzled)


def _quantize_mxfp8_tma_dual(
    input: torch.Tensor,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
):
    from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
        _blockscaled_tma_impl_dual,
    )

    return _blockscaled_tma_impl_dual(input, is_square_scaling, is_scale_swizzled)


_NVIDIA_SM100_OR_LATER = torch.version.rocm is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference_nd, name="reference"),
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
        qdata, scales = quantize_fn(cases.inputs, (-2, -1), False, False)
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
            (1, 32, 128),
            (3, 1, 32),
            (3, 129, 64),
            (3, 160, 160),
        ),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_dim_k(self, swizzle_type, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            self.skipTest("3D MXFP8 coverage uses only bfloat16")
        swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, swizzled
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-2, -1), False, swizzled)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales.shape, scales_ref.shape)
        self.assertEqual(scales, scales_ref)
        if len(shape) == 3:
            _, M, K = shape
            self.assertEqual(qdata.shape, shape)
            self.assertEqual(qdata.stride(), (M * K, K, 1))
            if swizzled:
                scales = from_blocked_3d(qdata, scales, 32)
        else:
            if swizzled:
                scales = from_blocked(qdata, scales.flatten(), 32)
        dequantized = from_blocked_format(qdata, scales).float()
        sqnr = compute_error(data_hp.float(), dequantized)
        self.assertGreater(sqnr.item(), 18.0)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 16), (64, 128), (160, 144), (2048, 2080), (1, 32, 128), (3, 160, 160)),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_dim_m(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            self.skipTest("3D MXFP8 coverage uses only bfloat16")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-1, -2), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-1, -2), False, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(qdata.shape, shape)
        self.assertEqual(qdata.stride(), qdata_ref.stride())
        M, K = shape[-2:]
        self.assertEqual(qdata.stride(), (M * K, 1, M) if len(shape) == 3 else (1, M))
        self.assertEqual(scales.shape, scales_ref.shape)
        self.assertEqual(scales.shape[-1], 1)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(scales.stride(), scales_ref.stride())

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
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-1, -2), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-1, -2), False, True)
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales, scales_ref)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 32), (64, 128), (160, 160), (2048, 2080), (3, 160, 160)),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_km(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, True
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-1, -2), False, True
        )
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(
            data_hp, False, True
        )
        M, K = shape[-2:]
        m_blocks, k_blocks = (M + 127) // 128, (K + 127) // 128
        scale_bytes = m_blocks * k_blocks * 512
        if len(shape) == 3:
            B = shape[0]
            stride_k = (M * K, K, 1)
            stride_m = (M * K, 1, M)
            scale_k_shape = (B, 1, scale_bytes)
            scale_m_shape = (B, scale_bytes, 1)
        else:
            stride_k = (K, 1)
            stride_m = (1, M)
            scale_k_shape = (1, scale_bytes)
            scale_m_shape = (scale_bytes, 1)
        self.assertEqual(qdata_k, qdata_k_ref)
        self.assertEqual(qdata_m, qdata_m_ref)
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(scales_m, scales_m_ref)
        self.assertEqual(qdata_k.shape, shape)
        self.assertEqual(qdata_m.shape, shape)
        self.assertEqual(qdata_k.stride(), stride_k)
        self.assertEqual(qdata_m.stride(), stride_m)
        self.assertEqual(scales_k.shape, scale_k_shape)
        self.assertEqual(scales_m.shape, scale_m_shape)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape,transposed",
        (
            ((32, 32), False),
            ((64, 128), False),
            ((160, 160), False),
            ((2048, 4224), False),
            ((3, 160, 160), False),
            ((3, 160, 160), True),
        ),
        name_fn=lambda shape, transposed: (
            (f"B{shape[0]}_" if len(shape) == 3 else "")
            + f"M{shape[-2]}_K{shape[-1]}"
            + ("_transposed" if transposed else "")
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square(self, input_dtype, shape, transposed, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            self.skipTest("3D MXFP8 coverage uses only bfloat16")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        if transposed:
            data_hp = data_hp.transpose(-2, -1)
        dim = (-1, -2) if transposed else (-2, -1)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(data_hp, dim, True, True)
        qdata, scales = _quantize_mxfp8_tma(data_hp, dim, True, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        if len(shape) == 3:
            scale_shape = (3, 2 * 2 * 512, 1) if transposed else (3, 1, 2 * 2 * 512)
            self.assertEqual(scales.shape, scale_shape)

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
        "shape",
        ((64, 128), (2, 128, 64)),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
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
        self, input_dtype, shape, dim, is_scale_swizzled, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            self.skipTest("3D MXFP8 coverage uses only bfloat16")
        data_hp = torch.randn(shape, dtype=input_dtype, device=device).transpose(-2, -1)
        axis = dim[-1] % 2
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, dim, False, is_scale_swizzled
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, dim, False, is_scale_swizzled)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(qdata.shape, data_hp.shape)
        M, K = data_hp.shape[-2:]
        expected_stride = (K, 1) if axis == 1 else (1, M)
        if len(shape) == 3:
            expected_stride = (M * K, *expected_stride)
        self.assertEqual(qdata.stride(), expected_stride)
        if is_scale_swizzled:
            scale_bytes = ((M + 127) // 128) * ((K + 127) // 128) * 512
            scale_shape = (1, scale_bytes) if axis == 1 else (scale_bytes, 1)
        else:
            scale_shape = (M // 32, K)
        if len(shape) == 3:
            scale_shape = (shape[0], *scale_shape)
        self.assertEqual(scales.shape, scale_shape)
        if len(shape) == 2 and not is_scale_swizzled:
            self.assertEqual(scales.stride(), (1, M // 32))

    @parametrize(
        "shape",
        ((64, 128), (2, 128, 64)),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_transposed_input_dual(self, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, dtype=torch.bfloat16, device=device).transpose(
            -2, -1
        )
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, True
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-1, -2), False, True
        )
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(
            data_hp, False, True
        )
        self.assertEqual(qdata_k, qdata_k_ref)
        self.assertEqual(qdata_m, qdata_m_ref)
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(scales_m, scales_m_ref)
        self.assertEqual(qdata_k.shape, data_hp.shape)
        self.assertEqual(qdata_m.shape, data_hp.shape)
        M, K = data_hp.shape[-2:]
        stride_k = (K, 1)
        stride_m = (1, M)
        if len(shape) == 3:
            stride_k = (M * K, *stride_k)
            stride_m = (M * K, *stride_m)
        self.assertEqual(qdata_k.stride(), stride_k)
        self.assertEqual(qdata_m.stride(), stride_m)
        scale_bytes = ((M + 127) // 128) * ((K + 127) // 128) * 512
        scale_k_shape = (1, scale_bytes)
        scale_m_shape = (scale_bytes, 1)
        if len(shape) == 3:
            scale_k_shape = (shape[0], *scale_k_shape)
            scale_m_shape = (shape[0], *scale_m_shape)
        self.assertEqual(scales_k.shape, scale_k_shape)
        self.assertEqual(scales_m.shape, scale_m_shape)

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

    @parametrize(
        "layout",
        (
            "leading",
            "leading_short_tile",
            "trailing",
            "trailing_short_tile",
            "trailing_batch_tile",
            "trailing_batch_partial",
            "interleaved_trailing_batch",
            "interleaved_batch",
            "interleaved_rank3",
            "interleaved_singleton",
            "interleaved_multi_batch",
            "permuted_interleaved_batch",
            "transposed_interleaved_batch",
            "permuted_trailing_batch",
            "sliced_trailing_batch",
            "interleaved",
            "permuted",
            "transposed",
        ),
    )
    @parametrize("mode", ("k_compact", "k_swizzled", "m_swizzled", "k_square"))
    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd(self, layout, mode, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if layout == "leading":
            data_hp = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16, device=device)
            axes = (2, 3)
        elif layout in ("leading_short_tile", "trailing_short_tile"):
            # Distinct batches expose aliases in the short dim-K tile's compact scale store.
            batched = torch.empty((2, 1024, 1152), dtype=torch.bfloat16, device=device)
            batched[0].fill_(0.25)
            batched[1].fill_(2.5)
            if layout == "leading_short_tile":
                data_hp, axes = batched, (1, 2)
            else:
                data_hp, axes = batched.permute(1, 2, 0).contiguous(), (0, 1)
        elif layout == "trailing":
            data_hp = torch.randn(2, 64, 128, 3, dtype=torch.bfloat16, device=device)
            axes = (1, 2)
        elif layout == "trailing_batch_tile":
            data_hp = torch.randn(2, 64, 128, 32, dtype=torch.bfloat16, device=device)
            axes = (1, 2)
        elif layout == "trailing_batch_partial":
            data_hp = torch.randn(2, 160, 96, 33, dtype=torch.bfloat16, device=device)
            axes = (1, 2)
        elif layout == "interleaved_trailing_batch":
            data_hp = torch.randn(
                2, 160, 3, 96, 64, dtype=torch.bfloat16, device=device
            )
            axes = (1, 3)
        elif layout == "interleaved_batch":
            data_hp = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
            axes = (1, 3)
        elif layout == "interleaved_rank3":
            data_hp = torch.randn(64, 4, 128, dtype=torch.bfloat16, device=device)
            axes = (0, 2)
        elif layout == "interleaved_singleton":
            data_hp = torch.randn(64, 1, 128, dtype=torch.bfloat16, device=device)
            axes = (0, 2)
        elif layout == "interleaved_multi_batch":
            data_hp = torch.randn(
                2, 3, 64, 4, 5, 128, dtype=torch.bfloat16, device=device
            )
            axes = (2, 5)
        elif layout == "permuted_interleaved_batch":
            data_hp = torch.randn(
                2, 3, 64, 4, 128, dtype=torch.bfloat16, device=device
            ).permute(2, 4, 0, 3, 1)
            axes = (0, 1)
        elif layout == "transposed_interleaved_batch":
            data_hp = torch.randn(
                2, 64, 4, 128, dtype=torch.bfloat16, device=device
            ).permute(0, 3, 2, 1)
            axes = (1, 3)
        elif layout == "permuted_trailing_batch":
            data_hp = torch.randn(
                2, 160, 96, 32, dtype=torch.bfloat16, device=device
            ).permute(0, 3, 1, 2)
            axes = (2, 3)
        elif layout == "sliced_trailing_batch":
            data_hp = torch.randn(2, 64, 256, 32, dtype=torch.bfloat16, device=device)[
                :, :, ::2, :
            ]
            axes = (1, 2)
        elif layout == "interleaved":
            data_hp = torch.randn(2, 64, 3, 128, 4, dtype=torch.bfloat16, device=device)
            axes = (1, 3)
        elif layout == "permuted":
            data_hp = torch.randn(
                2, 3, 64, 128, dtype=torch.bfloat16, device=device
            ).permute(0, 2, 3, 1)
            axes = (1, 2)
        else:
            data_hp = torch.randn(
                2, 3, 128, 64, dtype=torch.bfloat16, device=device
            ).permute(0, 3, 2, 1)
            axes = (1, 2)
        dim = axes[::-1] if mode == "m_swizzled" else axes
        square = mode == "k_square"
        swizzled = mode != "k_compact"
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, dim, square, swizzled
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, dim, square, swizzled)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(qdata.shape, data_hp.shape)
        self.assertEqual(qdata.stride()[dim[-1]], 1)
        self.assertEqual(scales.shape, scales_ref.shape)

    @parametrize(
        "layout",
        (
            "leading",
            "trailing",
            "trailing_batch_tile",
            "interleaved_batch",
            "interleaved_rank3",
            "interleaved_singleton",
            "interleaved_multi_batch",
            "permuted_interleaved_batch",
            "transposed_interleaved_batch",
            "interleaved",
            "permuted",
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd_dual(self, layout, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if layout == "leading":
            data_hp = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16, device=device)
            axes = (2, 3)
        elif layout == "trailing":
            data_hp = torch.randn(2, 64, 128, 3, dtype=torch.bfloat16, device=device)
            axes = (1, 2)
        elif layout == "trailing_batch_tile":
            data_hp = torch.randn(2, 64, 128, 32, dtype=torch.bfloat16, device=device)
            axes = (1, 2)
        elif layout == "interleaved_batch":
            data_hp = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
            axes = (1, 3)
        elif layout == "interleaved_rank3":
            data_hp = torch.randn(64, 4, 128, dtype=torch.bfloat16, device=device)
            axes = (0, 2)
        elif layout == "interleaved_singleton":
            data_hp = torch.randn(64, 1, 128, dtype=torch.bfloat16, device=device)
            axes = (0, 2)
        elif layout == "interleaved_multi_batch":
            data_hp = torch.randn(
                2, 3, 64, 4, 5, 128, dtype=torch.bfloat16, device=device
            )
            axes = (2, 5)
        elif layout == "permuted_interleaved_batch":
            data_hp = torch.randn(
                2, 3, 64, 4, 128, dtype=torch.bfloat16, device=device
            ).permute(2, 4, 0, 3, 1)
            axes = (0, 1)
        elif layout == "transposed_interleaved_batch":
            data_hp = torch.randn(
                2, 64, 4, 128, dtype=torch.bfloat16, device=device
            ).permute(0, 3, 2, 1)
            axes = (1, 3)
        elif layout == "interleaved":
            data_hp = torch.randn(2, 64, 3, 128, 4, dtype=torch.bfloat16, device=device)
            axes = (1, 3)
        else:
            data_hp = torch.randn(
                2, 3, 64, 128, dtype=torch.bfloat16, device=device
            ).permute(0, 2, 3, 1)
            axes = (1, 2)
        from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
            _blockscaled_tma_impl_dual,
        )

        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference_nd(
            data_hp, axes, False, True
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference_nd(
            data_hp, axes[::-1], False, True
        )
        qdata_k, scales_k, qdata_m, scales_m = _blockscaled_tma_impl_dual(
            data_hp, False, True, dim=(axes, axes[::-1])
        )
        self.assertEqual(qdata_k, qdata_k_ref)
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(qdata_m, qdata_m_ref)
        self.assertEqual(scales_m, scales_m_ref)
        self.assertEqual(qdata_k.shape, data_hp.shape)
        self.assertEqual(qdata_m.shape, data_hp.shape)
        self.assertEqual(qdata_k.stride()[axes[-1]], 1)
        self.assertEqual(qdata_m.stride()[axes[0]], 1)

    @skipIfNoCuteDSL
    def test_to_mx_rceil_strided_input_fallback(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 256, dtype=torch.bfloat16, device=device)[:, ::2]
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (-2, -1), False, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)

    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd_unaligned_fallback(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(2, 64, 128, 33, dtype=torch.bfloat16, device=device)[
            ..., 1:
        ]
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (1, 2), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (1, 2), False, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)

    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd_misaligned_contiguous_input(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        storage = torch.randn(2 * 64 * 128 + 1, dtype=torch.bfloat16, device=device)
        data_hp = storage[1:].view(2, 64, 128)
        self.assertTrue(data_hp.is_contiguous())
        self.assertNotEqual(data_hp.data_ptr() % 16, 0)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (1, 2), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (1, 2), False, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)

    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd_transposed_unaligned_pitch(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(2, 64, 17, dtype=torch.bfloat16, device=device).transpose(
            -2, -1
        )
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (1, 2), False, True
        )
        qdata, scales = _quantize_mxfp8_tma(data_hp, (1, 2), False, True)
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)

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
        (
            (0, 32),
            (32, 0),
            (0, 0),
            (0, 32, 64),
            (2, 0, 32),
            (2, 32, 0),
            (2, 0, 64, 128),
        ),
        name_fn=lambda shape: "".join(f"B{size}_" for size in shape[:-2])
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_empty(
        self, dim, is_square_scaling, is_scale_swizzled, shape, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape[-2:]
        scale_k_shape = (
            (1, ((M + 127) // 128) * ((K + 127) // 128) * 32 * 16)
            if is_scale_swizzled
            else (M, K // 32)
        )
        scale_m_shape = (((K + 127) // 128) * ((M + 127) // 128) * 32 * 16, 1)
        scale_shape = scale_k_shape if dim[-1] == -1 else scale_m_shape
        scale_shape = (*shape[:-2], *scale_shape)
        expected_shapes = (shape, scale_shape)
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
        (
            (0, 32),
            (32, 0),
            (0, 0),
            (0, 32, 64),
            (2, 0, 32),
            (2, 32, 0),
            (2, 0, 64, 128),
        ),
        name_fn=lambda shape: "".join(f"B{size}_" for size in shape[:-2])
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dual_empty(self, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape[-2:]
        qdata_k, scales_k, qdata_m, scales_m = _quantize_mxfp8_tma_dual(
            data_hp, False, True
        )
        self.assertEqual(qdata_k.shape, shape)
        self.assertEqual(qdata_m.shape, shape)
        scale_bytes = ((M + 127) // 128) * ((K + 127) // 128) * 512
        scale_k_shape = (1, scale_bytes)
        scale_m_shape = (scale_bytes, 1)
        scale_k_shape = (*shape[:-2], *scale_k_shape)
        scale_m_shape = (*shape[:-2], *scale_m_shape)
        self.assertEqual(scales_k.shape, scale_k_shape)
        self.assertEqual(scales_m.shape, scale_m_shape)
        for output in (qdata_k, scales_k, qdata_m, scales_m):
            self.assertEqual(output.device, data_hp.device)
            self.assertEqual(output.numel(), 0)


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
