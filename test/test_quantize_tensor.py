# Owner(s): ["module: linear algebra"]

import inspect
from unittest.mock import patch

import torch
from torch.nn import functional as F
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
    HardwareClassification,
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


_NVIDIA_SM100_OR_LATER = torch.version.rocm is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference_nd, name="reference"),
    subtest(F.quantize_tensor, name="public", decorators=[skipIfNoCuteDSL]),
)
_MXFP8_KWARGS = {
    "dim": (-2, -1),
    "qdata_dtype": torch.float8_e4m3fn,
    "scaling_algorithm": F.ScalingAlgorithm.MXFP_E8M0_RU,
    "scaling_type": F.ScalingType.BlockWise1x32,
}
_MXFP8_NO_SWIZZLE_KWARGS = {
    **_MXFP8_KWARGS,
    "swizzle_type": SwizzleType.NO_SWIZZLE,
}


def _get_scale_shape(qdata_shape, swizzle_type):
    if swizzle_type == SwizzleType.NO_SWIZZLE:
        return (qdata_shape[0], qdata_shape[1] // 32)
    if swizzle_type == SwizzleType.SWIZZLE_32_4_4:
        scale_bytes = (
            ((qdata_shape[0] + 127) // 128) * ((qdata_shape[1] + 127) // 128) * 512
        )
        return (1, scale_bytes)
    raise ValueError(f"unsupported MXFP8 swizzle type: {swizzle_type}")


def _make_nd_quantize_input(layout, device):
    if layout == "2d":
        return torch.randn(64, 128, dtype=torch.bfloat16, device=device), (0, 1)
    if layout == "leading":
        return torch.randn(2, 3, 64, 128, dtype=torch.bfloat16, device=device), (2, 3)
    if layout == "interleaved":
        return torch.randn(2, 64, 3, 128, 4, dtype=torch.bfloat16, device=device), (
            1,
            3,
        )
    if layout == "trailing":
        return torch.randn(2, 64, 128, 3, dtype=torch.bfloat16, device=device), (1, 2)
    if layout == "transposed":
        return torch.randn(2, 128, 64, dtype=torch.bfloat16, device=device).transpose(
            -2, -1
        ), (1, 2)
    if layout == "unaligned":
        return torch.randn(2, 64, 128, 33, dtype=torch.bfloat16, device=device)[
            ..., 1:
        ], (1, 2)
    if layout == "fallback":
        return torch.randn(2, 64, 256, 3, dtype=torch.bfloat16, device=device)[
            :, :, ::2, :
        ], (1, 2)
    raise ValueError(f"unknown quantization test layout: {layout}")


def _get_tensor_and_quant_dims(
    test_case_name: str,
    device: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if test_case_name == "leading":
        data_hp = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16, device=device)
        axes = (2, 3)
    elif test_case_name == "leading_reverse":
        data_hp = torch.randn(2, 3, 64, 128, dtype=torch.bfloat16, device=device)
        axes = (3, 2)
    elif test_case_name == "leading_short_tile":
        # Distinct batches expose aliases in the short dim-K tile's compact scale store.
        data_hp = torch.empty((2, 1024, 1152), dtype=torch.bfloat16, device=device)
        data_hp[0].fill_(0.25)
        data_hp[1].fill_(2.5)
        axes = (1, 2)
    elif test_case_name == "trailing":
        data_hp = torch.randn(2, 64, 128, 3, dtype=torch.bfloat16, device=device)
        axes = (1, 2)
    elif test_case_name == "trailing_singleton":
        data_hp = torch.randn(2, 64, 128, 1, dtype=torch.bfloat16, device=device)
        axes = (1, 2)
    elif test_case_name == "trailing_reverse":
        data_hp = torch.randn(2, 64, 128, 3, dtype=torch.bfloat16, device=device)
        axes = (2, 1)
    elif test_case_name == "trailing_short_tile":
        batched = torch.empty((2, 1024, 1152), dtype=torch.bfloat16, device=device)
        batched[0].fill_(0.25)
        batched[1].fill_(2.5)
        data_hp = batched.permute(1, 2, 0).contiguous()
        axes = (0, 1)
    elif test_case_name == "trailing_batch_tile":
        data_hp = torch.randn(2, 64, 128, 32, dtype=torch.bfloat16, device=device)
        axes = (1, 2)
    elif test_case_name == "trailing_batch_partial":
        data_hp = torch.randn(2, 160, 96, 33, dtype=torch.bfloat16, device=device)
        axes = (1, 2)
    elif test_case_name == "interleaved_trailing_batch":
        data_hp = torch.randn(2, 160, 3, 96, 64, dtype=torch.bfloat16, device=device)
        axes = (1, 3)
    elif test_case_name == "interleaved_batch":
        data_hp = torch.randn(2, 64, 4, 128, dtype=torch.bfloat16, device=device)
        axes = (1, 3)
    elif test_case_name == "interleaved_rank3":
        data_hp = torch.randn(64, 4, 128, dtype=torch.bfloat16, device=device)
        axes = (0, 2)
    elif test_case_name == "interleaved_singleton":
        data_hp = torch.randn(64, 1, 128, dtype=torch.bfloat16, device=device)
        axes = (0, 2)
    elif test_case_name == "interleaved_multi_batch":
        data_hp = torch.randn(2, 3, 64, 4, 5, 128, dtype=torch.bfloat16, device=device)
        axes = (2, 5)
    elif test_case_name == "permuted_interleaved_batch":
        data_hp = torch.randn(
            2, 3, 64, 4, 128, dtype=torch.bfloat16, device=device
        ).permute(2, 4, 0, 3, 1)
        axes = (0, 1)
    elif test_case_name == "transposed_interleaved_batch":
        data_hp = torch.randn(
            2, 64, 4, 128, dtype=torch.bfloat16, device=device
        ).permute(0, 3, 2, 1)
        axes = (1, 3)
    elif test_case_name == "permuted_trailing_batch":
        data_hp = torch.randn(
            2, 160, 96, 32, dtype=torch.bfloat16, device=device
        ).permute(0, 3, 1, 2)
        axes = (2, 3)
    elif test_case_name == "sliced_trailing_batch":
        data_hp = torch.randn(2, 64, 256, 32, dtype=torch.bfloat16, device=device)[
            :, :, ::2, :
        ]
        axes = (1, 2)
    elif test_case_name == "interleaved":
        data_hp = torch.randn(2, 64, 3, 128, 4, dtype=torch.bfloat16, device=device)
        axes = (1, 3)
    elif test_case_name == "permuted":
        data_hp = torch.randn(
            2, 3, 64, 128, dtype=torch.bfloat16, device=device
        ).permute(0, 2, 3, 1)
        axes = (1, 2)
    elif test_case_name == "transposed":
        data_hp = torch.randn(
            2, 3, 128, 64, dtype=torch.bfloat16, device=device
        ).permute(0, 3, 2, 1)
        axes = (1, 2)
    else:
        raise ValueError(f"unknown MXFP8 layout: {test_case_name}")
    return data_hp, axes


_MXFP8_SINGLE_LAYOUTS = (
    "leading",
    "leading_short_tile",
    "trailing",
    "trailing_singleton",
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
)

_MXFP8_DUAL_LAYOUTS = (
    "leading",
    "leading_reverse",
    "trailing",
    "trailing_reverse",
    "trailing_batch_tile",
    "interleaved_batch",
    "interleaved_rank3",
    "interleaved_singleton",
    "interleaved_multi_batch",
    "permuted_interleaved_batch",
    "transposed_interleaved_batch",
    "interleaved",
    "permuted",
)


def _assert_bitwise_eq(a: torch.Tensor, b: torch.Tensor) -> None:
    # Repeating uint8 views at each call site is verbose; self.assertEqual
    # on floats misses wrong signs on zero and NaN encodings.
    # TODO: Use this helper for other qdata checks in a future PR.
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


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
        if (
            quantize_fn is not _quantize_mxfp8_reference_nd
            and not _NVIDIA_SM100_OR_LATER
        ):
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, device=device)
        if quantize_fn is F.quantize_tensor:
            qdata, scales = quantize_fn(cases.inputs, **_MXFP8_NO_SWIZZLE_KWARGS)
        else:
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
            (2, 1024, 2048),
        ),
        # Keep this lambda inline because a shared helper would add indirection.
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_dim_k(self, swizzle_type, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            # Run one batched fp32 case to keep the test count down.
            if (input_dtype, shape, swizzle_type) != (
                torch.float32,
                (2, 1024, 2048),
                SwizzleType.SWIZZLE_32_4_4,
            ):
                self.skipTest("3D MXFP8 uses bf16, plus one fp32 case")
        swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, swizzled
        )
        qdata, scales = F.quantize_tensor(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        _assert_bitwise_eq(qdata, qdata_ref)
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
        (
            (32, 16),
            (64, 128),
            (160, 144),
            (2048, 2080),
            (1, 32, 128),
            (3, 160, 160),
            (2, 2048, 2080),
        ),
        name_fn=lambda shape: (f"B{shape[0]}_" if len(shape) == 3 else "")
        + f"M{shape[-2]}_K{shape[-1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_dim_m(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        if len(shape) == 3 and input_dtype != torch.bfloat16:
            # Run one batched fp32 case to keep the test count down.
            if input_dtype != torch.float32 or shape != (2, 2048, 2080):
                self.skipTest("3D MXFP8 uses bf16, plus one fp32 case")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-1, -2), False, True
        )
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (-1, -2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        _assert_bitwise_eq(qdata, qdata_ref)
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (-1, -2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(scales, scales_ref)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        (
            (32, 32),
            (64, 128),
            (160, 160),
            (2048, 2080),
            (3, 160, 160),
            (2, 2048, 2080),
        ),
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
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
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
        _assert_bitwise_eq(qdata_k, qdata_k_ref)
        _assert_bitwise_eq(qdata_m, qdata_m_ref)
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": dim},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        _assert_bitwise_eq(qdata, qdata_ref)
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
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
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
        )
        qdata, scales = F.quantize_tensor(
            data_hp, **{**_MXFP8_KWARGS, "dim": dim}, swizzle_type=swizzle_type
        )
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
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (-1, -2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        self.assertEqual(qdata, qdata_ref.t())
        self.assertEqual(scales, scales_ref.t())
        self.assertEqual(scales.shape, (scales.numel(), 1))
        self.assertEqual(qdata.shape, data_hp.shape)
        self.assertEqual(qdata.stride(), (1, data_hp.shape[0]))

    @parametrize(
        "shape,dim",
        (
            subtest(((64, 128), (1, 0)), name="matrix"),
            subtest(((2, 64, 128, 3), (2, 1)), name="trailing_batch"),
        ),
    )
    @parametrize(
        "is_square_scaling,is_scale_swizzled",
        (
            subtest((False, False), name="compact"),
            subtest((True, True), name="square"),
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_m_recipe_fallback(
        self, shape, dim, is_square_scaling, is_scale_swizzled, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, dtype=torch.bfloat16, device=device)
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, dim, is_square_scaling, is_scale_swizzled
        )
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": dim},
            swizzle_type=(
                SwizzleType.SWIZZLE_32_4_4
                if is_scale_swizzled
                else SwizzleType.NO_SWIZZLE
            ),
            scaling_type_use_square_block_size=is_square_scaling,
        )
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(qdata.stride(dim[-1]), 1)
        self.assertEqual(scales.stride(dim[-1]), 1)

    @parametrize(
        "is_square_scaling,is_scale_swizzled,error",
        (
            subtest(
                (
                    False,
                    False,
                    "dual mxfp8 quantization currently requires SWIZZLE_32_4_4",
                ),
                name="compact",
            ),
            subtest(
                (True, True, "dual mxfp8 quantization does not yet support 32x32"),
                name="square",
            ),
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dual_unsupported_recipe(
        self, is_square_scaling, is_scale_swizzled, error, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        # Copying cannot make both dual orientations use a dim-k-only recipe.
        with self.assertRaisesRegex(ValueError, error):
            F.quantize_tensor_dual(
                data_hp,
                **_MXFP8_KWARGS,
                swizzle_type=(
                    SwizzleType.SWIZZLE_32_4_4
                    if is_scale_swizzled
                    else SwizzleType.NO_SWIZZLE
                ),
                scaling_type_use_square_block_size=is_square_scaling,
            )

    @parametrize("layout", _MXFP8_SINGLE_LAYOUTS)
    @parametrize("mode", ("k_compact", "k_swizzled", "m_swizzled", "k_square"))
    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd(self, layout, mode, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp, axes = _get_tensor_and_quant_dims(layout, device)
        dim = axes[::-1] if mode == "m_swizzled" else axes
        square = mode == "k_square"
        swizzled = mode != "k_compact"
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, dim, square, swizzled
        )
        # Protect current no-copy fast paths, not a public API guarantee.
        expected_view = layout in {
            "leading",
            "leading_short_tile",
            "trailing_singleton",
            "interleaved_batch",
            "interleaved_rank3",
            "interleaved_singleton",
            "interleaved_multi_batch",
            "permuted_interleaved_batch",
            "permuted",
        } or (
            layout in {"transposed", "transposed_interleaved_batch"}
            and mode in {"k_swizzled", "m_swizzled"}
        )
        if expected_view:
            from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_layout import (
                _canonicalize_matrix_input,
                _plan_matrix_input,
            )

            matrix_layout = _plan_matrix_input(
                data_hp, dim, square, swizzled, dual=False
            )
            with patch.object(
                torch.Tensor,
                "contiguous",
                side_effect=AssertionError("materialized input"),
            ):
                matrix_input = _canonicalize_matrix_input(data_hp, matrix_layout)
            self.assertEqual(matrix_input.data_ptr(), data_hp.data_ptr())
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4 if swizzled else SwizzleType.NO_SWIZZLE
        )
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": dim},
            swizzle_type=swizzle_type,
            scaling_type_use_square_block_size=square,
        )
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)
        self.assertEqual(qdata.shape, data_hp.shape)
        self.assertEqual(qdata.stride()[dim[-1]], 1)
        self.assertEqual(scales.shape, scales_ref.shape)
        batch_axes = tuple(
            sorted(
                (axis for axis in range(data_hp.ndim) if axis not in axes),
                key=lambda axis: -data_hp.stride(axis),
            )
        )
        output_order = (*batch_axes, *dim)
        self.assertTrue(qdata.permute(output_order).is_contiguous())
        self.assertTrue(scales.permute(output_order).is_contiguous())

    @parametrize(
        "dim,is_square_scaling,is_scale_swizzled",
        (
            subtest(((1, 2), False, False), name="dim_k_compact"),
            subtest(((1, 2), False, True), name="dim_k_swizzled"),
            subtest(((2, 1), False, True), name="dim_m_swizzled"),
            subtest(((1, 2), True, True), name="dim_k_square"),
        ),
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_batch_axis_transpose_strides(
        self, dim, is_square_scaling, is_scale_swizzled, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn(2, 256, 128, 3, dtype=torch.bfloat16, device=device)
        swizzle = (
            SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
        )
        kwargs = {
            **_MXFP8_KWARGS,
            "dim": dim,
            "swizzle_type": swizzle,
            "scaling_type_use_square_block_size": is_square_scaling,
        }
        outputs = F.quantize_tensor(input, **kwargs)
        transposed_outputs = F.quantize_tensor(input.transpose(0, 3), **kwargs)
        for output, transposed in zip(outputs, transposed_outputs, strict=True):
            expected = output.transpose(0, 3)
            self.assertEqual(transposed, expected)
            self.assertEqual(transposed.stride(), expected.stride())

    @parametrize("layout", _MXFP8_DUAL_LAYOUTS)
    @skipIfNoCuteDSL
    def test_to_mx_rceil_nd_dual(self, layout, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp, axes = _get_tensor_and_quant_dims(layout, device)
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference_nd(
            data_hp, axes, False, True
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference_nd(
            data_hp, axes[::-1], False, True
        )
        # Protect current no-copy fast paths, not a public API guarantee.
        if layout in {
            "leading",
            "leading_reverse",
            "interleaved_batch",
            "interleaved_rank3",
            "interleaved_singleton",
            "interleaved_multi_batch",
            "permuted_interleaved_batch",
            "transposed_interleaved_batch",
        }:
            from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
                _canonicalize_matrix_input,
                _plan_matrix_input,
            )

            matrix_layout = _plan_matrix_input(data_hp, axes, False, True, dual=True)
            with patch.object(
                torch.Tensor,
                "contiguous",
                side_effect=AssertionError("materialized input"),
            ):
                matrix_input = _canonicalize_matrix_input(data_hp, matrix_layout)
            self.assertEqual(matrix_input.data_ptr(), data_hp.data_ptr())
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": axes},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        self.assertEqual(qdata_k, qdata_k_ref)
        self.assertEqual(scales_k, scales_k_ref)
        self.assertEqual(qdata_m, qdata_m_ref)
        self.assertEqual(scales_m, scales_m_ref)
        self.assertEqual(qdata_k.shape, data_hp.shape)
        self.assertEqual(qdata_m.shape, data_hp.shape)
        self.assertEqual(qdata_k.stride()[axes[-1]], 1)
        self.assertEqual(qdata_m.stride()[axes[0]], 1)
        batch_axes = tuple(
            sorted(
                (axis for axis in range(data_hp.ndim) if axis not in axes),
                key=lambda axis: -data_hp.stride(axis),
            )
        )
        for output in (qdata_k, scales_k):
            self.assertTrue(output.permute((*batch_axes, *axes)).is_contiguous())
        for output in (qdata_m, scales_m):
            self.assertTrue(output.permute((*batch_axes, *axes[::-1])).is_contiguous())

    @skipIfNoCuteDSL
    def test_to_mx_rceil_strided_input_fallback(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 256, dtype=torch.bfloat16, device=device)[:, ::2]
        qdata_ref, scales_ref = _quantize_mxfp8_reference_nd(
            data_hp, (-2, -1), False, True
        )
        qdata, scales = F.quantize_tensor(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata, qdata_ref)
        self.assertEqual(scales, scales_ref)

    @parametrize(
        "invalid",
        ("dtype", "size", "alignment", "grid_z", "folded_grid_z", "tma_dim", "grid_y"),
    )
    @skipIfNoCuteDSL
    def test_to_mx_invalid_input_before_materialization(self, invalid, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        dim = (0, 1)
        from torch._native.ops.quantize_tensor.blockscaled_tma.blockscaled_tma_impl import (
            _blockscaled_tma_impl,
        )

        if invalid == "size":
            data_hp = torch.empty((64, 96), dtype=torch.bfloat16, device=device)[:, ::2]
            error = "quantized axis 1 must be divisible by 32; got size 48"
        elif invalid == "alignment":
            data_hp = torch.empty(64 * 48 + 1, dtype=torch.bfloat16, device=device)[
                1:
            ].view(64, 48)
            error = "quantized axis 1 must be divisible by 32; got size 48"
        elif invalid == "grid_z":
            data_hp = torch.empty(1, dtype=torch.bfloat16, device=device).expand(
                64, 128, 65536
            )
            error = r"batch axes \(2,\) have sizes \(65536,\); product 65536 exceeds CUDA grid.z limit 65535"
        elif invalid == "folded_grid_z":
            data_hp = torch.empty(1, dtype=torch.bfloat16, device=device).expand(
                300, 64, 300, 128
            )
            dim = (1, 3)
            error = r"batch axes \(0, 2\) have sizes \(300, 300\); product 90000 exceeds CUDA grid.z limit 65535"
        elif invalid == "tma_dim":
            data_hp = torch.empty(1, dtype=torch.bfloat16, device=device).expand(
                64, 2**32 + 32, 2
            )
            error = "input axis 1 has size 4294967328; maximum TMA dimension size is 4294967296"
        elif invalid == "grid_y":
            data_hp = torch.empty(1, dtype=torch.bfloat16, device=device).expand(
                65536 * 128, 32, 2
            )
            error = r"grid=\(1, 65536, 2\)"
        else:
            data_hp = torch.empty((64, 128), dtype=torch.float64, device=device)[:, ::2]
            error = "blockscaled TMA supports only bf16, fp16, and fp32 input"
        with (
            patch.object(
                torch.Tensor, "contiguous", side_effect=AssertionError("copied input")
            ),
            patch.object(
                torch.Tensor, "clone", side_effect=AssertionError("cloned input")
            ),
        ):
            with self.assertRaisesRegex(ValueError, error):
                _blockscaled_tma_impl(data_hp, dim, False, True)

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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (1, 2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (1, 2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
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
        qdata, scales = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": (1, 2)},
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
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
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
        )
        outputs = F.quantize_tensor(
            data_hp,
            **{**_MXFP8_KWARGS, "dim": dim},
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
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
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

    @parametrize(
        "layout",
        (
            "2d",
            "leading",
            "interleaved",
            "trailing",
            "transposed",
            "fallback",
            "unaligned",
        ),
    )
    @parametrize("mode", ("k", "m", "compact", "square", "m_compact", "m_square"))
    @skipIfNoCuteDSL
    def test_quantize_tensor_nd_meta(self, layout, mode, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input, axes = _make_nd_quantize_input(layout, device)
        dim = (
            axes[::-1]
            if mode in ("m", "m_compact", "m_square")
            or (mode == "square" and layout == "transposed")
            else axes
        )
        if layout in ("leading", "interleaved"):
            dim = (dim[0] - input.ndim, dim[1])
        square = mode.endswith("square")
        swizzle = (
            SwizzleType.NO_SWIZZLE
            if mode.endswith("compact")
            else SwizzleType.SWIZZLE_32_4_4
        )
        actual = F.quantize_tensor(
            input,
            **{**_MXFP8_KWARGS, "dim": dim},
            swizzle_type=swizzle,
            scaling_type_use_square_block_size=square,
        )
        meta_input = torch.empty_strided(
            input.shape, input.stride(), dtype=input.dtype, device="meta"
        )
        meta = F.quantize_tensor(
            meta_input,
            **{**_MXFP8_KWARGS, "dim": dim},
            swizzle_type=swizzle,
            scaling_type_use_square_block_size=square,
        )
        for result, meta_result in zip(actual, meta, strict=True):
            self.assertEqual(result.shape, meta_result.shape)
            self.assertEqual(result.stride(), meta_result.stride())

    @parametrize(
        "layout",
        (
            "2d",
            "leading",
            "interleaved",
            "trailing",
            "transposed",
            "fallback",
            "unaligned",
        ),
    )
    @parametrize("reversed_axes", (False, True))
    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_nd_meta(self, layout, reversed_axes, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input, axes = _make_nd_quantize_input(layout, device)
        dim = axes[::-1] if reversed_axes else axes
        if layout in ("leading", "interleaved"):
            dim = (dim[0], dim[1] - input.ndim)
        expected = (
            *_quantize_mxfp8_reference_nd(input, dim, False, True),
            *_quantize_mxfp8_reference_nd(input, dim[::-1], False, True),
        )
        kwargs = {**_MXFP8_KWARGS, "dim": dim}
        actual = F.quantize_tensor_dual(
            input, **kwargs, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        meta_input = torch.empty_strided(
            input.shape, input.stride(), dtype=input.dtype, device="meta"
        )
        meta = F.quantize_tensor_dual(
            meta_input, **kwargs, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        for result, reference, meta_result in zip(actual, expected, meta, strict=True):
            self.assertEqual(result, reference)
            self.assertEqual(result.shape, meta_result.shape)
            self.assertEqual(result.stride(), reference.stride())
            self.assertEqual(result.stride(), meta_result.stride())

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

    @skipIfNoCuteDSL
    def test_quantize_tensor_nd_compile(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input, _ = _make_nd_quantize_input("interleaved", device)

        def fn(x):
            return (
                *F.quantize_tensor(
                    x,
                    **{**_MXFP8_KWARGS, "dim": (1, 3)},
                    swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                ),
                *F.quantize_tensor_dual(
                    x,
                    **{**_MXFP8_KWARGS, "dim": (3, 1)},
                    swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                ),
            )

        expected = fn(input)
        actual = torch.compile(fn, fullgraph=True)(input)
        for result, reference in zip(actual, expected, strict=True):
            self.assertEqual(result, reference)
            self.assertEqual(result.stride(), reference.stride())


@instantiate_parametrized_tests
class TestQuantizeTensorMeta(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_quantize_tensor_argument_names(self):
        # ensure that relevant argument names between F.quantize_tensor and
        # F.quantize_tensor_dual stay in sync
        self.assertEqual(F.ScalingAlgorithm.MXFP_E8M0_RU.value, 0)
        public_names = tuple(inspect.signature(F.quantize_tensor).parameters)
        dual_names = tuple(inspect.signature(F.quantize_tensor_dual).parameters)
        native = torch.ops.aten._quantize_tensor.default._schema.arguments
        native_dual = torch.ops.aten._quantize_tensor_dual.default._schema.arguments
        self.assertEqual(
            public_names,
            (
                "input",
                "dim",
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
            self.assertEqual(params[5].default, inspect.Parameter.empty)
        for schema_args in (native, native_dual):
            self.assertEqual(str(schema_args[1].type), "List[int]")
            self.assertEqual(str(schema_args[4].type), "int")
            self.assertFalse(schema_args[0].kwarg_only)
            for arg in schema_args[1:]:
                self.assertTrue(arg.kwarg_only)
            self.assertFalse(schema_args[5].has_default_value())

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
        expected_scale_shape = _get_scale_shape((rows, cols), swizzle)
        self.assertEqual(tuple(qdata.shape), (rows, cols))
        self.assertEqual(tuple(scales.shape), expected_scale_shape)
        self.assertEqual(qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)

    def test_dispatch_transposed_shape(self):
        input = torch.empty((32, 160), dtype=torch.float16, device="meta").t()
        qdata, scales = torch._quantize_tensor(  # pyrefly: ignore[missing-attribute]
            input,
            dim=(-2, -1),
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        self.assertEqual(tuple(qdata.shape), (160, 32))
        self.assertEqual(tuple(scales.shape), (1, 1024))

    def test_dual_meta_shapes(self):
        input = torch.empty((160, 288), dtype=torch.bfloat16, device="meta")
        outputs = F.quantize_tensor_dual(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(
            tuple(tuple(output.shape) for output in outputs),
            ((160, 288), (1, 3072), (160, 288), (3072, 1)),
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

    @parametrize("dim", ((0, 0), (0, -2), (2, 1), (-3, 1), (0,), (0, 1, 2)))
    @parametrize("dual", (False, True))
    def test_invalid_dim(self, dim, dual):
        input = torch.empty((64, 128), dtype=torch.bfloat16, device="meta")
        fn = F.quantize_tensor_dual if dual else F.quantize_tensor
        with self.assertRaisesRegex(
            ValueError, "quantization dim|distinct quantization axes"
        ):
            fn(
                input,
                **{**_MXFP8_KWARGS, "dim": dim},
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            )

    def test_dim_required(self):
        input = torch.empty((64, 128), dtype=torch.bfloat16, device="meta")
        for fn in (F.quantize_tensor, F.quantize_tensor_dual):
            with self.assertRaisesRegex(TypeError, "dim"):
                fn(
                    input,
                    **{
                        key: value
                        for key, value in _MXFP8_KWARGS.items()
                        if key != "dim"
                    },
                    swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                )

    def test_invalid_matrix_size(self):
        input = torch.empty((48, 80), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(
            ValueError, "quantized axis 1 must be divisible by 32; got size 80"
        ):
            F.quantize_tensor(
                input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
            )
        with self.assertRaisesRegex(
            ValueError, "quantized axis 0 must be divisible by 32; got size 48"
        ):
            F.quantize_tensor_dual(
                input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
            )


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
