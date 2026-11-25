"""Input checks shared by ``aten::_scaled_mm_mxfp6``'s CuTeDSL implementation and its meta.

No QuACK or CuTeDSL imports, so the meta can use them without loading either.
Shape checks go through ``torch._check_value`` so symbolic sizes defer them
instead of guarding during tracing.
"""

import torch
from torch._subclasses.fake_tensor import maybe_get_fake_device
from torch.fx.experimental.symbolic_shapes import sym_or


def _device(t: torch.Tensor) -> torch.device:
    # Inside FakeTensor meta dispatch ``.device`` reads meta.
    return maybe_get_fake_device(t) or t.device


def _scale_blocks(
    scale: torch.Tensor, rows, k, device: torch.device
) -> tuple[object, ...]:
    """QuACK's (ceil(rows/128), K/128, 32, 4, 4) blocks for a flat SWIZZLE_32_4_4 scale."""
    # uint8 is the e8m0 bytes, which QuACK's container also accepts.
    if (
        scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
        or _device(scale) != device
    ):
        raise ValueError(
            f"MXFP6 scaled_mm expected float8_e8m0fnu (or uint8) scales on {device}, "
            f"got {scale.dtype} on {_device(scale)}"
        )
    blocks = ((rows + 127) // 128, k // 128, 32, 4, 4)
    expected = blocks[0] * blocks[1] * 512
    torch._check_value(
        scale.numel() == expected,
        lambda: f"MXFP6 scaled_mm expected a scale of {expected} elements, "
        f"got {scale.numel()}",
    )
    torch._check_value(
        scale.is_contiguous(),
        lambda: f"MXFP6 scaled_mm expected a contiguous scale, got strides {scale.stride()}",
    )
    return blocks


def validate_mxfp6_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Check [M, K/4] x [K/4, N] float6_e2m3fn_x4 operands, their scales and ``out``.

    Returns each scale's QuACK block shape. Checks that need the device itself
    (architecture, pointer alignment) stay with the kernel.
    """
    if mat_a.dtype != torch.float6_e2m3fn_x4 or mat_b.dtype != mat_a.dtype:
        raise ValueError(
            "MXFP6 scaled_mm requires float6_e2m3fn_x4 operands, got "
            f"{mat_a.dtype} and {mat_b.dtype}"
        )
    if mat_a.dim() != 2 or mat_b.dim() != 2:
        raise ValueError(
            f"MXFP6 scaled_mm requires 2-D operands, got {mat_a.dim()}-D and "
            f"{mat_b.dim()}-D"
        )
    torch._check_value(
        mat_a.shape[1] == mat_b.shape[0],
        lambda: f"MXFP6 scaled_mm requires [M, K/4] x [K/4, N] operands, got "
        f"{tuple(mat_a.shape)} and {tuple(mat_b.shape)}",
    )
    for operand, k_dim in ((mat_a, 1), (mat_b, 0)):
        # An empty operand's strides are arbitrary.
        torch._check_value(
            sym_or(operand.numel() == 0, operand.stride(k_dim) == 1),
            lambda: "MXFP6 scaled_mm requires operands contiguous along K",
        )
    m, n, k = mat_a.shape[0], mat_b.shape[1], 4 * mat_a.shape[1]
    torch._check_value(
        k % 128 == 0,
        lambda: f"MXFP6 scaled_mm requires K to be a multiple of 128, got {k}",
    )
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP6 scaled_mm requires out_dtype bfloat16 or float32, got {out_dtype}"
        )
    device = _device(mat_a)
    if _device(mat_b) != device:
        raise ValueError(f"MXFP6 scaled_mm expected mat_b on {device}")
    blocks = (
        _scale_blocks(scale_a, m, k, device),
        _scale_blocks(scale_b, n, k, device),
    )
    if out is not None and (out.dtype != out_dtype or _device(out) != device):
        raise ValueError(
            f"MXFP6 scaled_mm expected out to be {out_dtype} on {device}, "
            f"got {out.dtype} on {_device(out)}"
        )
    return blocks
