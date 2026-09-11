# Copyright (c) 2026, Tri Dao.

import itertools
from functools import partial
from typing import Callable, Optional, Type, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cutlass import Int32, Float32

from torch._vendor.quack.compile_utils import make_fake_tensor as fake_tensor
from torch._vendor.quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.gemm_tvm_ffi_utils import div_for_dtype, make_scheduler_args
from torch._vendor.quack.blockscaled.operand import (
    BLOCKSCALED_FORMAT_REGISTRY,
    BlockScaledFormat,
    BlockScaledOperand,
    legacy_format_name,
)
from torch._vendor.quack.blockscaled.quantize import (  # noqa: F401  (pure-torch helpers re-exported)
    FP4_E2M1FN_VALUES,
    QUANTIZERS,
    _COMPILE_KW,
    _fp4_unpacked_to_value,
    dequant_operand,
    pack_scale_2d_to_blocked_contig,
    to_mx_compiled,
    to_mxfp4_compiled,
    to_nvfp4_compiled,
    unpack_scale_blocked_to_2d,
)
from torch._vendor.quack.varlen_utils import VarlenArguments


TORCH_DTYPE_MAP = {
    cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
    cutlass.Float32: torch.float32,
    cutlass.Float8E4M3FN: torch.float8_e4m3fn,
    cutlass.Float8E5M2: torch.float8_e5m2,
    cutlass.Float8E8M0FNU: torch.float8_e8m0fnu,
}

FLOAT8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_dtype_for_cutlass(dtype: Type[cutlass.Numeric]) -> torch.dtype:
    if dtype not in TORCH_DTYPE_MAP:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return TORCH_DTYPE_MAP[dtype]


def _make_fake_tensor_like(tensor: torch.Tensor, dtype: Type[cutlass.Numeric]) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        dtype,
        tensor.shape,
        stride=tensor.stride(),
        assumed_align=16,
    )


def _batch_first(tensor: torch.Tensor) -> torch.Tensor:
    """Batch-last (x, y, l) -> batch-first (l, x, y) view; rank-2 passes through."""
    return tensor.permute(2, 0, 1) if tensor.dim() == 3 else tensor


def _leading_dim_from_stride(tensor: torch.Tensor) -> int:
    # Size-1 dims carry an arbitrary (often 1) stride — e.g. the l=1 batch dim of a
    # batch-first (1, m, k) view — and must not shadow the real contiguous dim.
    for i, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        if stride == 1 and size != 1:
            return i
    for i, stride in enumerate(tensor.stride()):
        if stride == 1:
            return i
    raise ValueError(
        f"Tensor has no unit stride dimension: shape={tensor.shape}, stride={tensor.stride()}"
    )


def _make_compile_tensor_like(
    tensor: torch.Tensor, dtype: Type[cutlass.Numeric], dynamic_layout: bool = False
) -> cute.Tensor:
    compile_tensor = cute.runtime.from_dlpack(tensor)
    compile_tensor.element_type = dtype
    if dynamic_layout:
        marked = compile_tensor.mark_layout_dynamic(leading_dim=_leading_dim_from_stride(tensor))
        if marked is not None:
            compile_tensor = marked
    return compile_tensor


def _make_fake_compact_tensor(
    shape: Tuple[int, ...], dtype: Type[cutlass.Numeric], leading_dim: int
) -> cute.Tensor:
    logical_shape = list(shape)
    if dtype == cutlass.Float4E2M1FN:
        logical_shape[leading_dim] *= 2
    return fake_tensor(
        dtype,
        tuple(logical_shape),
        leading_dim=leading_dim,
        divisibility=div_for_dtype(dtype),
    )


def _fp4_e2m1fn_value_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(FP4_E2M1FN_VALUES, dtype=torch.float32, device=device)


def _pack_fp4_e2m1fn_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical FP4 codes into torch.float4_e2m1fn_x2 storage."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 FP4 codes, got {codes.dtype}")
    packed_shape = (codes.shape[0], ceil_div(codes.shape[1], 2), codes.shape[2])
    packed = torch.empty(packed_shape, dtype=torch.float4_e2m1fn_x2, device=codes.device)
    packed_u8 = packed.view(torch.uint8)
    low = codes[:, 0::2, :]
    high = torch.zeros_like(low)
    high[:, : codes[:, 1::2, :].shape[1], :] = codes[:, 1::2, :]
    packed_u8.copy_(low | (high << 4))
    return packed


def _create_fp4_operand_tensor(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    *,
    init: str,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if is_mode0_major:
        raise ValueError("Float4E2M1FN blockscaled operands must be K-major")
    # (mn, k/2, l) K-major view of a contiguous (l, mn, k/2) buffer; allocating
    # (mn, k/2, l) directly would put stride 1 on L instead of K for l > 1.
    tensor = torch.empty(
        (l, mode0, ceil_div(mode1, 2)), dtype=torch.float4_e2m1fn_x2, device="cuda"
    ).permute(1, 2, 0)
    tensor.view(torch.uint8).zero_()
    if init == "empty":
        return None, tensor
    if init != "normal":
        raise ValueError(f"Unsupported init: {init}")

    magnitudes = torch.randint(0, 8, (mode0, mode1, l), device="cuda", dtype=torch.uint8)
    signs = torch.randint(0, 2, (mode0, mode1, l), device="cuda", dtype=torch.uint8)
    signs = torch.where(magnitudes == 0, torch.zeros_like(signs), signs << 3)
    codes = magnitudes | signs
    tensor.copy_(_pack_fp4_e2m1fn_codes(codes))
    ref = _fp4_e2m1fn_value_table(tensor.device)[codes.long()]
    return ref, tensor


def create_blockscaled_operand_tensor(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: Type[cutlass.Numeric],
    *,
    init: str = "normal",
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if dtype == cutlass.Float4E2M1FN:
        return _create_fp4_operand_tensor(l, mode0, mode1, is_mode0_major, init=init)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_dtype = torch_dtype_for_cutlass(dtype)
    gen_dtype = torch.bfloat16 if torch_dtype in FLOAT8_DTYPES else torch_dtype
    tensor = torch.empty(shape, dtype=gen_dtype, device="cuda")
    if init == "normal":
        tensor.normal_(std=mode1 ** (-0.5))
    elif init != "empty":
        raise ValueError(f"Unsupported init: {init}")
    # Do NOT .contiguous() after .permute() — that would re-materialize with wrong
    # strides (L innermost) and break K-majorness / N-majorness for l > 1.
    # The original (l, mode0/1, mode1/0) is contiguous, and the permuted view has
    # the correct per-mode strides: stride=1 on the intended contiguous dim.
    tensor = tensor.to(torch_dtype).permute(permute_order)
    ref = tensor.float() if init != "empty" else None
    return ref, tensor


def _pack_blockscaled_scales(ref_blocks: torch.Tensor) -> torch.Tensor:
    """Rearrange (mn, sf_k, l) scales into the (l, rm, rk, 32, 4, 4) blocked layout."""
    mn, sf_k, l = ref_blocks.shape
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    packed_6d = torch.zeros((l, rm, rk, 32, 4, 4), dtype=torch.float32, device=ref_blocks.device)
    packed_view = packed_6d.permute(3, 4, 1, 5, 2, 0)  # (32, 4, rm, 4, rk, l)
    m_idx = torch.arange(mn, device=ref_blocks.device)
    k_idx = torch.arange(sf_k, device=ref_blocks.device)
    l_idx = torch.arange(l, device=ref_blocks.device)
    packed_view[
        m_idx[:, None, None] % 32,
        (m_idx[:, None, None] // 32) % 4,
        m_idx[:, None, None] // 128,
        k_idx[None, :, None] % 4,
        k_idx[None, :, None] // 4,
        l_idx[None, None, :],
    ] = ref_blocks
    return packed_6d


def create_blockscaled_scale_tensor(
    l: int,
    mn: int,
    k: int,
    sf_vec_size: int,
    dtype: Type[cutlass.Numeric],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sf_k = ceil_div(k, sf_vec_size)
    if dtype == cutlass.Float8E8M0FNU:
        exponents = torch.randint(0, 2, (mn, sf_k, l), device="cuda", dtype=torch.int32)
        ref_blocks = torch.pow(2.0, exponents.float())
    else:
        ref_blocks = torch.randint(1, 4, (mn, sf_k, l), device="cuda", dtype=torch.int32).float()

    packed_f32 = _pack_blockscaled_scales(ref_blocks)
    packed = torch.empty_like(packed_f32, dtype=torch_dtype_for_cutlass(dtype))
    packed.copy_(packed_f32)
    ref = (
        ref_blocks.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )[:, :k, :]
    return ref, packed


# Legacy short-name view over the descriptor registry (quantizer-backed formats):
# format: (torch operand dtype, torch SF dtype, sf_vec_size). Derived - the
# descriptors in quack.blockscaled.operand are the single source of truth.
BLOCKSCALED_FORMATS = {
    legacy_format_name(f): (f.qdata_dtype, f.scale_dtype, f.sf_vec_size)
    for f in BLOCKSCALED_FORMAT_REGISTRY.values()
    if f.name in QUANTIZERS
}


# compiled (static shapes) so the swizzle fuses into a couple of kernels instead
# of an eager pad/permute/contiguous chain; see the dynamic=False and
# recompile_limit notes in quantize.py (per-wrapper limit where torch supports
# it, global-config fallback on torch < 2.13)
_pack_scale_compiled = torch.compile(pack_scale_2d_to_blocked_contig, **_COMPILE_KW)


def blockscaled_quantize(
    x: torch.Tensor, format: str = "mxfp8", per_tensor_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw-parts quantizer: thin wrapper over
    :meth:`BlockScaledOperand.quantize` (the canonical API).

    Returns ``(q, sf)`` quantizer outputs, NOT a GEMM operand: quack GEMMs take
    only BlockScaledOperand containers, so wrap the parts via
    ``BlockScaledOperand.from_parts(q, sf, format)`` (or quantize with
    :meth:`BlockScaledOperand.quantize` directly). For nvfp4,
    ``per_tensor_scale`` (scalar fp32) folds the global scale into the block
    scales; the per-tensor scale itself is not part of the returned parts, so
    pass it to ``from_parts`` (or use ``BlockScaledOperand.quantize``, which
    stores it and folds it into GEMM alpha automatically).
    """
    t = BlockScaledOperand.quantize(x, format, per_tensor_scale=per_tensor_scale)
    return t.qdata, t.scale


def blockscaled_quantize_dim0(
    x: torch.Tensor, format: str = "mxfp8"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (M, K) bf16/fp32 tensor along M (dim 0) for a blockscaled GEMM
    whose reduction dim is M — the dgrad/wgrad orientations of training linears.

    Returns ``(q, sf)`` quantizer outputs, NOT a GEMM operand:
      q:  (M, K) fp8, same row-major layout as ``x``.
      sf: blocked (rm, rk, 32, 4, 4) scale factors for the logical operand
          (mn=K, sf_k=M/32) - the same tensor serves both usages below.
    Wrap for the GEMM as ``op = BlockScaledOperand.from_parts(q, sf, "mxfp8",
    quant_dim=-2)`` (scales run along M, dim 0): pass ``op`` as an MN-major B
    operand (reduction dim first), or ``op.mT`` as an MN-major A operand.
    """
    from torch._vendor.quack.blockscaled.quantize import to_mx_dim0_compiled

    assert format == "mxfp8", "dim0 quantization currently supports mxfp8 only"
    assert x.ndim == 2, f"expected (M, K), got shape {tuple(x.shape)}"
    sf_vec = BLOCKSCALED_FORMATS[format][2]
    assert x.shape[0] % sf_vec == 0, f"M ({x.shape[0]}) must be divisible by {sf_vec}"
    q, sc = to_mx_dim0_compiled(x, sf_vec)  # sc: (M/32, K)
    sf = _pack_scale_compiled(sc.mT.contiguous().unsqueeze(0))
    return q, sf.squeeze(0)


def scale_view_for_kernel(scale_contig: torch.Tensor, mn: int, sf_k: int, l: int) -> torch.Tensor:
    """Validate a (l, rm, rk, 32, 4, 4) scale tensor and return it unchanged.
    Only the innermost (32, 4, 4) atom (one 512 B tile) must be contiguous
    (strides (16, 4, 1)); outer (L, rm, rk) strides are free — the kernel
    reads them from the passed tensor. This lets callers pass a slice/view of
    a larger buffer with no extra copy. Works for both E8M0 (MX) and E4M3
    (NVFP4)."""
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    assert scale_contig.shape == (l, rm, rk, 32, 4, 4), (
        f"expected (l, rm, rk, 32, 4, 4) = ({l}, {rm}, {rk}, 32, 4, 4), "
        f"got {tuple(scale_contig.shape)}"
    )
    assert scale_contig.stride()[-3:] == (16, 4, 1), (
        f"inner (32, 4, 4) atom must be contiguous with strides (16, 4, 1), "
        f"got {scale_contig.stride()[-3:]}"
    )
    return scale_contig


def scale_blocked_for_cublas(
    scale_contig: torch.Tensor, mn: int, sf_k: int, l_idx: int = 0
) -> torch.Tensor:
    """Flatten a (l, rm, rk, 32, 4, 4) scale tensor to the 1D swizzled layout
    torch._scaled_mm expects. Uses a single l slice."""
    assert scale_contig.is_contiguous() and scale_contig.dim() == 6
    return scale_contig[l_idx].reshape(-1)


def _blockscaled_format_of(ab_dtype, sf_dtype, sf_vec_size) -> str:
    """Identify which quantizer-backed format the (ab, sf, vec) tuple corresponds to.

    Thin shim over :meth:`BlockScaledFormat.from_cutlass_dtypes` returning the legacy short
    name this module's test/bench generators branch on. Packed fp6 is rejected:
    it requires the unified API's separate storage and MMA dtype plumbing.
    """
    from torch._vendor.quack.blockscaled.operand import BlockScaledFormat

    try:
        fmt = BlockScaledFormat.from_cutlass_dtypes(ab_dtype, sf_dtype, sf_vec_size)
    except ValueError:
        fmt = None
    if fmt is None or fmt.name not in {"mxfp8_e4m3", "mxfp8_e5m2", "mxfp4", "nvfp4"}:
        raise ValueError(
            f"init=quant does not support (ab={ab_dtype}, sf={sf_dtype}, vec={sf_vec_size}). "
            f"Supported: MXFP8 (e4m3/e5m2+e8m0+32), MXFP4 (e2m1+e8m0+32), "
            f"NVFP4 (e2m1+e4m3+16)."
        )
    return legacy_format_name(fmt)


def create_blockscaled_operand_quantized(
    l: int,
    mn: int,
    k: int,
    is_mn_major: bool,
    sf_vec_size: int = 32,
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
    *,
    randn_std: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate bf16 randn, quantize to MXFP8/MXFP4/NVFP4 and produce:
    ref:   (mn, k, l) float32 dequantized reference
    q_mkl: (mn, k, l) operand tensor in the layout the quack kernel consumes
           (float8_e4m3fn for fp8 formats; int8 with packed nibbles for fp4)
    scale_contig: (l, rm, rk, 32, 4, 4) contiguous scale storage. Each inner
           (32, 4, 4) atom (512 B) is one 128 MN × 4 K swizzled tile. Byte layout matches
           cuBLAS `to_blocked`. Pass directly to the quack kernel, or use
           `scale_blocked_for_cublas` for cuBLAS.
    """
    fmt = _blockscaled_format_of(ab_dtype, sf_dtype, sf_vec_size)
    is_mxfp8 = fmt in ("mxfp8", "mxfp8_e5m2")
    if is_mn_major and not is_mxfp8:
        raise NotImplementedError(
            f"is_mn_major=True is only supported for MXFP8 (tcgen05 MMA requires "
            f"K-major for MXFP4/NVFP4 operands); got fmt={fmt}"
        )
    assert k % sf_vec_size == 0, f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size})"
    sf_k = k // sf_vec_size
    std = randn_std if randn_std is not None else k**-0.5

    x_hp = (torch.randn(l, mn, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    x_flat = x_hp.view(l * mn, k)

    if is_mxfp8:
        to_fp8 = to_mx_compiled if fmt == "mxfp8" else QUANTIZERS["mxfp8_e5m2"][1]
        q_flat, scale_2d = to_fp8(x_flat, sf_vec_size)  # (l*mn, k), (l*mn, sf_k)
        if is_mn_major:
            # Operand: (mn, k, l) MN-major. Start from (l, mn, k) contig, transpose
            # to (l, k, mn) contig, then permute to (mn, k, l) with strides (1, mn, mn*k).
            q_mkl = (
                q_flat.view(l, mn, k).transpose(1, 2).contiguous().permute(2, 1, 0)
            )  # strides (1, mn, mn*k)
        else:
            # Operand: (mn, k, l) K-major VIEW of contiguous (l, mn, k).
            # Do NOT call .contiguous() here — that would materialize as (mn, k, l) row-major,
            # making L the innermost stride=1 dim and BREAKING K-majorness for l > 1.
            q_mkl = q_flat.view(l, mn, k).contiguous().permute(1, 2, 0)  # strides (k, 1, mn*k)
        q_vals = q_flat.float().view(l, mn, k)
        scale_vals = scale_2d.float().view(l, mn, sf_k).repeat_interleave(sf_vec_size, dim=-1)
        ref_mkl = (q_vals * scale_vals).permute(1, 2, 0).contiguous()
        scale_2d = scale_2d.view(l, mn, sf_k)
    elif fmt in ("mxfp4", "nvfp4"):
        if fmt == "mxfp4":
            q_packed, scale_2d = to_mxfp4_compiled(x_flat, sf_vec_size)  # (l*mn, k/2), (l*mn, sf_k)
        else:
            q_packed, scale_2d, _pts = to_nvfp4_compiled(x_flat, sf_vec_size, None)
        # q_packed is uint8, two 4-bit codes per byte (low nibble=even K, high=odd K).
        # Decode for ref: code -> {0,.5,1,1.5,2,3,4,6,-0,-.5,...} via lookup.
        codes_lo = (q_packed & 0x0F).view(l, mn, k // 2)
        codes_hi = ((q_packed >> 4) & 0x0F).view(l, mn, k // 2)
        vals_lo = _fp4_unpacked_to_value(codes_lo)  # (l, mn, k/2)
        vals_hi = _fp4_unpacked_to_value(codes_hi)
        q_values = torch.stack([vals_lo, vals_hi], dim=-1).reshape(l, mn, k)  # interleave back
        scale_vals = scale_2d.float().view(l, mn, sf_k).repeat_interleave(sf_vec_size, dim=-1)
        ref_mkl = (q_values * scale_vals).permute(1, 2, 0).contiguous()
        # Kernel operand: (mn, k/2, l) K-major view (no post-contiguous!)
        q_mkl = (
            q_packed.view(l, mn, k // 2).contiguous().permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        )
        scale_2d = scale_2d.view(l, mn, sf_k)
    scale_contig = pack_scale_2d_to_blocked_contig(scale_2d)
    return ref_mkl, q_mkl, scale_contig


def create_blockscaled_varlen_m_operands(
    num_experts: int,
    m_per: int,
    n: int,
    k: int,
    sf_vec_size: int,
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
    *,
    randn_std: Optional[float] = None,
    seqlens_m: Optional[list] = None,
    b_major: str = "k",
):
    """Generate bf16 randn + quantize for a varlen_m blockscaled GEMM.

    Per-expert seqlens may be arbitrary (not required to be multiples of 128).
    SF is stored with tile-aligned per-batch padding: each expert `i`'s scales
    occupy `ceildiv(m_i, 128) * 128` rows at offset
    `(cu_seqlens_m[i] + i * 128) // 128 * 128` in the padded scale buffer.
    The kernel decodes via `VarlenManager.offset_batch_SFA` which applies the
    same formula.

    Returns (a_ref, b_ref, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_m):
      a_ref: (total_m, k) fp32 dequantized
      b_ref: (num_experts, n, k) fp32 dequantized
      qa:   (total_m, k_storage) 2D K-major quantized operand
      qb:   (n, k_storage, num_experts) 3D K-major quantized operand
      a_sc_contig: (1, total_padded_rm, rk, 32, 4, 4) — M-padded SFA (tile-aligned per batch).
        total_padded_rm = ((total_m + num_experts * 128) // 128).
      b_sc_contig: (num_experts, rn, rk, 32, 4, 4) — regular per-expert SFB.
      cu_seqlens_m: (num_experts+1,) int32

    Supports all kernel-ready SM100 formats. Packed fp4/fp6 formats require
    b_major="k". NVFP4 uses no per-tensor scale here (it would just fold into
    alpha).
    """
    assert k % sf_vec_size == 0
    if seqlens_m is None:
        seqlens_m = [m_per] * num_experts
    assert len(seqlens_m) == num_experts, (
        f"seqlens_m length {len(seqlens_m)} != num_experts {num_experts}"
    )
    total_m = int(sum(seqlens_m))
    std = randn_std if randn_std is not None else k**-0.5
    sf_k = k // sf_vec_size

    fmt = BlockScaledFormat.from_cutlass_dtypes(ab_dtype, sf_dtype, sf_vec_size)
    if fmt.is_packed:
        assert b_major == "k", f"{fmt.name} requires K-major operands, got {b_major=!r}"

    def quantize(x2d):
        """(rows, k) bf16 -> packed qdata, 2D scales, and dequantized reference."""
        if fmt.name in ("mxfp8_e4m3", "mxfp8_e5m2"):
            q, sc = QUANTIZERS[fmt.name][1](x2d, sf_vec_size)
            vals = q.float()
        elif fmt.name in ("mxfp4", "nvfp4"):
            if fmt.name == "mxfp4":
                q_packed, sc = to_mxfp4_compiled(x2d, sf_vec_size)
            else:
                q_packed, sc, _ = to_nvfp4_compiled(x2d, sf_vec_size, None)
            q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
            vals = dequant_operand(q, fmt)
        elif fmt.name in ("mxfp6_e2m3_packed", "mxfp6_e3m2_packed"):
            q, sc = QUANTIZERS[fmt.name][1](x2d, sf_vec_size)
            vals = dequant_operand(q, fmt)
        else:
            raise NotImplementedError(f"varlen_m operand generation does not support {fmt.name}")
        ref = vals * sc.float().repeat_interleave(sf_vec_size, dim=-1)
        return q, sc, ref

    # Quantize A: (total_m, k) bf16 -> (total_m, k_storage) K-major.
    # A data itself is stored packed (no per-expert padding); only SFA is padded.
    a_hp = (torch.randn(total_m, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    qa, sa_2d, a_ref = quantize(a_hp)

    # Build padded SFA storage (tile-aligned per-batch). Each expert's m_i rows of
    # scales are written at padded tile offset `cu_seqlens[i] // 128 + i`.
    # Allocation: `ceildiv(total_m, 128) + (L - 1)` tiles — proven sufficient
    # in AI/varlen_blockscaled_sf_layout.md (proof 2's "tighter alternative").
    # Matches `total_m // 128 + L` when total_m % 128 > 0; 1 tile smaller
    # when total_m is an exact multiple of 128.
    tile = 128
    total_padded_rm = (total_m + tile - 1) // tile + (num_experts - 1)
    total_padded_m = total_padded_rm * tile
    sa_2d_padded = torch.zeros(total_padded_m, sf_k, dtype=sa_2d.dtype, device=sa_2d.device)
    offset = 0
    for i, m_i in enumerate(seqlens_m):
        offset_padded = (offset // tile + i) * tile
        sa_2d_padded[offset_padded : offset_padded + m_i] = sa_2d[offset : offset + m_i]
        offset += m_i
    a_sc_contig = pack_scale_2d_to_blocked_contig(sa_2d_padded.view(1, total_padded_m, sf_k))

    # Quantize B: (num_experts, n, k) bf16 -> (n, k_storage, num_experts).
    # b_major selects k-major or n-major (8-bit formats only).
    assert b_major in ("k", "n"), f"b_major must be 'k' or 'n', got {b_major!r}"
    b_hp = (torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda") * std).contiguous()
    qb_flat, sb_2d, b_ref_flat = quantize(b_hp.view(num_experts * n, k))
    kb = qb_flat.shape[-1]
    if b_major == "k":
        qb = (
            qb_flat.view(num_experts, n, kb).contiguous().permute(1, 2, 0)
        )  # (n, kb, l) stride (kb, 1, n*kb)
    else:
        qb = (
            qb_flat.view(num_experts, n, kb).transpose(1, 2).contiguous().permute(2, 1, 0)
        )  # (n, k, l) stride (1, n, n*k)
    b_sc_contig = pack_scale_2d_to_blocked_contig(sb_2d.view(num_experts, n, sf_k))
    b_ref = b_ref_flat.view(num_experts, n, k)

    cu_seqlens_m = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_m)), dtype=torch.int32, device="cuda"
    )
    return a_ref, b_ref, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_m


def create_blockscaled_varlen_k_operands(
    num_experts: int,
    k_per: int,
    m: int,
    n: int,
    sf_vec_size: int,
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
    *,
    randn_std: Optional[float] = None,
    seqlens_k: Optional[list] = None,
    sf_pad_byte: int = 0,
    b_dtype: Optional[Type[cutlass.Numeric]] = None,
):
    """Generate bf16 randn + quantize for a varlen_k blockscaled GEMM.

    Pass b_dtype != ab_dtype for mixed-precision mxf8f6f4 (fp8 pairs only:
    varlen_k needs m-major A / n-major B, and packed sub-byte operands must be
    K-major).

    Per-expert `k_i` is arbitrary (any positive int): neither `sf_vec_size` nor
    `sf_vec_size * 4` (= 128 for MXFP8) alignment is required. A non-multiple-of-32
    `k_i` just means the expert's last scale block covers a partial chunk; the
    kernel's ragged value TMA zero-fills beyond `cu_seqlens_k[i+1]`, so the tail
    contributes exactly 0.
    The SF buffer uses tile-aligned per-batch K padding: each expert `i`'s scales occupy
    `ceildiv(k_i, 128) * 128` bytes worth of K at offset
    `(cu_seqlens_k[i] + i * 128) // 128 * 128` (in source-K units). A and B
    operand data stay packed and unpadded along K — only their SF buffers pad.

    SF pad regions inside each expert's last 512 B atom are loaded by the
    kernel (TMA loads whole atom columns) but never consumed: the mma loop
    skips the MMA instructions for pad k-blocks (one instruction per SF block
    for mxfp8; see `GemmSm100.mma`), so the pad may hold arbitrary bytes —
    including 0xFF (e8m0 NaN). `sf_pad_byte` sets the pad fill so tests can
    poison it deliberately.

    Returns (a_ref_list, b_ref_list, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_k):
      a_ref_list: list of per-expert (m, k_i) fp32 dequantized A.
      b_ref_list: list of per-expert (n, k_i) fp32 dequantized B.
      qa:  (m, total_k) M-major fp8 (stride (1, m)).
      qb:  (n, total_k) N-major fp8 (stride (1, n)).
      a_sc_contig: (1, rm, total_padded_rk, 32, 4, 4) K-padded SFA (tile-aligned per batch).
      b_sc_contig: (1, rn, total_padded_rk, 32, 4, 4) K-padded SFB (tile-aligned per batch).
      cu_seqlens_k: (num_experts+1,) int32.
    """
    fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
    b_dtype = b_dtype if b_dtype is not None else ab_dtype
    if not (
        ab_dtype in fp8_dtypes
        and b_dtype in fp8_dtypes
        and sf_dtype == cutlass.Float8E8M0FNU
        and sf_vec_size == 32
    ):
        raise NotImplementedError(
            f"varlen_k currently only supports MXFP8 e4m3/e5m2 (got a={ab_dtype}, b={b_dtype}, "
            f"sf={sf_dtype}, vec={sf_vec_size}). Packed fp4/fp6 are k-major-only and not wired up."
        )
    if seqlens_k is None:
        seqlens_k = [k_per] * num_experts
    assert len(seqlens_k) == num_experts, (
        f"seqlens_k length {len(seqlens_k)} != num_experts {num_experts}"
    )
    for i, k_i in enumerate(seqlens_k):
        assert k_i > 0, f"seqlens_k[{i}]={k_i} must be positive"
    total_k = int(sum(seqlens_k))
    std = randn_std if randn_std is not None else (max(seqlens_k)) ** -0.5

    from torch._vendor.quack.blockscaled.quantize import to_mx_compiled

    def quantize(mn, k_i, elem_dtype):
        # The quantizer reshapes K into sf_vec_size chunks, so zero-pad k_i up to a
        # multiple of it; zeros never raise a chunk amax, so the real elements
        # quantize identically. Values are sliced back to k_i; scales keep the
        # ceil(k_i / sf_vec_size) blocks (the last one covers a partial chunk).
        k_q = (k_i + sf_vec_size - 1) // sf_vec_size * sf_vec_size
        hp = torch.zeros(mn, k_q, dtype=torch.bfloat16, device="cuda")
        hp[:, :k_i] = torch.randn(mn, k_i, dtype=torch.bfloat16, device="cuda") * std
        q, sc = to_mx_compiled(hp, sf_vec_size, elem_dtype=torch_dtype_for_cutlass(elem_dtype))
        q = q[:, :k_i]
        ref = q.float() * sc.float().repeat_interleave(sf_vec_size, dim=-1)[:, :k_i]
        return q, sc, ref

    a_q_list, a_sc_list, a_ref_list = [], [], []
    b_q_list, b_sc_list, b_ref_list = [], [], []
    for k_i in seqlens_k:
        a_q, a_sc, a_ref = quantize(m, k_i, ab_dtype)
        a_q_list.append(a_q)
        a_sc_list.append(a_sc)
        a_ref_list.append(a_ref)

        b_q, b_sc, b_ref = quantize(n, k_i, b_dtype)
        b_q_list.append(b_q)
        b_sc_list.append(b_sc)
        b_ref_list.append(b_ref)

    # Pack operand data along K: (m, total_k), (n, total_k). varlen_k's
    # ragged TMA descriptors are built for MN-major operands (stride 1 on
    # M/N), so store M-major A and N-major B.
    # cat gives K-major; transpose → contiguous → transpose to get M-major.
    qa = torch.cat(a_q_list, dim=1).t().contiguous().t()  # (m, total_k) stride (1, m)
    qb = torch.cat(b_q_list, dim=1).t().contiguous().t()  # (n, total_k) stride (1, n)
    assert qa.stride() == (1, qa.shape[0])
    assert qb.stride() == (1, qb.shape[0])

    # Pad SFA/SFB per-expert to multiples of 128 source-K (= 4 scales).
    # offset_tile = cu_seqlens[i] // 128 + i (same formula the kernel uses).
    # Allocation = ceildiv(total_k, 128) + (L - 1) tiles (tighter than
    # total_k//128 + L when total_k is a multiple of 128; same otherwise).
    tile = 128  # sf_vec_size * 4
    total_padded_rk = (total_k + tile - 1) // tile + (num_experts - 1)
    total_padded_k = total_padded_rk * tile
    total_padded_sf_k = total_padded_k // sf_vec_size
    sa_2d_padded = torch.full(
        (m, total_padded_sf_k), sf_pad_byte, dtype=torch.uint8, device="cuda"
    ).view(a_sc_list[0].dtype)
    sb_2d_padded = torch.full(
        (n, total_padded_sf_k), sf_pad_byte, dtype=torch.uint8, device="cuda"
    ).view(b_sc_list[0].dtype)
    k_offset = 0
    for i, k_i in enumerate(seqlens_k):
        sf_k_i = (k_i + sf_vec_size - 1) // sf_vec_size
        k_offset_padded = (k_offset // tile + i) * tile
        sf_k_offset_padded = k_offset_padded // sf_vec_size
        sa_2d_padded[:, sf_k_offset_padded : sf_k_offset_padded + sf_k_i] = a_sc_list[i]
        sb_2d_padded[:, sf_k_offset_padded : sf_k_offset_padded + sf_k_i] = b_sc_list[i]
        k_offset += k_i

    a_sc_contig = pack_scale_2d_to_blocked_contig(sa_2d_padded.view(1, m, total_padded_sf_k))
    b_sc_contig = pack_scale_2d_to_blocked_contig(sb_2d_padded.view(1, n, total_padded_sf_k))

    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)), dtype=torch.int32, device="cuda"
    )
    return a_ref_list, b_ref_list, qa, qb, a_sc_contig, b_sc_contig, cu_seqlens_k


def compile_blockscaled_gemm_tvm_ffi(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    d_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    mA: torch.Tensor,
    mB: torch.Tensor,
    mD: torch.Tensor,
    mSFA: torch.Tensor,
    mSFB: torch.Tensor,
    *,
    use_clc_persistence: bool = True,
    varlen_m: bool = False,
    varlen_k: bool = False,
    split_k: int = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> Callable:
    """Compile the SM100 blockscaled GEMM.

    Caller convention is batch-LAST — mA (m, k, l), mB (n, k, l), mD (m, n, l) —
    matching the reference-math einsums. The kernel itself expects batch-FIRST
    tensors and rotates (l, x, y) -> (x, y, l) at trace time
    (GemmBase.rotate_batch_last), so this wrapper converts at the boundary: the
    compile-time fakes are built batch-first, and run(...) passes batch-first
    views (a free .permute). Rank-2 (varlen-flattened) operands and the SF
    tensors pass through untouched (the kernel does not rotate SFA/SFB).

    This direct TVM-FFI path takes plain tensors only (raw qdata + scale buffers
    with explicit cutlass dtypes). BlockScaledOperand operands are
    the quack.gemm_interface layer's job; unwrap before calling this.

    When varlen_m: mA is (total_m, k) K-major, mD is (total_m, n) N-major,
    mB is (n, k, l); run(...) takes an extra cu_seqlens_m tensor.
    When varlen_k: mA is (m, total_k), mB is (n, total_k), mD is (m, n, l);
    run(...) takes an extra cu_seqlens_k tensor.

    split_k > 1 (dense only): block-scaled composes with the dense finalizer-only
    split-K device path with no kernel changes (the SF loads ride the same
    k_tile_start-offset copy list as A/B; the accumulator is already descaled f32
    before the epilogue). run(...) allocates the per-tile completion flag and the f32
    partials workspace per call (mirroring quack.gemm.gemm) and threads them through.
    SERIAL/PARALLEL only — SEPARATE needs a block-scaled reduction-kernel path.
    """
    assert not isinstance(mA, BlockScaledOperand) and not isinstance(mB, BlockScaledOperand), (
        "compile_blockscaled_gemm_tvm_ffi takes plain tensors; unwrap BlockScaledOperand "
        "(use .qdata / .scale) or call quack.gemm"
    )
    device_capacity = get_device_capacity(mA.device)
    if device_capacity[0] not in (10, 11):
        raise RuntimeError("Blockscaled SM100 GEMM requires SM100/SM110")
    assert not (varlen_m and varlen_k), "Only one of varlen_m / varlen_k"
    split_k_mode = SplitKMode(split_k_mode)
    if split_k > 1:
        if varlen_m or varlen_k:
            raise ValueError("block-scaled split_k requires a dense GEMM (no varlen)")
        if split_k_mode == SplitKMode.SEPARATE:
            raise NotImplementedError(
                "block-scaled split_k does not support SEPARATE yet; use SERIAL or PARALLEL"
            )

    # Lazy: this SM100 compile helper must not force the kernel-class import
    # chain (gemm_default_epi -> gemm_sm90) onto every consumer of the
    # blockscaled package (e.g. nvfp4_utils, imported by kernel-side code).
    from torch._vendor.quack.gemm_default_epi import GemmDefaultSm100

    gemm = partial(
        GemmDefaultSm100,
        sf_vec_size=sf_vec_size,
        use_clc_persistence=use_clc_persistence,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )(cutlass.Float32, ab_dtype, mma_tiler_mn, (*cluster_shape_mn, 1))
    # Per-CTA tile shape (post 2-CTA halving): the workspace stripe is exactly
    # cta_tile_m * cta_tile_n f32 elements per output tile. cta_tile_shape_mnk is only
    # populated later in _setup_attributes, so derive from use_2cta_instrs + mma_tiler.
    cta_tile_m = mma_tiler_mn[0] // (2 if gemm.use_2cta_instrs else 1)
    cta_tile_n = mma_tiler_mn[1]
    split_k_compile = split_k > 1  # SERIAL/PARALLEL reach the in-kernel finalize path
    if split_k_compile:
        compile_epi_args = gemm.EpilogueArguments(
            split_k_semaphore=fake_tensor(
                Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int()), leading_dim=1
            ),
            split_k_workspace=fake_tensor(
                Float32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()),
                leading_dim=0,
                divisibility=4,
            ),
        )
    else:
        compile_epi_args = gemm.EpilogueArguments()
    scheduler_args = make_scheduler_args(
        get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
        max_swizzle_size=8,
        tile_count_semaphore=None,
        batch_idx_permute=None,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    from torch._vendor.quack.gemm_tvm_ffi_utils import make_fake_varlen_args

    varlen_args_fake = make_fake_varlen_args(varlen_m, varlen_k, False, None) or VarlenArguments()

    # Fake operand tensors with sym_ints (varlen-aware shapes).
    if varlen_m:
        total_m_sym = cute.sym_int()
        n_sym, l_sym = cute.sym_int(), cute.sym_int()
        # Sub-byte (fp4) operands need the contiguous K extent statically divisible
        # by the packing factor; harmless for 8-bit dtypes.
        k_sym = cute.sym_int(divisibility=div_for_dtype(ab_dtype) if ab_dtype.width < 8 else 1)
        # Detect B's leading (stride-1) dim so n-major B is accepted for varlen_m
        # (mxfp8 only; fp4 is always K-major). A must be K-major for varlen_m —
        # the public API enforces this (see quack/gemm.py) for all dtypes.
        fake_mA = fake_tensor(
            ab_dtype,
            (total_m_sym, k_sym),
            leading_dim=_leading_dim_from_stride(mA),
            divisibility=div_for_dtype(ab_dtype),
        )
        fake_mB = fake_tensor(
            ab_dtype,
            (l_sym, n_sym, k_sym),
            leading_dim=_leading_dim_from_stride(_batch_first(mB)),
            divisibility=div_for_dtype(ab_dtype),
        )
        fake_mD = fake_tensor(
            d_dtype,
            (total_m_sym, n_sym),
            leading_dim=_leading_dim_from_stride(mD),
            divisibility=div_for_dtype(d_dtype),
        )
    elif varlen_k:
        total_k_sym = cute.sym_int()
        m_sym, n_sym, l_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
        # varlen_k uses MN-major A/B convention (stride 1 on M/N axis), but
        # detect from the actual tensor so either layout works.
        fake_mA = fake_tensor(
            ab_dtype,
            (m_sym, total_k_sym),
            leading_dim=_leading_dim_from_stride(mA),
            divisibility=div_for_dtype(ab_dtype),
        )
        fake_mB = fake_tensor(
            ab_dtype,
            (n_sym, total_k_sym),
            leading_dim=_leading_dim_from_stride(mB),
            divisibility=div_for_dtype(ab_dtype),
        )
        fake_mD = fake_tensor(
            d_dtype,
            (l_sym, m_sym, n_sym),
            leading_dim=_leading_dim_from_stride(_batch_first(mD)),
            divisibility=div_for_dtype(d_dtype),
        )
    else:
        # Detect each operand's leading (stride-1) dim so m-major A / n-major B
        # are accepted along with the default k-major. Fakes are batch-first to
        # match the kernel's calling convention (see docstring).
        mA_bf, mB_bf, mD_bf = _batch_first(mA), _batch_first(mB), _batch_first(mD)
        fake_mA = _make_fake_compact_tensor(
            mA_bf.shape, ab_dtype, leading_dim=_leading_dim_from_stride(mA_bf)
        )
        fake_mB = _make_fake_compact_tensor(
            mB_bf.shape, ab_dtype, leading_dim=_leading_dim_from_stride(mB_bf)
        )
        fake_mD = _make_fake_compact_tensor(
            mD_bf.shape, d_dtype, leading_dim=_leading_dim_from_stride(mD_bf)
        )

    if split_k_compile:

        @cute.jit
        def runner(
            a: cute.Tensor,
            b: cute.Tensor,
            d: cute.Tensor,
            sfa: cute.Tensor,
            sfb: cute.Tensor,
            sem: cute.Tensor,
            ws: cute.Tensor,
            varlen_args,
            stream,
        ):
            epi = compile_epi_args._replace(split_k_semaphore=sem, split_k_workspace=ws)
            gemm(a, b, d, None, epi, scheduler_args, varlen_args, stream, sfa, sfb, None)

        compiled = cute.compile(
            runner,
            fake_mA,
            fake_mB,
            fake_mD,
            _make_compile_tensor_like(mSFA, sf_dtype, dynamic_layout=True),
            _make_compile_tensor_like(mSFB, sf_dtype, dynamic_layout=True),
            compile_epi_args.split_k_semaphore,
            compile_epi_args.split_k_workspace,
            varlen_args_fake,
            stream,
            options="--enable-tvm-ffi",
        )

        def run(a, b, d, sfa, sfb):
            # Allocate the per-tile completion flag + f32 partials workspace per call,
            # mirroring quack.gemm.gemm. d is (m, n, l) here.
            num_l = d.shape[2]
            ntile_m = ceil_div(d.shape[0], cta_tile_m)
            ntile_n = ceil_div(d.shape[1], cta_tile_n)
            ntile_m = ceil_div(ntile_m, cluster_shape_mn[0]) * cluster_shape_mn[0]
            ntile_n = ceil_div(ntile_n, cluster_shape_mn[1]) * cluster_shape_mn[1]
            sem = torch.zeros((num_l, ntile_m, ntile_n), dtype=torch.int32, device=d.device)
            alloc = torch.empty if split_k_mode == SplitKMode.SERIAL else torch.zeros
            ws = alloc(
                (num_l, ntile_m, ntile_n, cta_tile_m * cta_tile_n),
                dtype=torch.float32,
                device=d.device,
            )
            compiled(
                _batch_first(a),
                _batch_first(b),
                _batch_first(d),
                sfa,
                sfb,
                # sem/ws are not TileLoad/TileStore epi fields, so the kernel does
                # not rotate them; pass kernel-order views (same as quack.gemm.gemm).
                sem.permute(1, 2, 0),
                ws.permute(3, 1, 2, 0),
                VarlenArguments(),
            )

        return run

    @cute.jit
    def runner(
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        varlen_args,
        stream,
    ):
        gemm(a, b, d, None, compile_epi_args, scheduler_args, varlen_args, stream, sfa, sfb)

    compiled = cute.compile(
        runner,
        fake_mA,
        fake_mB,
        fake_mD,
        _make_compile_tensor_like(mSFA, sf_dtype, dynamic_layout=True),
        _make_compile_tensor_like(mSFB, sf_dtype, dynamic_layout=True),
        varlen_args_fake,
        stream,
        options="--enable-tvm-ffi",
    )

    if varlen_m or varlen_k:

        def run(a, b, d, sfa, sfb, cu_seqlens):
            varlen_args = VarlenArguments(
                mCuSeqlensM=cu_seqlens if varlen_m else None,
                mCuSeqlensK=cu_seqlens if varlen_k else None,
            )
            compiled(_batch_first(a), _batch_first(b), _batch_first(d), sfa, sfb, varlen_args)
    else:

        def run(a, b, d, sfa, sfb):
            compiled(_batch_first(a), _batch_first(b), _batch_first(d), sfa, sfb, VarlenArguments())

    return run


def blockscaled_gemm_reference(
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum(
        "mkl,nkl->mnl",
        torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref),
        torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref),
    )
