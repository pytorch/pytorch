"""Minimal MX / NVFP4 quantization + scale swizzling utilities.

Ported from torchao (BSD-3) to avoid the runtime dependency:
  torchao/prototype/mx_formats/{mx_tensor, nvfp4_tensor, utils, constants}.py
  torchao/prototype/custom_fp_utils.py
  torchao/prototype/mx_formats/kernels.py

All quantizers are pure-PyTorch. Use the `to_mx_compiled` / `to_mxfp4_compiled` /
`to_nvfp4_compiled` module-level handles if you want torch.compile-generated
Triton kernels (much faster on big tensors; one-time compile overhead).

Scale modes: `to_mx` (MXFP8) supports "rceil" (default) and "floor". This is a
deliberate departure from torchao, whose MX default is FLOOR: NVIDIA's MXFP8
pretraining recipe (arXiv:2506.08027) requires round-up (RCEIL) scales for
bf16 loss parity — FLOOR clips block maxima in (448, 512)·2^e, which hurts
especially on gradient tensors. The FP4 quantizers remain FLOOR-only.
"""

import inspect

import torch

F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E4M3_MAX_POW2 = 8
F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
F8E5M2_MAX_POW2 = 15
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255
F32_EXP_BIAS = 127
F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)  # 2**-126
MBITS_F32 = 23
EBITS_F32 = 8

# FP4 E2M1 constants
F4_E2M1_MAX = 6.0
F4_E2M1_MAX_POW2 = 2
F4_E2M1_MAX_INT = 7  # 3-bit magnitude mask
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def to_mx(
    data_hp: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    elem_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """MXFP8 quantization (e4m3 or e5m2 target).

    Args:
        data_hp: (..., K) bf16 or fp32 tensor, contiguous, K % block_size == 0.
        scaling_mode:
            "rceil" (default): scale = 2^ceil(log2(max_abs / fp8_max)), the OCP
                MX hardware-conversion rule — the smallest power of two such
                that the block max never saturates the target fp8 type.
                NVIDIA's MXFP8 pretraining recipe (arXiv:2506.08027) requires
                this for bf16 loss parity.
            "floor": scale = 2^(floor(log2(max_abs)) - max_pow2), torchao's MX
                default; clips block maxima in (fp8_max, 2^(max_pow2+1))·2^e.
                Kept for torchao parity.
        elem_dtype: float8_e4m3fn (default) or float8_e5m2 (fp8_max 448 / 57344).
    Returns:
        qdata: (..., K) elem_dtype
        scale: (..., K // block_size) float8_e8m0fnu
    """
    assert data_hp.dtype in (torch.bfloat16, torch.float32)
    assert data_hp.shape[-1] % block_size == 0
    assert data_hp.is_contiguous()
    assert scaling_mode in ("rceil", "floor")
    assert elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    fp8_max, fp8_max_pow2 = (
        (F8E4M3_MAX, F8E4M3_MAX_POW2)
        if elem_dtype == torch.float8_e4m3fn
        else (F8E5M2_MAX, F8E5M2_MAX_POW2)
    )

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if scaling_mode == "rceil":
        scale_e8m0_biased = _compute_e8m0_scale_rceil(max_abs, fp8_max)
    else:
        scale_e8m0_biased = _compute_e8m0_scale_floor(max_abs, fp8_max_pow2)

    # reconstruct fp32 scale from biased exponent
    scale_fp32 = (torch.bitwise_left_shift(scale_e8m0_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    # avoid 2**-127 being flushed to 0 (pytorch #125557)
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = data_hp / scale_fp32
    # eager fp8 cast is unsaturated; clamp explicitly
    if not torch._dynamo.is_compiling():
        data_lp = torch.clamp(data_lp, min=-fp8_max, max=fp8_max)

    qdata = data_lp.to(elem_dtype).reshape(orig_shape)
    scale = scale_e8m0_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return qdata, scale


def to_mx_e5m2(data_hp: torch.Tensor, block_size: int = 32, scaling_mode: str = "rceil"):
    """MXFP8-e5m2 quantization: :func:`to_mx` with the e5m2 target (fp8_max
    57344), positional-signature-compatible with the QUANTIZERS registry."""
    return to_mx(data_hp, block_size, scaling_mode, elem_dtype=torch.float8_e5m2)


def to_mx_dim0(data_hp: torch.Tensor, block_size: int = 32, scaling_mode: str = "rceil"):
    """MXFP8-e4m3 quantization of a 2D tensor along dim 0 ("columnwise").

    Same numerics as `to_mx` on the transposed input, but without materializing
    a transposed high-precision copy: qdata keeps the input's (M, K) shape and
    row-major layout; only the scale blocks run along M.

    Training GEMMs whose reduction dim is the token axis (wgrad) or the
    out-feature axis (dgrad) need this orientation: MX scale blocks must run
    along the reduction dim, and since SM100 accepts MN-major operands the data
    layout can stay row-major — rowwise and dim0 copies differ in fp8 *values*
    (per-block scales differ), so each must be quantized from the hp source.

    Args:
        data_hp: (M, K) bf16 or fp32 tensor, contiguous, M % block_size == 0.
    Returns:
        qdata: (M, K) float8_e4m3fn, row-major
        scale: (M // block_size, K) float8_e8m0fnu
    """
    assert data_hp.ndim == 2
    assert data_hp.dtype in (torch.bfloat16, torch.float32)
    assert data_hp.shape[0] % block_size == 0
    assert data_hp.is_contiguous()
    assert scaling_mode in ("rceil", "floor")

    m, k = data_hp.shape
    blocks = data_hp.view(m // block_size, block_size, k)
    max_abs = torch.amax(torch.abs(blocks), 1, keepdim=True).to(torch.float32)

    if scaling_mode == "rceil":
        scale_e8m0_biased = _compute_e8m0_scale_rceil(max_abs, F8E4M3_MAX)
    else:
        scale_e8m0_biased = _compute_e8m0_scale_floor(max_abs, F8E4M3_MAX_POW2)

    scale_fp32 = (torch.bitwise_left_shift(scale_e8m0_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = blocks.to(torch.float32) / scale_fp32
    if not torch._dynamo.is_compiling():
        data_lp = torch.clamp(data_lp, min=-F8E4M3_MAX, max=F8E4M3_MAX)

    qdata = data_lp.to(torch.float8_e4m3fn).view(m, k)
    scale = scale_e8m0_biased.view(torch.float8_e8m0fnu).squeeze(1)
    return qdata, scale


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """FP32 -> sub-byte float (uint8, code in low bits). Verbatim from torchao.

    Round-to-nearest-even via magic-adder; saturation on overflow; no NaN.
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp
    return x.to(torch.uint8)


def _pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit uint8 values in pairs: pair (a,b) -> byte (b<<4 | a)."""
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] | uint8_data[1::2] << 4).view(*shape[:-1], shape[-1] // 2)


def pack_uint6(codes: torch.Tensor) -> torch.Tensor:
    """Pack 6-bit codes (one per uint8, low 6 bits) into a little-endian bit
    stream along the last dim: element i occupies bits [6*i, 6*i + 6) of its
    row, so 4 codes pack into 3 bytes as

        b0 = c0 | (c1 << 6);  b1 = (c1 >> 2) | (c2 << 4);  b2 = (c2 >> 4) | (c3 << 2)

    (..., K) uint8 -> (..., 3*K//4) uint8. This is the CUTLASS SubbyteReference
    bit order - the gmem layout the CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B unpack
    tensormap consumes (verified bit-exact against the DSL's fp6 fill kernel).
    """
    assert codes.dtype == torch.uint8
    k = codes.shape[-1]
    assert k % 4 == 0, f"K ({k}) must be divisible by 4 to pack 6-bit codes"
    c = codes.reshape(*codes.shape[:-1], k // 4, 4)
    c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    # uint8 shifts truncate mod 256, keeping exactly the in-byte bits.
    packed = torch.stack([c0 | (c1 << 6), (c1 >> 2) | (c2 << 4), (c2 >> 4) | (c3 << 2)], dim=-1)
    return packed.reshape(*codes.shape[:-1], 3 * (k // 4))


def unpack_uint6(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_uint6`: (..., 3*K//4) uint8 bit stream ->
    (..., K) uint8 codes in the low 6 bits."""
    assert packed.dtype == torch.uint8
    kb = packed.shape[-1]
    assert kb % 3 == 0, f"packed extent ({kb}) must be whole 3-byte groups"
    b = packed.reshape(*packed.shape[:-1], kb // 3, 3)
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    codes = torch.stack(
        [b0 & 0x3F, (b0 >> 6) | ((b1 & 0x0F) << 2), (b1 >> 4) | ((b2 & 0x03) << 4), b2 >> 2],
        dim=-1,
    )
    return codes.reshape(*packed.shape[:-1], kb // 3 * 4)


def _compute_e8m0_scale_floor(max_abs: torch.Tensor, target_max_pow2: int) -> torch.Tensor:
    """Return biased E8M0 scale (uint8) for FLOOR-mode MX quantization."""
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((torch.bitwise_right_shift(max_abs_int32, MBITS_F32)) & 0xFF) - F32_EXP_BIAS
    scale_unbiased = extracted_pow2 - target_max_pow2
    scale_unbiased = torch.clamp(
        scale_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1
    )
    scale_biased = (scale_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)
    scale_biased = torch.where(torch.isnan(max_abs), E8M0_EXPONENT_NAN_VAL, scale_biased)
    return scale_biased


def _compute_e8m0_scale_rceil(max_abs: torch.Tensor, target_max: float) -> torch.Tensor:
    """Return biased E8M0 scale (uint8) for RCEIL-mode MX quantization.

    2^ceil(log2(max_abs / target_max)): round the exact scale up to the next
    power of two, so max_abs / scale <= target_max always (no saturation).
    """
    scale_int32 = (max_abs / target_max).view(torch.int32)
    exp_biased = torch.bitwise_right_shift(scale_int32, MBITS_F32) & 0xFF
    has_frac = (scale_int32 & _n_ones(MBITS_F32)) != 0
    # +1 exponent unless already an exact power of two; a denormal ratio rounds
    # up to 2^-126 (biased 1), keeping the no-saturation guarantee.
    scale_biased = torch.clamp(exp_biased + has_frac.to(torch.int32), max=E8M0_EXPONENT_NAN_VAL)
    scale_biased = scale_biased.to(torch.uint8)
    # inf lands on the sentinel via exp 255; make NaN explicit like FLOOR does
    scale_biased = torch.where(torch.isnan(max_abs), E8M0_EXPONENT_NAN_VAL, scale_biased)
    return scale_biased


def to_mxfp4(x: torch.Tensor, block_size: int = 32):
    """MXFP4 quantization: E2M1 data + E8M0 per-block scales, FLOOR scaling.

    Args:
        x: (..., K) bf16/fp16/fp32, contiguous, K % block_size == 0.
    Returns:
        qdata_packed: uint8, shape (..., K // 2). Two FP4 values per byte
                      (first -> low nibble, second -> high nibble).
        scale: float8_e8m0fnu, shape (..., K // block_size).
    """
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.shape[-1] % block_size == 0
    assert x.is_contiguous()

    orig_shape = x.shape
    data_hp = x.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    scale_biased = _compute_e8m0_scale_floor(max_abs, F4_E2M1_MAX_POW2)
    scale_fp32 = (torch.bitwise_left_shift(scale_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = data_hp / scale_fp32
    data_lp = data_lp.reshape(orig_shape)
    data_lp = _f32_to_floatx_unpacked(data_lp.float(), EBITS_F4_E2M1, MBITS_F4_E2M1)
    data_lp = _pack_uint4(data_lp)

    scale = scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return data_lp, scale


# fp6 max-normal exponents (E2M3: 1.875 * 2^2 = 7.5; E3M2: 1.75 * 2^4 = 28)
F6_E2M3_MAX_POW2 = 2
F6_E3M2_MAX_POW2 = 4


def _to_mx_floatx_unpacked(x: torch.Tensor, block_size: int, ebits: int, mbits: int, max_pow2: int):
    """E8M0-scaled quantization to UNPACKED floatx codes (one code per uint8,
    in the low bits), FLOOR scaling. The code<->value logic is shared; storage
    packing (e.g. :func:`pack_uint6` for fp6) is a separate step on top.

    Returns (codes uint8 (..., K), scale float8_e8m0fnu (..., K // block_size)).
    """
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.shape[-1] % block_size == 0
    assert x.is_contiguous()

    orig_shape = x.shape
    data_hp = x.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    scale_biased = _compute_e8m0_scale_floor(max_abs, max_pow2)
    scale_fp32 = (torch.bitwise_left_shift(scale_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = (data_hp / scale_fp32).reshape(orig_shape)
    codes = _f32_to_floatx_unpacked(data_lp.float(), ebits, mbits)
    scale = scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return codes, scale


def to_mxfp4_byte(x: torch.Tensor, block_size: int = 32):
    """Deprecated MXFP4 byte-container compatibility encoder.

    Produces one E2M1 code per uint8. This representation remains useful for
    old checkpoints and host-side conversion, but blockscaled GEMM requires
    canonical packed ``mxfp4`` storage.
    """
    return _to_mx_floatx_unpacked(x, block_size, EBITS_F4_E2M1, MBITS_F4_E2M1, F4_E2M1_MAX_POW2)


def to_mxfp6_e2m3(x: torch.Tensor, block_size: int = 32):
    """MXFP6-E2M3 quantization using origin/main's byte-container layout."""
    return _to_mx_floatx_unpacked(x, block_size, 2, 3, F6_E2M3_MAX_POW2)


def to_mxfp6_e3m2(x: torch.Tensor, block_size: int = 32):
    """MXFP6-E3M2 quantization using origin/main's byte-container layout."""
    return _to_mx_floatx_unpacked(x, block_size, 3, 2, F6_E3M2_MAX_POW2)


def to_mxfp6_e2m3_packed(x: torch.Tensor, block_size: int = 32):
    """MXFP6-E2M3 quantization: packed 6-bit codes + E8M0 scales.

    Returns (qdata_packed uint8 (..., 3*K//4) - see :func:`pack_uint6` for the
    bit stream layout - and scale float8_e8m0fnu (..., K // block_size)).
    Requires K % 4 == 0 (subsumed by K % block_size == 0 for block_size 32).
    """
    codes, scale = to_mxfp6_e2m3(x, block_size)
    return pack_uint6(codes), scale


def to_mxfp6_e3m2_packed(x: torch.Tensor, block_size: int = 32):
    """MXFP6-E3M2 quantization: packed 6-bit codes + E8M0 scales (see
    :func:`to_mxfp6_e2m3_packed` for the layout)."""
    codes, scale = to_mxfp6_e3m2(x, block_size)
    return pack_uint6(codes), scale


# Compatibility aliases introduced during packed-FP6 development. The
# unversioned functions are the stable origin/main byte-container API.
to_mxfp6_e2m3_byte = to_mxfp6_e2m3
to_mxfp6_e3m2_byte = to_mxfp6_e3m2


def nvfp4_per_tensor_scale(amax: torch.Tensor) -> torch.Tensor:
    """NVFP4 per-tensor scale: amax / (F8E4M3_MAX * F4_E2M1_MAX) = amax / 2688."""
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def to_nvfp4(x: torch.Tensor, block_size: int = 16, per_tensor_scale=None):
    """NVFP4 quantization: E2M1 data + E4M3 per-block scales + optional fp32 per-tensor scale.

    Args:
        x: (..., K) bf16/fp32, contiguous, K % 16 == 0.
        block_size: must be 16.
        per_tensor_scale: scalar fp32 tensor, or None (uses 1.0 / returns unit).
    Returns:
        qdata_packed: uint8, shape (..., K // 2)
        scale: float8_e4m3fn, shape (..., K // 16)
        per_tensor_scale: scalar fp32 tensor (1.0 if None was passed)
    """
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert x.shape[-1] % block_size == 0
    assert x.is_contiguous()
    assert block_size == 16, "NVFP4 requires block_size=16"

    orig_shape = x.shape
    data_hp = x.float().reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), dim=-1)
    block_scale = max_abs / F4_E2M1_MAX

    if per_tensor_scale is None:
        block_scale_fp8 = torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        recip = 1.0 / block_scale_fp8.to(torch.float32)
        returned_pts = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    else:
        scaled = block_scale.to(torch.float32) / per_tensor_scale
        block_scale_fp8 = torch.clamp(scaled, min=E4M3_EPS, max=F8E4M3_MAX).to(torch.float8_e4m3fn)
        recip = (1.0 / per_tensor_scale) / block_scale_fp8.to(torch.float32)
        returned_pts = per_tensor_scale.to(torch.float32)

    data_scaled = data_hp * recip.unsqueeze(-1)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)
    data_lp = _f32_to_floatx_unpacked(data_scaled.float(), EBITS_F4_E2M1, MBITS_F4_E2M1)
    data_lp = _pack_uint4(data_lp)
    return data_lp, block_scale_fp8, returned_pts


# ---------------------------------------------------------------------------
# torch.compile-wrapped fast paths. Generates fused Triton quant kernels via
# Inductor. dynamic=False is load-bearing: with dynamic shapes Inductor emits a
# ~24x slower kernel for this reduce-then-quantize pattern (measured on B300 at
# (65536, 2048) bf16: 254 GB/s dynamic vs 6 TB/s static — near HBM speed).
# Static shapes recompile per distinct shape; a transformer's quantize sites
# see roughly a dozen (per-linear x/dy/w in two orientations), which exceeds
# dynamo's default recompile_limit of 8 — past it, dynamo SILENTLY falls back
# to eager (the 24x-slower path) for new shapes. The limit must be raised via
# the per-wrapper torch.compile kwarg, NOT `torch._dynamo.config.recompile_limit
# = 64`: config assignments are thread-local, and backward-pass shapes compile
# on the autograd worker thread, which would still see the default 8.
#
# The per-wrapper `recompile_limit=` kwarg only exists on torch >= 2.13. Older
# torch must stay importable: on Blackwell with r575 (CUDA 12.9) drivers,
# torch >= 2.13 pins a triton whose ptxas emits CUDA 13.1 cubins the driver
# cannot load (gh-182), so torch 2.11/2.12 is the working combination there.
# On those versions fall back to raising the process-global dynamo limit
# (named cache_size_limit before the recompile_limit rename) — the thread-local
# config semantics arrived alongside the per-wrapper kwarg, so the global
# assignment is visible to autograd-thread compiles there.
# ---------------------------------------------------------------------------
if "recompile_limit" in inspect.signature(torch.compile).parameters:
    _COMPILE_KW = dict(dynamic=False, recompile_limit=64)
else:
    _COMPILE_KW = dict(dynamic=False)
    _dynamo_cfg = torch._dynamo.config
    _limit_name = (
        "recompile_limit" if hasattr(_dynamo_cfg, "recompile_limit") else "cache_size_limit"
    )
    setattr(_dynamo_cfg, _limit_name, max(getattr(_dynamo_cfg, _limit_name), 64))

to_mx_compiled = torch.compile(to_mx, **_COMPILE_KW)
to_mx_e5m2_compiled = torch.compile(to_mx_e5m2, **_COMPILE_KW)
to_mx_dim0_compiled = torch.compile(to_mx_dim0, **_COMPILE_KW)
to_mxfp4_compiled = torch.compile(to_mxfp4, **_COMPILE_KW)
to_nvfp4_compiled = torch.compile(to_nvfp4, **_COMPILE_KW)

# In-repo quantizers by canonical format name: (eager fn, torch.compile'd fn).
# Membership is the single source of "which formats can be quantized in-repo";
# formats without an entry are from_parts-only until an encoder lands.
to_mxfp6_e2m3_compiled = torch.compile(to_mxfp6_e2m3, dynamic=True)
to_mxfp6_e3m2_compiled = torch.compile(to_mxfp6_e3m2, dynamic=True)
to_mxfp6_e2m3_packed_compiled = torch.compile(to_mxfp6_e2m3_packed, dynamic=True)
to_mxfp6_e3m2_packed_compiled = torch.compile(to_mxfp6_e3m2_packed, dynamic=True)
to_mxfp4_byte_compiled = torch.compile(to_mxfp4_byte, dynamic=True)
to_mxfp6_e2m3_byte_compiled = to_mxfp6_e2m3_compiled
to_mxfp6_e3m2_byte_compiled = to_mxfp6_e3m2_compiled

QUANTIZERS = {
    "mxfp8_e4m3": (to_mx, to_mx_compiled),
    "mxfp8_e5m2": (to_mx_e5m2, to_mx_e5m2_compiled),
    "mxfp4": (to_mxfp4, to_mxfp4_compiled),
    "mxfp4_byte": (to_mxfp4_byte, to_mxfp4_byte_compiled),
    "mxfp6_e2m3": (to_mxfp6_e2m3, to_mxfp6_e2m3_compiled),
    "mxfp6_e3m2": (to_mxfp6_e3m2, to_mxfp6_e3m2_compiled),
    "mxfp6_e2m3_packed": (to_mxfp6_e2m3_packed, to_mxfp6_e2m3_packed_compiled),
    "mxfp6_e3m2_packed": (to_mxfp6_e3m2_packed, to_mxfp6_e3m2_packed_compiled),
    "nvfp4": (to_nvfp4, to_nvfp4_compiled),
}


def _ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Swizzle a (H, W) e8m0 scale tensor into the 128x4 blocked layout
    cuBLAS expects for MXFP8 _scaled_mm. Returns a 1-D flat tensor of size
    32*ceil(H/128) * 16*ceil(W/4)."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# -- Pure-torch operand/scale-layout helpers ---------------------------------
#
# These live here (not in blockscaled/utils.py) so that BlockScaledOperand
# quantize/dequantize and the scale-layout math are usable without importing
# the CuTe-DSL kernel stack. blockscaled/utils.py re-exports them.

FP4_E2M1FN_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_FP4_E2M1_CODE_TO_VALUE = torch.tensor(FP4_E2M1FN_VALUES, dtype=torch.float32)


def _fp4_unpacked_to_value(codes_u8: torch.Tensor) -> torch.Tensor:
    """Convert FP4 E2M1 codes in [0,16) to signed float values via table lookup.
    Code layout: bit 3 = sign, bits 0-2 = magnitude index into {0,.5,1,1.5,2,3,4,6}."""
    table = _FP4_E2M1_CODE_TO_VALUE.to(codes_u8.device)
    return table[codes_u8.long()]


def floatx_unpacked_to_value(codes: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Decode unpacked floatx codes (sign | ebits | mbits in the low bits of a
    uint8) to float32 values. Inverse of :func:`_f32_to_floatx_unpacked` on
    representable values."""
    sign = 1.0 - 2.0 * ((codes >> (ebits + mbits)) & 1).float()
    exp = ((codes >> mbits) & ((1 << ebits) - 1)).float()
    man = (codes & ((1 << mbits) - 1)).float()
    bias = 2 ** (ebits - 1) - 1
    normal = torch.exp2(exp - bias) * (1.0 + man / 2**mbits)
    subnormal = torch.exp2(torch.tensor(1.0 - bias, device=codes.device)) * (man / 2**mbits)
    return sign * torch.where(exp > 0, normal, subnormal)


# Low-bit code shape by CuTe-DSL element type. Storage layout decides whether
# the uint8 tensor is decoded directly (legacy byte container) or unpacked
# first (packed_lsb_v1); the format name alone is deliberately insufficient.
_FLOATX_CODE_BITS = {
    "Float4E2M1FN": (2, 1),
    "Float6E2M3FN": (2, 3),
    "Float6E3M2FN": (3, 2),
}


def dequant_operand(x: torch.Tensor, fmt=None) -> torch.Tensor:
    """Dequantize an operand tensor to float32 values (without scale factors).

    fp8 tensors convert directly; ``float4_e2m1fn_x2`` tensors unpack two codes
    per byte (low nibble = even K, high nibble = odd K), doubling the last dim.
    uint8 packed-fp6 bit streams (3 bytes per 4 codes along the last dim) are
    ambiguous from the dtype alone - pass the BlockScaledFormat.
    """
    if x.dtype == torch.float4_e2m1fn_x2:
        u8 = x.view(torch.uint8)
        lo = _fp4_unpacked_to_value(u8 & 0x0F)
        hi = _fp4_unpacked_to_value((u8 >> 4) & 0x0F)
        return torch.stack([lo, hi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)
    if x.dtype == torch.uint8:
        bits = None if fmt is None else _FLOATX_CODE_BITS.get(fmt.cutlass_dtype_name)
        if bits is None:
            raise ValueError(
                "uint8 operand bytes are ambiguous; pass a supported byte-container "
                "or packed-fp6 BlockScaledFormat"
            )
        if getattr(fmt, "storage_layout", None) == "packed_lsb_v1":
            if fmt.elem_bits != 6:
                raise ValueError(f"packed_lsb_v1 dequantization is unsupported for {fmt.name}")
            x = unpack_uint6(x)
        return floatx_unpacked_to_value(x, *bits)
    return x.float()


def check_blocked_scale_atom(SF: torch.Tensor, name: str = "scale") -> None:
    """Validate the hardware-fixed 128x4 blocked scale layout: rank 5/6 with a
    trailing (32, 4, 4) atom contiguous as one 512 B block (strides (16, 4, 1)).
    Single source of this contract (operand container, GEMM interface, tests)."""
    if SF.ndim not in (5, 6) or tuple(SF.shape[-3:]) != (32, 4, 4):
        raise ValueError(
            f"{name}: expected (rm, rk, 32, 4, 4) or (L, rm, rk, 32, 4, 4) blocked "
            f"scale factors, got shape {tuple(SF.shape)}"
        )
    if tuple(SF.stride()[-3:]) != (16, 4, 1):
        raise ValueError(
            f"{name}: inner (32, 4, 4) atom must be contiguous with strides (16, 4, 1), "
            f"got {SF.stride()[-3:]}"
        )


def pack_scale_2d_to_blocked_contig(scale_2d: torch.Tensor) -> torch.Tensor:
    """Rearrange a (l, mn, sf_k) or (mn, sf_k) e8m0 scale tensor into the
    contiguous (l, rm, rk, 32, 4, 4) blocked layout shared by the quack kernel
    and cuBLAS's block-scaling. Each inner (32, 4, 4) atom (512 B) holds one
    128 MN x 4 K swizzled tile. Pads `mn` to a multiple of 128 and `sf_k` to a
    multiple of 4 with zeros."""
    if scale_2d.dim() == 2:
        scale_2d = scale_2d.unsqueeze(0)
    assert scale_2d.dim() == 3, f"expected (l, mn, sf_k), got shape {tuple(scale_2d.shape)}"
    orig_dtype = scale_2d.dtype
    l, mn, sf_k = scale_2d.shape
    rm = _ceil_div(mn, 128)
    rk = _ceil_div(sf_k, 4)
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    u8 = scale_2d.contiguous().view(torch.uint8)
    if mn_pad != mn or sf_k_pad != sf_k:
        padded = torch.zeros(l, mn_pad, sf_k_pad, device=scale_2d.device, dtype=torch.uint8)
        padded[:, :mn, :sf_k] = u8
    else:
        padded = u8
    # (l, mn_pad, sf_k_pad) -> (l, rm, 128, rk, 4) -> (l, rm, rk, 128, 4)
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    # split 128 into (4 outer, 32 inner), then swap to (32, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    return blocks.view(orig_dtype)


def unpack_scale_blocked_to_2d(blocked: torch.Tensor, mn: int, sf_k: int) -> torch.Tensor:
    """Unswizzle (l, rm, rk, 32, 4, 4) blocked scale factors to (l, mn, sf_k)."""
    l, rm, rk = blocked.shape[:3]
    assert tuple(blocked.shape[3:]) == (32, 4, 4)
    orig_dtype = blocked.dtype
    u8 = blocked.view(torch.uint8)
    # (32=m%32, 4=m//32, 4=k%4) -> (4, 32, 4) -> (l, rm, rk, 128, 4) -> (l, mn_pad, sf_k_pad)
    u8 = u8.transpose(3, 4).reshape(l, rm, rk, 128, 4)
    u8 = u8.permute(0, 1, 3, 2, 4).reshape(l, rm * 128, rk * 4)
    return u8[:, :mn, :sf_k].contiguous().view(orig_dtype)
