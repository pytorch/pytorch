# Copyright (c) 2025, Tri Dao
from dataclasses import replace
from typing import NamedTuple, Optional, Tuple, Literal
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from torch._vendor.quack.blockscaled.operand import (
    BlockScaledFormat,
    BlockScaledOperand,
    mma_kind_for_pair,
)

# SplitKMode is re-exported here as its canonical public import path; it is *defined*
# in the gemm_config leaf because the kernel layer needs it at import time and this
# module sits above quack.gemm in the import graph (importing it from here would cycle).
from torch._vendor.quack.gemm_config import (
    GemmConfig,
    SplitKMode,
    blockscaled_config_ok,
    blockscaled_default_config,
    config_supports,
    cta_tile_shape_m,
    default_config,
    get_all_configs,
)

from torch._vendor.quack.autotuner import autotune, AutotuneConfig
from torch._vendor.quack.cute_dsl_utils import get_device_capacity
from torch._vendor.quack.gemm import gemm as gemm_dispatch, run_gemm_plan
from torch._vendor.quack.gemm_iface import (
    IfacePlan,
    VariantSpec,
    alloc_outputs,
    make_iface_plan,
    run_variant,
)
from torch._vendor.quack.gemm_tvm_ffi_utils import tensor_key, scalar_mode
from torch._vendor.quack.gemm_symmetric import gemm_symmetric as gemm_symmetric_dispatch, run_gemm_symmetric_plan
from torch._vendor.quack.rms_final_reduce import rms_final_reduce
from torch._vendor.quack.rounding import RoundingMode


def _empty_k_matmul_into(
    out: Tensor,
    *,
    bias: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    beta: float | Tensor = 1.0,
) -> None:
    """K=0 fast path: write `beta * C + bias` (or zero if neither) into `out`.

    Used by every gemm-flavored wrapper to skip a kernel launch when the
    contraction dim is empty. The matmul A @ B contributes zero, so the only
    remaining terms are the C term and the (broadcast) bias.
    """
    if C is not None:
        if isinstance(beta, float) and beta == 1.0:
            out.copy_(C)
        else:
            torch.mul(C, beta, out=out)
    else:
        out.zero_()
    if bias is not None:
        out += bias


def _silu_tanh(x: Tensor) -> Tensor:
    x_half = 0.5 * x
    return x_half * torch.tanh(x_half) + x_half


def _swiglu_oai_tanh(gate: Tensor, up: Tensor, alpha: float = 1.702) -> Tensor:
    gate_half = 0.5 * gate
    return (gate_half * torch.tanh(alpha * gate_half) + gate_half) * (up + 1)


# Dictionary mapping activation names to PyTorch functions
act_to_pytorch_fn_map = {
    None: lambda x: x,
    "silu": F.silu,
    "silu-tanh": _silu_tanh,
    "relu": F.relu,
    "relu_sq": lambda x: F.relu(x).square(),
    "gelu_tanh_approx": partial(F.gelu, approximate="tanh"),
    "tanh": torch.tanh,
}


# Dictionary mapping gated activation names to their forward functions
# Each function takes (gate, up) and returns postact
gated_to_pytorch_fn_map = {
    "swiglu": lambda gate, up: F.silu(gate) * up,
    "swiglu-tanh": lambda gate, up: _silu_tanh(gate) * up,
    "swiglu_oai": lambda gate, up: gate * torch.sigmoid(1.702 * gate) * (up + 1),
    "swiglu_oai-tanh": _swiglu_oai_tanh,
    "reglu": lambda gate, up: F.relu(gate) * up,
    "geglu": lambda gate, up: F.gelu(gate, approximate="tanh") * up,
    "glu": lambda gate, up: torch.sigmoid(gate) * up,
}


ActActivation = Literal[None, "silu", "silu-tanh", "relu", "relu_sq", "gelu_tanh_approx", "tanh"]
GatedActivation = Literal[
    "swiglu",
    "swiglu-tanh",
    "swiglu_oai",
    "swiglu_oai-tanh",
    "reglu",
    "geglu",
    "glu",
]
Activation = Literal[
    None,
    "silu",
    "silu-tanh",
    "relu",
    "relu_sq",
    "gelu_tanh_approx",
    "tanh",
    "swiglu",
    "swiglu-tanh",
    "swiglu_oai",
    "swiglu_oai-tanh",
    "reglu",
    "geglu",
    "glu",
]


def _check_split_k_unsupported(name: str, split_k: int) -> None:
    """For use in interface wrappers that do not support splitK yet."""
    if split_k not in (None, 1):
        raise NotImplementedError(
            f"{name} does not support split_k > 1; split_k is currently supported by "
            "gemm, gemm_add, and gemm_add_inplace."
        )


def _concat_interleave(t):
    """Interleave halves along non-contiguous dim: [first; second] → [f0, s0, f1, ...]"""
    dim = -2 if t.stride(-1) == 1 else -1
    return t.unflatten(dim, (2, t.shape[dim] // 2)).transpose(dim - 1, dim).flatten(dim - 1, dim)


def _concat_interleave_bias(t):
    """Interleave [gate; up] along last dim for bias vectors."""
    half = t.shape[-1] // 2
    return t.unflatten(-1, (2, half)).transpose(-2, -1).flatten(-2, -1)


# -- Blockscaled (MXFP8 / MXFP4 / NVFP4) operand handling --------------------
#
# Blockscaled A / B operands are :class:`quack.blockscaled.operand.BlockScaledOperand`
# instances carrying (qdata, scale, format, per_tensor_scale) - see
# AI/blockscaled_api.md. The container is the ONLY blockscaled operand form;
# (data, scale_factor) tuples are rejected with a TypeError at the unwrap site.
# Scale factors use the canonical cuBLAS/CUTLASS 128x4
# blocked layout: shape ``(rm, rk, 32, 4, 4)`` (optionally with a leading batch L),
# where the ``(32, 4, 4)`` inner block is ``(m % 32, (m // 32) % 4, k_block % 4)``
# with strides ``(16, 4, 1)`` — one contiguous 512-byte atom per 128 rows x 4
# K-blocks, matching torchao's ``to_blocked`` and ``torch._scaled_mm``.
# All format properties (scale vec size, element packing, scale dtype) come from
# the BlockScaledFormat descriptor; nothing below this layer may re-derive them
# from tensor dtypes. Packed operands carry the storage K extent in their shapes
# (fp4: ``torch.float4_e2m1fn_x2``, two elements per byte -> K/2; fp6: packed
# 6-bit uint8 bit stream -> 3*K/4 bytes); K here always refers to logical K.


def _launch(op):
    """Pick the raw impl for eager calls: the pure-Python custom-op boundary
    (autograd wrapper + per-call mutates_args aliasing checks) costs ~85us per
    call and only pays for itself under torch.compile/fake tensors, where the
    op schema is required."""
    if torch.compiler.is_compiling():
        return op
    return getattr(op, "_init_fn", op)


class _Operand(NamedTuple):
    data: Tensor
    sf: Optional[Tensor] = None
    fmt: Optional[BlockScaledFormat] = None
    per_tensor_scale: Optional[Tensor] = None  # NVFP4
    quant_dim: Optional[int] = None  # container operands: which dim the scales run along


def _unpack_operand(X) -> _Operand:
    """Split an ``A`` / ``B`` argument into (data, scale, format, per_tensor_scale)."""
    if isinstance(X, BlockScaledOperand):
        return _Operand(X.qdata, X.scale, X.format, X.per_tensor_scale, quant_dim=X.quant_dim)
    if isinstance(X, (tuple, list)):
        raise TypeError(
            "blockscaled operands must be BlockScaledOperand containers; "
            "(data, scale_factor) tuples are no longer accepted - wrap the parts with "
            "torch._vendor.quack.BlockScaledOperand.from_parts(data, scale_factor, format)"
        )
    return _Operand(X)


def _prep_blockscaled(
    opA: _Operand, opB: _Operand
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[str], Optional[str]]:
    """Validate an (A, B) operand pair; return encoded (SFA, SFB) and the format
    NAMES (what the op schemas carry).

    A and B carry independent formats. Pair legality (which combinations the
    hardware can express at all) is decided here via mma_kind_for_pair; whether a
    legal pair is IMPLEMENTED is enforced per-architecture by the gemm_smXXX
    kernel classes (each SM version supports different mx dtype combinations)."""
    assert (opA.sf is None) == (opB.sf is None), (
        "A and B must both (or neither) carry scale factors"
    )
    if opA.sf is None:
        return None, None, None, None
    mma_kind_for_pair(opA.fmt, opB.fmt)  # ValueError on unrepresentable pairs
    # The quantized axis must be each operand's contraction axis: A is (M, K) -
    # last dim; B is (K, N) - dim -2. Containers carry quant_dim, so wrong-axis
    # operands fail here even when the shapes happen to line up (square K == N),
    # which storage checks alone cannot catch.
    if opA.quant_dim != -1:
        raise ValueError(
            "A must be quantized along its last dim (the contraction dim K of (M, K)); "
            "got scales along dim -2 - remove a stray transpose or quantize with dim=-1"
        )
    if opB.quant_dim != -2:
        raise ValueError(
            "B must be quantized along dim -2 (the contraction dim K of (K, N)); pass "
            "W.mT of an (N, K) weight, or quantize the (K, N) data with dim=-2"
        )
    # Container scales were validated at construction. Rank is preserved
    # (5-D unbatched or 6-D batched); callers batch-canonicalize as needed.
    return _sf_encode(opA.sf), _sf_encode(opB.sf), opA.fmt.name, opB.fmt.name


def _fold_per_tensor_scales(alpha, opA: _Operand, opB: _Operand):
    """Fold NVFP4 per-tensor scales into alpha, out-of-place and without a host
    sync: a present scale turns alpha into a device scalar (the tensor-alpha path)."""
    scale = opA.per_tensor_scale
    if opB.per_tensor_scale is not None:
        scale = opB.per_tensor_scale if scale is None else scale * opB.per_tensor_scale
    if scale is None:
        return alpha
    scale = scale.reshape(1)
    return scale if (isinstance(alpha, float) and alpha == 1.0) else alpha * scale


def _reserve_blockscaled_out(out_dtype) -> None:
    """``out_dtype`` may name a BlockScaledFormat to request a blockscaled output
    (the call then returns a BlockScaledOperand). ``gemm`` implements it via the
    SF-generation epilogue (quack.epilogue.quantize_out); the other entry points
    still reject it here."""
    if _resolve_blockscaled_out(out_dtype) is not None:
        raise NotImplementedError(
            f"blockscaled output (out_dtype={out_dtype}) is only supported by quack.gemm "
            f"for now - see AI/blockscaled_api.md section 7"
        )


def _resolve_blockscaled_out(out_dtype) -> Optional[BlockScaledFormat]:
    """``out_dtype`` naming a BlockScaledFormat requests a quantized (blockscaled)
    output; returns the descriptor, or None for plain-dtype outputs."""
    if isinstance(out_dtype, BlockScaledFormat):
        return out_dtype
    if isinstance(out_dtype, str):
        return BlockScaledFormat.from_name(out_dtype)
    return None


def _alloc_blockscaled_out(out_shape, fmt: BlockScaledFormat, device, num_varlen_batches=None):
    """Allocate (out, out_sf) for a quantized-output GEMM (SF vectors along N).

    out_shape is the logical (..., M, N) shape; fp4 values are stored packed as
    ``float4_e2m1fn_x2`` with N/2 bytes per row. out_sf is the blocked
    (..., rm, rk, 32, 4, 4) scale tensor consumed by the kernel (and by the next
    GEMM's BlockScaledOperand input).

    For varlen_m pass ``num_varlen_batches``: M is total_m and out_sf becomes
    one M-padded ``(1, ceil(total_m/128) + L - 1, rk, 32, 4, 4)`` buffer with
    tile-aligned per-batch padding - the same layout varlen_m input SFA uses,
    so it feeds the next varlen_m blockscaled GEMM directly.
    """
    vec = fmt.sf_vec_size
    *batch, m, n = out_shape
    if fmt.qdata_dtype == torch.float4_e2m1fn_x2:
        assert n % 32 == 0, f"fp4 output requires N % 32 == 0 (16 B rows), got N={n}"
        out = torch.empty(*batch, m, n // 2, dtype=fmt.qdata_dtype, device=device)
    else:
        assert n % 16 == 0, f"fp8 output requires N % 16 == 0 (16 B rows), got N={n}"
        out = torch.empty(*batch, m, n, dtype=fmt.qdata_dtype, device=device)
    rm, rk = -(-m // 128), -(-n // (4 * vec))
    if num_varlen_batches is not None:
        assert not batch
        rm += num_varlen_batches - 1
        out_sf = torch.empty(1, rm, rk, 32, 4, 4, dtype=fmt.scale_dtype, device=device)
    else:
        out_sf = torch.empty(*batch, rm, rk, 32, 4, 4, dtype=fmt.scale_dtype, device=device)
    return out, out_sf


def _alloc_blockscaled_out_col(out_shape, fmt: BlockScaledFormat, device):
    """Column-direction variant of :func:`_alloc_blockscaled_out`: SF vectors
    along M, out_sf blocked over (N, M) as (..., rn, rm_k, 32, 4, 4) with rm_k
    padded to whole 128-row stripes (128 // (4 * vec) atom groups)."""
    vec = fmt.sf_vec_size
    *batch, m, n = out_shape
    assert fmt.qdata_dtype != torch.float4_e2m1fn_x2, (
        "column-direction quantized output supports fp8 values only (fp4 packs along N, "
        "which no consumer contracting over M can use)"
    )
    assert n % 16 == 0, f"fp8 output requires N % 16 == 0 (16 B rows), got N={n}"
    out = torch.empty(*batch, m, n, dtype=fmt.qdata_dtype, device=device)
    stripe = 128 // (4 * vec)
    rn = -(-n // 128)
    rm_k = -(-(-(-m // (4 * vec))) // stripe) * stripe
    out_sf = torch.empty(*batch, rn, rm_k, 32, 4, 4, dtype=fmt.scale_dtype, device=device)
    return out, out_sf


def _sf_batch_canonicalize(SFA: Tensor, SFB: Tensor, batched: bool) -> Tuple[Tensor, Tensor]:
    """Unbatched 5-D SFs only pair with 2D dense operands (the kernel prepends
    the trivial batch mode at trace time); batched/varlen calls get the batch
    dim added here."""
    if batched:
        SFA = SFA.unsqueeze(0) if SFA.ndim == 5 else SFA
        SFB = SFB.unsqueeze(0) if SFB.ndim == 5 else SFB
    return SFA, SFB


def _sf_encode(SF: Tensor) -> Tensor:
    """View e8m0 scale factors as uint8 for the custom-op boundary.

    Upstream PyTorch bug: a ``float8_e8m0fnu`` input to any mutable custom op
    makes Inductor's ``decompose_auto_functionalized`` pass fail with
    "auto_functionalized_v2 was not removed" (e4m3 is unaffected), breaking
    torch.compile. The uint8 view is zero-copy and unambiguous (uint8 == e8m0,
    e4m3 stays itself); :func:`_sf_decode` restores the dtype inside the op.

    Eager calls skip the view: they bypass the custom-op boundary entirely
    (see :func:`_launch`), and :func:`_sf_decode` no-ops on non-uint8 input.
    """
    if SF.dtype == torch.float8_e8m0fnu and torch.compiler.is_compiling():
        return SF.view(torch.uint8)
    return SF


def _sf_decode(SF: Optional[Tensor], bs_format: Optional[str]) -> Optional[Tensor]:
    """Inverse of :func:`_sf_encode`, applied inside op bodies. Format-driven:
    the scale dtype comes from the descriptor, never sniffed from uint8 (an
    NVFP4 e4m3 scale seen as uint8 must not silently decode as vec-32 e8m0)."""
    if SF is not None and SF.dtype == torch.uint8:
        SF = SF.view(BlockScaledFormat.from_name(bs_format).scale_dtype)
    return SF


def nvmmh_config(A, B, device_capacity):
    """Use nvMatmulHeuristics to pick a config for pure GEMM (no varlen/gather/epilogue).

    Returns None if unavailable, caller should fall back to default_config.
    """
    try:
        from torch._vendor.quack.nvmmh_heuristic import nvmmh_default_config

        return nvmmh_default_config(A, B, device_capacity)
    except Exception:
        return None


def _expand_split_k_configs(configs, A, B, device_capacity, blockscaled=False):
    """Add split_k > 1 variants of each surviving config for occupancy-starved shapes.

    Only called when the user passed split_k=None ("let the autotuner choose the
    factor"). Candidates are powers of two that lift the CTA count toward saturation
    without over-decomposing K; the autotuner then picks by measurement. The split
    MODE is never expanded (serial/parallel/staged differ in determinism semantics).
    """
    if A.ndim == 3:
        L, M, K = A.shape
    else:
        (M, K), L = A.shape, 1
    N = B.shape[-1]
    sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
    expanded = list(configs)
    for conf in configs:
        c = conf.kwargs["config"]
        cta_tile_m = cta_tile_shape_m(c.tile_m, c.cluster_m, device_capacity, blockscaled)
        tile_m, tile_n = (cta_tile_m, c.tile_n) if not c.swap_ab else (c.tile_n, cta_tile_m)
        ntiles = -(-M // tile_m) * -(-N // tile_n) * L
        k_tiles = -(-K // (c.tile_k or 64))
        for s in (2, 4, 8, 16):
            starved = ntiles < sm_count and ntiles * s <= 4 * sm_count
            if starved and 2 * s <= k_tiles:
                expanded.append(AutotuneConfig(config=replace(c, split_k=s)))
    return expanded


def prune_invalid_gemm_configs(configs, named_args: dict, **kwargs):
    kwargs = named_args | kwargs
    device_capacity = get_device_capacity(kwargs["A"].device)[0]
    configs = [conf for conf in configs if conf.kwargs["config"].device_capacity == device_capacity]
    gather_A = kwargs.get("A_idx", None) is not None
    varlen_m = kwargs.get("cu_seqlens_m", None) is not None
    varlen_k = kwargs.get("cu_seqlens_k", None) is not None
    configs = [
        conf
        for conf in configs
        if config_supports(conf.kwargs["config"], gather_A=gather_A, varlen_m=varlen_m)
    ]
    # use_tma_gather only valid when gather_A is active on SM100/SM110
    if not gather_A or device_capacity not in [10, 11]:
        configs = [conf for conf in configs if not conf.kwargs["config"].use_tma_gather]
    if kwargs.get("SFA", None) is not None:  # blockscaled (SM100 tcgen05 MMA constraints)
        configs = [conf for conf in configs if blockscaled_config_ok(conf.kwargs["config"])]
    if (
        kwargs.get("SFD", None) is not None or kwargs.get("SFDCol", None) is not None
    ):  # quantized output (SM100 SFD epilogue)
        col_only = kwargs.get("SFDCol", None) is not None

        def _sfd_ok(c: GemmConfig) -> bool:
            if c.swap_ab:  # SFD assumes N-major D and un-swapped N
                return False
            if c.device_capacity in (10, 11):
                # tile_n % 64 keeps the epi tile N at 32/64, covering whole SF
                # vectors for both vec sizes (32 for mx, 16 for nvfp4). The CTA
                # tile M must be 128 (the (4,1) epilogue warp shape, one full epi
                # row per thread): tile_m 128 with an even cluster_m selects the
                # 2-CTA MMA whose 64-row CTA tile uses the (2,2) warp shape,
                # splitting SF vectors across threads (and producing strided
                # fixup epi tiles the SF atom layout cannot divide).
                return (
                    c.tile_n % 64 == 0
                    and c.tile_m in (128, 256)
                    and not (c.tile_m == 128 and c.cluster_m % 2 == 0)
                )
            if c.device_capacity == 12:
                if col_only:
                    # Col vectors run along M: the 64-row epi tile covers
                    # whole 32-row vectors; the tile must divide into epi
                    # subtiles for the per-subtile SF flush.
                    return c.tile_m % 64 == 0
                # SM90-style register epilogue: epi tile N = gcd(32|64, tile_n)
                # must cover whole SF vectors (tile_n % 32 covers both vec
                # sizes), and the epi tile M must divide the 128-row SF pad
                # (excludes the 192-row epi tile of 192-multiple tile_m).
                return c.tile_n % 32 == 0 and c.tile_m % 64 == 0 and c.tile_m % 192 != 0
            return False

        configs = [conf for conf in configs if _sfd_ok(conf.kwargs["config"])]
    # Autotuned split-K: only the plain-gemm family exposes the knob; split_k=None means
    # "tune the factor" (an explicit int is forced in gemm_tuned and needs no variants).
    if (
        "split_k" in kwargs
        and kwargs["split_k"] is None
        and not (varlen_m or varlen_k or gather_A)
        and device_capacity in (9, 10, 11, 12)
    ):
        configs = _expand_split_k_configs(
            configs, kwargs["A"], kwargs["B"], device_capacity, kwargs.get("SFA") is not None
        )
    return configs


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["dynamic_scheduler", "split_k", "split_k_mode", "bs_format_a", "bs_format_b"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_tuned(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float | Tensor = 1.0,  # (1,)
    beta: float | Tensor = 1.0,  # (1,)
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    cu_seqlens_k: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
    SFA: Optional[Tensor] = None,  # (L, rm, rk, 32, 4, 4) blocked scale factors
    SFB: Optional[Tensor] = None,  # (L, rn, rk, 32, 4, 4)
    # BlockScaledFormat names for A / B (independent; required when SFA/SFB are passed)
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: Optional[int] = 1,  # None: let the autotuner choose the factor (config.split_k)
    split_k_mode: int = SplitKMode.SERIAL,
    # Quantized output (SM100 SFD epilogue): blocked output scale factors,
    # written by the kernel. SFD = row direction (SF vectors along N), SFDCol =
    # column direction (vectors along M); mutually exclusive. sfd_norm_const is
    # the optional fp32 norm constant folded into the stored scales (the
    # reciprocal of the nvfp4 per-tensor scale).
    SFD: Optional[Tensor] = None,
    sfd_norm_const: Optional[float | Tensor] = None,
    SFDCol: Optional[Tensor] = None,
) -> Tuple[GemmConfig, int, bool, object]:  # (config, split_k, dynamic_scheduler, dispatch plan)
    blockscaled = SFA is not None
    if blockscaled:
        SFA, SFB = _sf_decode(SFA, bs_format_a), _sf_decode(SFB, bs_format_b)
    quant_out = SFD is not None or SFDCol is not None
    if config is None:
        if blockscaled:
            m = A.shape[-2]
            config = blockscaled_default_config(
                m, B.shape[-1], device_capacity=get_device_capacity(A.device)[0]
            )
        else:
            # Use nvMMH heuristic for pure GEMM (no varlen, no gather, no epilogue).
            # Quantized output bypasses it: nvMMH may pick swap_ab / tile_n / 2-CTA
            # shapes the SFD epilogue cannot run (see _sfd_ok in the config prune).
            is_pure_gemm = (
                cu_seqlens_m is None
                and cu_seqlens_k is None
                and A_idx is None
                and C is None
                and bias is None
                and not add_to_output
                and not quant_out
            )
            if is_pure_gemm:
                device_capacity = get_device_capacity(A.device)[0]
                config = nvmmh_config(A, B, device_capacity)
            if config is None:
                config = default_config(A.device)
    if split_k is None:
        # Autotuned split-K: the factor comes from the (possibly split_k-expanded) config.
        split_k = config.split_k
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    if blockscaled:
        assert not gather_A, "Blockscaled GEMM does not support gather_A yet"
        assert not concat_layout, "Blockscaled GEMM does not support concat_layout"
        assert not config.swap_ab, "Blockscaled GEMM does not support swap_ab yet"
    if gather_A:
        assert varlen, "gather_A requires either varlen_m or varlen_k"
        assert config.cluster_n == 1, "gather_A requires cluster_n=1"
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if quant_out:
        # Explicit-config calls skip the autotune prune; re-assert the SFD
        # constraint set (see _sfd_ok in prune_invalid_gemm_configs).
        assert not config.swap_ab, "Quantized output (SFD) requires an un-swapped config"
        if config.device_capacity == 12:
            if SFDCol is not None:
                assert config.tile_m % 64 == 0, (
                    "Column-direction quantized output (SFDCol) requires tile_m % 64 == 0"
                )
            else:
                assert config.tile_n % 32 == 0, "Quantized output (SFD) requires tile_n % 32 == 0"
                assert config.tile_m % 64 == 0 and config.tile_m % 192 != 0, (
                    "Quantized output (SFD) on SM120 requires a 64/128/256-row tile "
                    "(the 192-row epi tile does not divide the 128-row SF pad)"
                )
        else:
            assert config.tile_n % 64 == 0, "Quantized output (SFD) requires tile_n % 64 == 0"
            assert config.tile_m in (128, 256) and not (
                config.tile_m == 128 and config.cluster_m % 2 == 0
            ), (
                "Quantized output (SFD) requires the (4,1) epilogue warp shape "
                "(the 64-row 2-CTA tile splits SF vectors across threads)"
            )
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    dispatch_plan = _gemm_execute(
        A,
        B,
        out,
        C,
        bias=bias,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        dynamic_scheduler=dynamic_scheduler,
        config=config,
        rounding_mode=rounding_mode,
        sr_seed=sr_seed,
        concat_layout=concat_layout,
        SFA=SFA,
        SFB=SFB,
        split_k=split_k,
        split_k_mode=split_k_mode,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        SFD=SFD,
        sfd_norm_const=sfd_norm_const,
        SFDCol=SFDCol,
    )
    # Resolved decisions, so the eager `gemm` wrapper can record an interface
    # plan (see _GemmIfacePlan). The custom-op route discards this.
    return config, split_k, dynamic_scheduler, dispatch_plan


def _gemm_execute(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor],
    *,
    bias: Optional[Tensor],
    alpha: float | Tensor,
    beta: float | Tensor,
    cu_seqlens_m: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    A_idx: Optional[Tensor],
    batch_idx_permute: Optional[Tensor],
    add_to_output: bool,
    dynamic_scheduler: bool,
    config: GemmConfig,
    rounding_mode: int,
    sr_seed: int | Tensor,
    concat_layout: tuple | None,
    SFA: Optional[Tensor],
    SFB: Optional[Tensor],
    split_k: int,
    split_k_mode: int,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    SFD: Optional[Tensor] = None,
    sfd_norm_const: Optional[float | Tensor] = None,
    SFDCol: Optional[Tensor] = None,
    dispatch_plan=None,
):
    """Transform operands (batch dims, swap_ab, concat) and launch the GEMM.

    All metadata-derived decisions are already resolved here: ``config``,
    ``split_k`` and ``dynamic_scheduler`` are final values and SFs are decoded.
    The per-call work is only the argument routing the dispatch layer needs, so
    a cached interface plan (see _GemmIfacePlan) can call this directly —
    passing its captured ``dispatch_plan`` to also skip the dispatch layer's
    key. Returns the dispatch plan for the interface plan to capture.
    """
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    capacity = get_device_capacity(A.device)[0]
    sm90_plus = capacity >= 9
    # Trace-time relabels replace per-call torch views (~1.5us of dispatcher
    # overhead each): b_kn passes B in its native (K, N) orientation and
    # dense_2d keeps unbatched operands rank-2 — the kernel transposes /
    # appends the trivial batch mode when it's compiled.
    b_kn = sm90_plus and not varlen and not config.swap_ab and not concat_layout
    if not b_kn:
        B = B.mT  # (N, K) or (L, N, K) or (N, total_K)
    dense_2d = (
        A.ndim == 2
        and B.ndim == 2
        and out.ndim == 2
        and (C is None or C.ndim == 2)
        and not varlen
        and split_k == 1
        and not concat_layout
        and batch_idx_permute is None
        and sm90_plus
    )
    if not dense_2d:
        if A.ndim == 2 and not varlen:
            A = A.unsqueeze(0)  # (1, M, K)
        if B.ndim == 2 and not varlen_k:
            B = B.unsqueeze(0)  # (1, N, K), or (1, K, N) if b_kn
        if C is not None and C.ndim == 2 and not varlen_m:
            C = C.unsqueeze(0)  # (1, M, N)
        if out.ndim == 2 and not varlen_m:
            out = out.unsqueeze(0)
        # Unbatched 5-D SFs must follow their rank-2 operands into batched form.
        if SFA is not None and SFA.ndim == 5 and not varlen:
            SFA = SFA.unsqueeze(0)
        if SFB is not None and SFB.ndim == 5 and not varlen:
            SFB = SFB.unsqueeze(0)
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    if varlen_m:
        # If gather_A (A_idx provided), use its length; otherwise use A.shape[0]
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-2])
    else:
        n_ext = B.shape[-1] if b_kn else B.shape[-2]
        if dense_2d:
            out_shape = (A.shape[0], n_ext)
        else:
            batch_size = B.shape[0] if not varlen_k else cu_seqlens_k.shape[0] - 1
            out_shape = (batch_size, A.shape[-2], n_ext)
    # fp4 output is stored packed: the last dim holds N/2 float4_e2m1fn_x2 bytes.
    if out.dtype == torch.float4_e2m1fn_x2:
        out_shape = (*out_shape[:-1], out_shape[-1] // 2)
    assert out.shape == out_shape, f"out shape mismatch: {out.shape} vs {out_shape}"
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and capacity == 9
        else None
    )
    # Handle bias concat layout: transform "bias" key to kernel-level key or permute data.
    if concat_layout and "bias" in concat_layout:
        if bias is not None and bias.dtype.itemsize >= 4:
            # fp32: kernel permutes via layout; replace "bias" with the kernel-level key
            concat_layout = tuple("mRowVecBroadcast" if k == "bias" else k for k in concat_layout)
        else:
            # No bias or sub-fp32: strip "bias" from concat_layout; permute data if needed
            concat_layout = tuple(k for k in concat_layout if k != "bias")
            if bias is not None:
                bias = _concat_interleave_bias(bias)
    # When swap_ab, A↔B (out/C stay, but .mT flips their strides so the kernel
    # auto-detects the correct non-contiguous dim).
    swap = config.swap_ab
    A_d = A if not swap else B
    B_d = B if not swap else A
    out_d = out if not swap else out.mT
    C_d = (C if not swap else C.mT) if C is not None else None
    rowvec_d = bias if not swap else None
    colvec_d = bias if swap else None
    if dispatch_plan is not None:
        # Warm replay: the interface plan key vouches for the metadata, so skip
        # the dispatch layer's own mode derivation, key, and lookup.
        run_gemm_plan(
            dispatch_plan,
            A_d,
            B_d,
            out_d,
            C_d,
            tile_count_semaphore=tile_count_semaphore,
            rowvec_bias=rowvec_d,
            colvec_bias=colvec_d,
            alpha=alpha,
            beta=beta,
            sr_seed=sr_seed,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            SFA=SFA,
            SFB=SFB,
            SFD=SFD,
            sfd_norm_const=sfd_norm_const,
            SFDCol=SFDCol,
        )
        return dispatch_plan
    _swap_map = {"A": "B", "B": "A", "out": "out", "C": "C", "mRowVecBroadcast": "mColVecBroadcast"}
    swapped_concat = (
        tuple(_swap_map.get(k, k) for k in concat_layout)
        if swap and concat_layout
        else concat_layout
    )
    return gemm_dispatch(
        A_d,
        B_d,
        out_d,
        C_d,
        tile_count_semaphore,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.cluster_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=rowvec_d,
        colvec_bias=colvec_d,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=sr_seed,
        use_tma_gather=config.use_tma_gather,
        concat_layout=swapped_concat,
        num_warps=config.num_warps,
        tile_K=config.tile_k,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        split_k=split_k,
        split_k_mode=split_k_mode,
        b_kn=b_kn,
        SFD=SFD,
        sfd_norm_const=sfd_norm_const,
        SFDCol=SFDCol,
    )


## ── gemm_act / gemm_gated ───────────────────────────────────────────────────
# Ported to the epilogue-object surface (see the gemm_rms note below):
# quack.epilogue.library.linear_act_mod owns canonicalization, plan caching, and
# tuning (incl. varlen/gather, blockscaled, concat_layout, and swap-at-trace
# for the element-mode forms; the gated config space never had swap_ab).
# SR (rounding_mode/sr_seed) is not exposed here; call
# linear_act_mod(..., sr=True).gemm(rounding_mode=...) directly.


def _gemm_act_call(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor],
    bias: Optional[Tensor],
    *,
    activation,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,  # decoded (real-dtype) scale factors
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,  # BlockScaledFormat names (see quack.gemm.gemm)
    bs_format_b: Optional[str] = None,
    concat_layout: tuple | None = None,
    dynamic_scheduler: bool,
    alpha: float | Tensor = 1.0,
    tuned: bool,
    config: Optional[GemmConfig] = None,
) -> None:
    from torch._vendor.quack.epilogue.library import linear_act_mod

    if concat_layout:
        # A 16-bit concat bias is materialized interleaved; a wide bias keeps
        # its broadcast port with the concat relabel applied there. Swap is
        # structurally excluded on the concat path (it vetoes b_kn).
        bias, concat_layout = _act_concat_bias(bias, concat_layout, False)
        concat_layout = tuple(sorted(concat_layout)) if concat_layout else None
    has_alpha = scalar_mode(alpha) != 0
    mod = linear_act_mod(
        activation,
        gated=activation in gated_to_pytorch_fn_map,
        has_c=C is not None,
        has_rowvec=bias is not None,
        has_colvec=False,
        sr=False,
        has_alpha=has_alpha,
    )
    outs = {"mAuxOut": postact_out}
    store_d = preact_out is not None
    if store_d:
        outs["D"] = preact_out
    operands = {}
    if bias is not None:
        operands["mRowVecBroadcast"] = bias
    if has_alpha:
        operands["alpha"] = alpha
    mod(
        A,
        B,
        C,
        out=outs,
        store_d=store_d,
        config=config,
        tuned=tuned,
        dynamic_scheduler=dynamic_scheduler,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=concat_layout,
        **operands,
    )


def _act_concat_bias(bias, concat_layout, swap_ab):
    """Resolve the bias leg of a concat [gate; up] weight layout.

    A wide (fp32) bias rides its broadcast port with the concat relabel
    applied there; a 16-bit bias is materialized interleaved instead (the
    epilogue reads it through the packed F2 contract)."""
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    if concat_layout and "bias" in concat_layout:
        if bias is not None and bias.dtype.itemsize >= 4:
            bias_key = "mColVecBroadcast" if swap_ab else "mRowVecBroadcast"
            concat_layout = tuple(bias_key if k == "bias" else k for k in concat_layout)
        else:
            concat_layout = tuple(k for k in concat_layout if k != "bias")
            if bias is not None:
                bias = _concat_interleave_bias(bias)
    return bias, concat_layout


## ── gemm_dact / gemm_dgated ─────────────────────────────────────────────────
# Ported to the epilogue-object surface (see the gemm_rms note below):
# quack.epilogue.library.dact_mod / dgated_mod own canonicalization, plan caching,
# and tuning (incl. varlen/gather and dynamic_scheduler=True, dact's default).
# The colvec reduce comes back finalized via the generic
# VecReduce.host_finalize — the same partials.sum(dim=-1) the old wrapper ran.


def _gemm_dact_call(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    dx_out: Tensor,
    postact_out: Tensor,
    *,
    activation,
    colvec_scale: Optional[Tensor] = None,
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,  # decoded (real-dtype) scale factors
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,  # BlockScaledFormat names (see quack.gemm.gemm)
    bs_format_b: Optional[str] = None,
    dynamic_scheduler: bool,
    tuned: bool,
    config: Optional[GemmConfig] = None,
) -> Optional[Tensor]:
    """Launch dact/dgated on the epilogue object. Returns the finalized colvec
    reduce (colvec_reduce=True) or None."""
    from torch._vendor.quack.epilogue.library import dact_mod, dgated_mod

    mod_fn = dgated_mod if activation in gated_to_pytorch_fn_map else dact_mod
    mod = mod_fn(activation, has_scale=colvec_scale is not None, has_reduce=colvec_reduce)
    operands = {}
    if colvec_scale is not None:
        operands["mColVecBroadcast"] = colvec_scale
    res = mod(
        A,
        B,
        PreAct,
        out={"D": dx_out, "mAuxOut": postact_out},
        config=config,
        tuned=tuned,
        dynamic_scheduler=dynamic_scheduler,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        **operands,
    )
    return res.get("mColVecReduce")


class _GemmIfacePlan(NamedTuple):
    """Interface-level launch plan: the metadata-derived decisions of a ``gemm``
    call (validation, config resolution — including the autotuned winner — and
    the output-allocation recipe).

    Cached per metadata key so a warm eager call jumps straight to
    :func:`_gemm_execute`, skipping the wrapper layers, the per-call asserts,
    and the heuristic/autotuner lookup. It also captures the dispatch layer's
    resolved plan (see ``_GemmPlan`` in gemm.py), so this key is the only one
    on the warm path — which is exactly why it must subsume everything the
    dispatch key covers (hence the alpha/sr modes in the key).
    """

    config: GemmConfig
    split_k: int
    dynamic_scheduler: bool
    out_shape: tuple
    out_dtype: torch.dtype
    dispatch_plan: object


_gemm_iface_plan_cache: dict[tuple, _GemmIfacePlan] = {}


def gemm(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    # For blockscaled (MXFP8/MXFP4/NVFP4/MXFP6): a BlockScaledOperand container
    # carrying the blocked (rm, rk, 32, 4, 4) / (L, rm, rk, 32, 4, 4) scale factors.
    A: Tensor | BlockScaledOperand,
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m; a BlockScaledOperand
    # container for a quantized output (matching the out_dtype format).
    out: Optional[Tensor | BlockScaledOperand] = None,
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float | Tensor = 1.0,
    # a BlockScaledFormat (or its name) requests a quantized (blockscaled) output:
    # the kernel writes fp8/fp4 values plus blocked scale factors (SF-generation
    # epilogue) and the call returns a BlockScaledOperand.
    out_dtype: Optional[torch.dtype | BlockScaledFormat | str] = None,
    # Quantized output only. out_quant_dim: which logical dim the SF vectors run
    # along - -1 (default: along N, the next GEMM's contraction dim) or -2 (along
    # M, for backward consumers contracting over this output's M; fp8 only,
    # values stay (M, N) n-major). out_transposed (with out_quant_dim=-2):
    # return D^T instead - a (N, M) / (L, N, M) row-quantized BlockScaledOperand
    # (quant_dim -1 in its own frame) whose values are contiguous along the
    # consumer's K = this GEMM's M, i.e. a directly loadable k-major operand
    # for the backward GEMM. Implemented as the swapped GEMM D^T = B^T A^T
    # riding the ordinary row-direction path, so it is fp4-capable (nvfp4/mxfp4
    # pack along M) and pays no cross-warp exchange on SM120.
    # out_per_tensor_scale: NVFP4 second-level scale - dequant = q * sf * scale;
    # its reciprocal is folded into the stored scale factors.
    out_quant_dim: int = -1,
    out_transposed: bool = False,
    out_per_tensor_scale: Optional[float | Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
    split_k: Optional[int] = 1,  # K-dim CTAs per tile; None = let the autotuner choose
    split_k_mode: int = SplitKMode.SERIAL,  # see SplitKMode: SERIAL/SEPARATE deterministic, PARALLEL fastest but arrival-order
) -> Tensor | BlockScaledOperand:
    """GEMM with optional output tensor and tuning control."""
    fmt_d = _resolve_blockscaled_out(out_dtype)
    if fmt_d is not None and out_transposed:
        # D^T quantized along its rows = col-direction quantization of D with
        # values contiguous along M: run the swapped GEMM (operand .mT views,
        # no data movement) through the ordinary row-direction path. The
        # kernel sees its native n-major D — this is the only orientation in
        # which "SFD under swap_ab" is coherent, and it needs no kernel-side
        # swap plumbing at all.
        assert out_quant_dim == -2, "out_transposed requires out_quant_dim=-2"
        assert bias is None, "out_transposed quantized output does not support bias yet"
        assert cu_seqlens_m is None and cu_seqlens_k is None and A_idx is None, (
            "out_transposed quantized output does not support varlen/gather"
        )
        assert batch_idx_permute is None and not concat_layout
        return gemm(
            B.mT,
            A.mT,
            out=out,
            alpha=alpha,
            out_dtype=out_dtype,
            out_quant_dim=-1,
            out_per_tensor_scale=out_per_tensor_scale,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            rounding_mode=rounding_mode,
            sr_seed=sr_seed,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
    assert not out_transposed, "out_transposed requires a blockscaled out_dtype"
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    alpha = _fold_per_tensor_scales(alpha, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(
            SFA, SFB, A.ndim == 3 or cu_seqlens_m is not None or cu_seqlens_k is not None
        )
    out_sf = None
    if fmt_d is not None:
        assert out_quant_dim in (-1, -2), f"out_quant_dim must be -1 or -2, got {out_quant_dim}"
        # rounding_mode composes: RS quantizes the rescaled values through
        # cvt.rs (hw fp8x4/e2m1x4 on sm_100a/103a, sw emulation elsewhere);
        # the SF bytes themselves stay RN/ceil either way.
        assert not concat_layout, "quantized output does not support concat_layout"
        assert split_k in (1, None), "quantized output does not support split_k yet"
        split_k = 1
        if out_quant_dim == -2:
            assert cu_seqlens_m is None and cu_seqlens_k is None, (
                "column-direction quantized output does not support varlen yet"
            )
        if isinstance(out, BlockScaledOperand):
            assert out.format == fmt_d, (
                f"out container format {out.format.name} != out_dtype {fmt_d.name}"
            )
            # A wrong-direction container has a differently-shaped SF buffer;
            # the kernel derives its store bounds from that shape, so this
            # would silently truncate/corrupt instead of erroring later.
            assert out.quant_dim == out_quant_dim, (
                f"out container quant_dim {out.quant_dim} != out_quant_dim {out_quant_dim}"
            )
            out_sf, out = out.scale, out.qdata
    else:
        assert out_per_tensor_scale is None, "out_per_tensor_scale requires a blockscaled out_dtype"
        assert not isinstance(out, BlockScaledOperand), (
            "a BlockScaledOperand out requires a blockscaled out_dtype"
        )
    # Eager plan fast path: replay of a previously validated call with different
    # data pointers. The key covers everything the slow path's decisions read,
    # including the alpha/sr modes: the captured dispatch plan depends on them,
    # so this key must subsume the dispatch key. Varlen/gather/permute/concat
    # calls always take the general path; compiled code and quantized outputs
    # take the custom-op route below.
    plan_key = None
    if not torch.compiler.is_compiling() and (
        cu_seqlens_m is None
        and cu_seqlens_k is None
        and A_idx is None
        and batch_idx_permute is None
        and not concat_layout
        and fmt_d is None
    ):
        plan_key = (
            tensor_key(A),
            tensor_key(B),
            tensor_key(out),
            tensor_key(bias),
            tensor_key(SFA),
            tensor_key(SFB),
            bs_format_a,
            bs_format_b,
            opA.quant_dim,
            opB.quant_dim,
            A.device,
            out_dtype,
            tuned,
            dynamic_scheduler,
            rounding_mode,
            split_k,
            split_k_mode,
            scalar_mode(alpha),
            isinstance(sr_seed, Tensor),
        )
        plan = _gemm_iface_plan_cache.get(plan_key)
        if plan is not None:
            if out is None:
                out = torch.empty(plan.out_shape, dtype=plan.out_dtype, device=A.device)
            # No empty-input checks: empty calls return before plan recording below,
            # so their metadata never produces a plan hit. SFs pass through in the
            # rank the caller gave (5-D unbatched or 6-D), same as the slow path.
            _gemm_execute(
                A,
                B,
                out,
                None,
                bias=bias,
                alpha=alpha,
                beta=1.0,
                cu_seqlens_m=None,
                cu_seqlens_k=None,
                A_idx=None,
                batch_idx_permute=None,
                add_to_output=False,
                dynamic_scheduler=plan.dynamic_scheduler,
                config=plan.config,
                rounding_mode=rounding_mode,
                sr_seed=sr_seed,
                concat_layout=None,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                split_k=plan.split_k,
                split_k_mode=split_k_mode,
                dispatch_plan=plan.dispatch_plan,
            )
            return out
    if out is None:
        if out_dtype is None:
            # Blockscaled inputs are fp8/fp4; default to bf16 output.
            out_dtype = torch.bfloat16 if SFA is not None else A.dtype
        varlen_m = cu_seqlens_m is not None
        varlen_k = cu_seqlens_k is not None
        if varlen_m:
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            out_shape = (total_m, B.shape[-1])
        elif varlen_k:
            L = cu_seqlens_k.shape[0] - 1
            # For varlen_k, the first dimension is always A.shape[0] (M dimension)
            out_shape = (L, A.shape[0], B.shape[-1])
        else:
            out_shape = (
                (A.shape[0], B.shape[-1]) if A.ndim == 2 else (A.shape[0], A.shape[-2], B.shape[-1])
            )
        if fmt_d is not None:
            if out_quant_dim == -1:
                num_varlen_batches = cu_seqlens_m.shape[0] - 1 if varlen_m else None
                out, out_sf = _alloc_blockscaled_out(
                    out_shape, fmt_d, A.device, num_varlen_batches=num_varlen_batches
                )
            else:
                out, out_sf = _alloc_blockscaled_out_col(out_shape, fmt_d, A.device)
        else:
            out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if fmt_d is not None:
        assert out_sf is not None, (
            "quantized output requires out as a BlockScaledOperand (or out=None to allocate)"
        )
        pts = None
        sfd_norm_const = None
        if out_per_tensor_scale is not None:
            assert fmt_d.has_per_tensor_scale, f"{fmt_d.name} does not take out_per_tensor_scale"
            if isinstance(out_per_tensor_scale, Tensor):
                pts = out_per_tensor_scale.float().reshape(1)
                sfd_norm_const = pts.reciprocal()
            else:
                pts = torch.full(
                    (1,), float(out_per_tensor_scale), dtype=torch.float32, device=out.device
                )
                sfd_norm_const = 1.0 / out_per_tensor_scale
        out_op = BlockScaledOperand(
            qdata=out, scale=out_sf, format=fmt_d, per_tensor_scale=pts, quant_dim=out_quant_dim
        )
    # Empty-input fast path: skip kernel launch.
    # M=0 / N=0 — the tile scheduler's ceil_div over a zero dim divides by zero.
    # K=0 — the kernel rejects stride-0 inputs (stride must be divisible by 8);
    #       semantically the empty contraction yields a zero matrix.
    if out.numel() == 0:
        return out_op if fmt_d is not None else out
    if A.numel() == 0:
        assert fmt_d is None, "quantized output does not support K == 0"
        _empty_k_matmul_into(out, bias=bias)
        return out
    if fmt_d is not None:
        # Quantized output always takes the custom-op route (compiling or
        # eager): the op marshals the SFD/norm-const args and calls the tuner.
        gemm_quant_out(
            A,
            B,
            out,
            _sf_encode(out_sf),
            bs_format_d=fmt_d.name,
            sfd_dim="n" if out_quant_dim == -1 else "m",
            bias=bias,
            alpha=alpha if isinstance(alpha, float) else 1.0,
            alpha_tensor=alpha if not isinstance(alpha, float) else None,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            sfd_norm_const=sfd_norm_const if isinstance(sfd_norm_const, float) else 1.0,
            sfd_norm_const_tensor=sfd_norm_const if isinstance(sfd_norm_const, Tensor) else None,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            rounding_mode=rounding_mode,
            sr_seed=sr_seed if isinstance(sr_seed, int) else 0,
            sr_seed_tensor=sr_seed if isinstance(sr_seed, Tensor) else None,
        )
        return out_op
    if torch.compiler.is_compiling():
        # The torch-library boundary needs schema-typed args: alpha/sr_seed are
        # split by type, concat_layout stringified, e8m0 SFs uint8-viewed
        # (_sf_encode above).
        alpha_tensor = alpha if not isinstance(alpha, float) else None
        alpha = alpha if isinstance(alpha, float) else 1.0
        sr_seed_tensor = sr_seed if isinstance(sr_seed, Tensor) else None
        sr_seed_int = sr_seed if isinstance(sr_seed, int) else 0
        concat_str = ",".join(concat_layout) if concat_layout else None
        gemm_out(
            A,
            B,
            out,
            bias=bias,
            alpha=alpha,
            alpha_tensor=alpha_tensor,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            rounding_mode=rounding_mode,
            sr_seed=sr_seed_int,
            sr_seed_tensor=sr_seed_tensor,
            concat_layout=concat_str,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
        return out
    # Eager: skip the custom-op marshalling and call the tuner directly with
    # natural argument types; it returns the resolved decisions for the plan.
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    config, split_k_resolved, dynamic_resolved, dispatch_plan = fn(
        A,
        B,
        out,
        C=None,
        bias=bias,
        alpha=alpha,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        rounding_mode=rounding_mode,
        sr_seed=sr_seed,
        concat_layout=concat_layout,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )
    if plan_key is not None:
        _gemm_iface_plan_cache[plan_key] = _GemmIfacePlan(
            config=config,
            split_k=split_k_resolved,
            dynamic_scheduler=dynamic_resolved,
            out_shape=tuple(out.shape),
            out_dtype=out.dtype,
            dispatch_plan=dispatch_plan,
        )
    return out


@torch.library.custom_op(
    "torch_vendor_quack::gemm_out",
    mutates_args=("out",),
    device_types="cuda",
    # We have to split out alpha and alpha_tensor since torch.library requires
    # each argument to have a fixed type
    # schema="(Tensor A, Tensor B, Tensor(a2!) out, Tensor? bias, float alpha=1.0, Tensor? alpha_tensor=None, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_out(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int = 0,
    sr_seed_tensor: Optional[Tensor] = None,
    concat_layout: Optional[str] = None,
    # Blockscaled scale factors, (L, rm/rn, rk, 32, 4, 4); operands are unpacked
    # to these flat args before the custom-op boundary since torch.library
    # schemas have no (Tensor, Tensor) argument type. ``bs_format_a/_b`` are the
    # (independent) BlockScaledFormat names - the descriptors are the single
    # source of format properties below this boundary.
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    """GEMM with pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    # Shared helpers: drift between this eager body and the register_fake side
    # is structurally impossible because both call the same functions.
    alpha = _merge_tensor(alpha, alpha_tensor)
    sr_seed_arg = _merge_tensor(sr_seed, sr_seed_tensor)
    fn(
        A,
        B,
        out,
        C=None,
        bias=bias,
        alpha=alpha,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        rounding_mode=rounding_mode,
        sr_seed=sr_seed_arg,
        concat_layout=_parse_concat_layout(concat_layout),
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


@torch.library.custom_op(
    "torch_vendor_quack::gemm_quant_out",
    mutates_args=("out", "SFD"),
    device_types="cuda",
)
def gemm_quant_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N[/2 for fp4]) or (L, M, N[/2]) or (total_M, N[/2]) quantized values
    # Output scale factors for quantized D (mxfp8/mxfp4/nvfp4), written by the
    # kernel; e8m0 crosses the boundary as a uint8 view (see _sf_encode),
    # decoded from bs_format_d. A separate op from gemm_out because
    # torch.library breaks on None values for tensors named in mutates_args -
    # here SFD is required.
    # varlen_m: one M-padded (1, total_padded_rm, rk, 32, 4, 4) buffer with
    # tile-aligned per-batch padding (input-SFA convention).
    SFD: Tensor,
    bs_format_d: str,
    sfd_dim: str = "n",
    bias: Optional[Tensor] = None,
    alpha: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    batch_idx_permute: Optional[Tensor] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    sfd_norm_const: float = 1.0,
    sfd_norm_const_tensor: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int = 0,
    sr_seed_tensor: Optional[Tensor] = None,
) -> None:
    """GEMM with quantized output: fp8/fp4 values in ``out``, block scale
    factors in ``SFD`` (vectors along N for sfd_dim="n", along M for "m")."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = _merge_tensor(alpha, alpha_tensor)
    sr_seed_arg = _merge_tensor(sr_seed, sr_seed_tensor)
    sfd_norm_const_arg = _merge_tensor(sfd_norm_const, sfd_norm_const_tensor)
    if isinstance(sfd_norm_const_arg, float) and sfd_norm_const_arg == 1.0:
        sfd_norm_const_arg = None
    SFD = _sf_decode(SFD, bs_format_d)
    if SFD.ndim == 5:
        SFD = SFD.unsqueeze(0)
    fn(
        A,
        B,
        out,
        C=None,
        bias=bias,
        alpha=alpha,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        rounding_mode=rounding_mode,
        sr_seed=sr_seed_arg,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        SFD=SFD if sfd_dim == "n" else None,
        sfd_norm_const=sfd_norm_const_arg,
        SFDCol=SFD if sfd_dim == "m" else None,
    )


def gemm_ref(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    out_dtype: Optional[torch.dtype] = None,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tensor:
    """Reference implementation for GEMM with pre-allocated output."""
    # The out_dtype argument requires torch >= 2.8
    out_dtype = A.dtype if out_dtype is None else out_dtype
    if concat_layout:
        if "A" in concat_layout:
            A = _concat_interleave(A)
        if "B" in concat_layout:
            B = _concat_interleave(B)
        if "bias" in concat_layout and bias is not None:
            bias = _concat_interleave_bias(bias)
    if cu_seqlens_m is None and cu_seqlens_k is None:
        fn = torch.bmm if A.ndim == 3 else torch.mm
        out = fn(A, B, out_dtype=out_dtype, out=out)
        if not isinstance(alpha, float) or alpha != 1.0:
            out *= alpha
        if bias is not None:
            bias = bias if A.ndim == 2 else bias.unsqueeze(1)
            out += bias
    elif cu_seqlens_m is not None:
        # Handle varlen_m case
        if out is None:
            # When gather_A (A_idx provided), output size is determined by A_idx length
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            out = torch.empty((total_m, B.shape[-1]), dtype=out_dtype, device=A.device)
        for i in range(cu_seqlens_m.shape[0] - 1):
            A_slice = (
                A[A_idx[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]]
                if A_idx is not None
                else A[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]
            )
            torch.mm(A_slice, B[i], out=out[cu_seqlens_m[i] : cu_seqlens_m[i + 1]])
            if not isinstance(alpha, float) or alpha != 1.0:
                out[cu_seqlens_m[i] : cu_seqlens_m[i + 1]] *= alpha
            if bias is not None:
                out[cu_seqlens_m[i] : cu_seqlens_m[i + 1]] += bias[i]
    else:  # cu_seqlens_k is not None
        L = cu_seqlens_k.shape[0] - 1
        if out is None:
            out = torch.empty((L, A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
        for i in range(L):
            A_slice = (
                A[:, A_idx[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]]
                if A_idx is not None
                else A[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
            )
            torch.mm(A_slice, B[cu_seqlens_k[i] : cu_seqlens_k[i + 1], :], out=out[i])
        if not isinstance(alpha, float) or alpha != 1.0:
            out *= alpha
        if bias is not None:
            out += bias
    if concat_layout and "out" in concat_layout:
        # out is n-major (ref allocates contiguous). Split rows (non-contiguous dim).
        out = torch.cat([out[..., ::2, :], out[..., 1::2, :]], dim=-2)
    return out


def gemm_blockscaled_ref(
    A: BlockScaledOperand,  # (M, K) or (L, M, K) logical view
    B: BlockScaledOperand,  # (K, N) or (L, K, N) logical view, K-contig qdata
    alpha: float | Tensor = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize-and-matmul reference for blockscaled GEMM."""
    from torch._vendor.quack.blockscaled.utils import dequant_operand, unpack_scale_blocked_to_2d

    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    assert opA.fmt is not None and opB.fmt is not None, (
        "gemm_blockscaled_ref requires blockscaled A and B"
    )
    mma_kind_for_pair(opA.fmt, opB.fmt)  # legality; mixed MX pairs are fine
    assert opA.fmt.sf_vec_size == opB.fmt.sf_vec_size
    # unpack_scale_blocked_to_2d wants the batched 6-D form
    SFA = opA.sf.unsqueeze(0) if opA.sf.ndim == 5 else opA.sf
    SFB = opB.sf.unsqueeze(0) if opB.sf.ndim == 5 else opB.sf
    sf_vec = opA.fmt.sf_vec_size
    batched = A.ndim == 3
    a3 = A if batched else A.unsqueeze(0)  # (l, m, k_storage)
    b3 = (B if batched else B.unsqueeze(0)).mT  # (l, n, k_storage)
    a_val = dequant_operand(a3, opA.fmt)  # (l, m, k) fp32
    b_val = dequant_operand(b3, opB.fmt)
    l, m, k = a_val.shape
    n = b_val.shape[1]
    sfa = unpack_scale_blocked_to_2d(SFA, m, k // sf_vec).float()
    sfb = unpack_scale_blocked_to_2d(SFB, n, k // sf_vec).float()
    a_dq = a_val * sfa.repeat_interleave(sf_vec, dim=-1)
    b_dq = b_val * sfb.repeat_interleave(sf_vec, dim=-1)
    out = torch.einsum("lmk,lnk->lmn", a_dq, b_dq)
    for scale in (opA.per_tensor_scale, opB.per_tensor_scale):  # NVFP4 per-tensor scales
        if scale is not None:
            out = out * scale
    if not (isinstance(alpha, float) and alpha == 1.0):
        out = out * alpha
    out = out.to(out_dtype)
    return out if batched else out.squeeze(0)


def gemm_add(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    # For blockscaled: a BlockScaledOperand container - see gemm().
    A: Tensor | BlockScaledOperand,
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N); rides the epilogue alongside C
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype | BlockScaledFormat | str] = None,  # format: see gemm()
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
    split_k: Optional[int] = 1,  # K-dim CTAs per tile; None = let the autotuner choose
    split_k_mode: int = SplitKMode.SERIAL,  # see SplitKMode: SERIAL/SEPARATE deterministic, PARALLEL fastest but arrival-order
) -> Tensor:
    """GEMM with addition and optional output tensor:
    D = alpha * A @ B + beta * C [+ bias]. C and bias are independent epilogue
    terms (the kernel entry takes both), so a residual add and a bias can fuse
    into one launch."""
    _reserve_blockscaled_out(out_dtype)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    alpha = _fold_per_tensor_scales(alpha, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(
            SFA, SFB, A.ndim == 3 or cu_seqlens_m is not None or cu_seqlens_k is not None
        )
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16 if SFA is not None else A.dtype
        varlen_m = cu_seqlens_m is not None
        varlen_k = cu_seqlens_k is not None
        if varlen_m:
            # If A_idx is provided (gather_A), use its length; otherwise use A.shape[0]
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            out_shape = (total_m, B.shape[-1])
        elif varlen_k:
            L = cu_seqlens_k.shape[0] - 1
            # For varlen_k, the first dimension is always A.shape[0] (M dimension)
            out_shape = (L, A.shape[0], B.shape[-1])
        else:
            out_shape = (
                (A.shape[0], B.shape[-1]) if A.ndim == 2 else (A.shape[0], A.shape[-2], B.shape[-1])
            )
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    add_to_output = C is out and isinstance(beta, float) and beta == 1.0 and cu_seqlens_m is None
    # Empty-input fast path: skip kernel launch (see gemm() for rationale).
    # K=0 reduces D = alpha*A@B + beta*C [+ bias] to D = beta*C [+ bias].
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        if add_to_output:
            return out  # out IS C, and out += alpha * 0 is a no-op
        _empty_k_matmul_into(out, C=C, beta=beta)
        if bias is not None:
            out += bias if bias.ndim == 1 else bias.unsqueeze(-2)
        return out
    alpha_tensor = alpha if not isinstance(alpha, float) else None
    alpha = alpha if isinstance(alpha, float) else 1.0
    beta_tensor = beta if not isinstance(beta, float) else None
    beta = beta if isinstance(beta, float) else 1.0
    alpha_arg = _merge_tensor(alpha, alpha_tensor)
    beta_arg = _merge_tensor(beta, beta_tensor)
    concat_str = ",".join(concat_layout) if concat_layout else None
    if add_to_output:
        # Pass flat parts: the operands were already unpacked and validated, and
        # per-tensor scales were already folded into alpha above.
        _launch(_gemm_add_inplace_parts)(
            A,
            B,
            out,
            bias=bias,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            alpha=alpha_arg,
            beta=beta_arg,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            concat_layout=concat_str,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
    else:
        _launch(gemm_add_out)(
            A,
            B,
            C,
            out,
            bias,
            alpha,
            beta,
            alpha_tensor,
            beta_tensor,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            add_to_output=add_to_output,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            concat_layout=concat_str,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
    return out


@torch.library.custom_op(
    "torch_vendor_quack::gemm_add_out",
    mutates_args=("out",),
    device_types="cuda",
    # We have to split out alpha and alpha_tensor since torch.library requires
    # each argument to have a fixed type
    # schema="(Tensor A, Tensor B, Tensor C, Tensor(a3!) out, float alpha=1.0, float beta=1.0, Tensor? alpha_tensor=None, Tensor? beta_tensor=None, Tensor? cu_seqlens_m=None, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_add_out(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    """GEMM with addition and pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = _merge_tensor(alpha, alpha_tensor)
    beta = _merge_tensor(beta, beta_tensor)
    fn(
        A,
        B,
        out,
        C,
        bias=bias,
        alpha=alpha,
        beta=beta,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        dynamic_scheduler=dynamic_scheduler,
        concat_layout=_parse_concat_layout(concat_layout),
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


def gemm_add_ref(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    out_dtype: Optional[torch.dtype] = None,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tensor:
    """Reference implementation for GEMM with addition and pre-allocated output."""
    if concat_layout:
        if "A" in concat_layout:
            A = _concat_interleave(A)
        if "B" in concat_layout:
            B = _concat_interleave(B)
        if "bias" in concat_layout and bias is not None:
            bias = _concat_interleave_bias(bias)
        if "C" in concat_layout:
            C = _concat_interleave(C)
    if cu_seqlens_m is None and cu_seqlens_k is None:
        if isinstance(alpha, float) and isinstance(beta, float) and A.ndim == 2:
            # addmm rejects out_dtype=None (omit the kwarg) and is 2D-only;
            # batched inputs take the generic branch below
            dt_kw = {"out_dtype": out_dtype} if out_dtype is not None else {}
            out = torch.addmm(C, A, B, alpha=alpha, beta=beta, out=out, **dt_kw)
        else:
            out_dtype = (
                out.dtype if out is not None else (out_dtype if out_dtype is not None else A.dtype)
            )
            result = (alpha * (A @ B) + beta * C).to(out_dtype)
            if out is not None:
                out.copy_(result)
            else:
                out = result
        if bias is not None:
            bias = bias if A.ndim == 2 else bias.unsqueeze(1)
            out += bias
    elif cu_seqlens_m is not None:
        # Handle varlen_m case
        if out is None:
            # When gather_A (A_idx provided), output size is determined by A_idx length
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            out_dtype = out_dtype if out_dtype is not None else A.dtype
            out = torch.empty((total_m, B.shape[-1]), dtype=out_dtype, device=A.device)
        for i in range(cu_seqlens_m.shape[0] - 1):
            A_slice = (
                A[A_idx[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]]
                if A_idx is not None
                else A[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]
            )
            C_slice = C[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]
            out_slice = out[cu_seqlens_m[i] : cu_seqlens_m[i + 1]]
            result = alpha * torch.mm(A_slice, B[i]) + beta * C_slice
            if bias is not None:
                result += bias[i]
            out_slice.copy_(result)
    else:  # cu_seqlens_k is not None
        # Handle varlen_k case
        L = cu_seqlens_k.shape[0] - 1
        out_dtype = out_dtype if out_dtype is not None else A.dtype
        if out is None:
            out = torch.empty((L, A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
        for i in range(L):
            A_slice = (
                A[:, A_idx[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]]
                if A_idx is not None
                else A[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
            )
            B_slice = B[cu_seqlens_k[i] : cu_seqlens_k[i + 1], :]
            result = alpha * torch.mm(A_slice, B_slice) + beta * C[i]
            out[i].copy_(result)
        if bias is not None:
            out += bias
    if concat_layout and "out" in concat_layout:
        out = torch.cat([out[..., ::2, :], out[..., 1::2, :]], dim=-2)
    return out


# ── functional façades (graph-insertable ops) ────────────────────────────────
# The canonical functional/out-variant pairing: the *_out ops mutate a caller
# buffer (allocation lives in the python wrappers), which is right for eager
# dispatch but means an FX pass cannot insert them directly — the graph would
# need the alloc plus post-functionalization mutation bookkeeping. These ops
# allocate inside and carry real fakes, so compiler passes (and dynamo traces)
# get one clean node per call. Plain dense tensors only: varlen / blockscale /
# out= callers use the python wrappers. No autograd formula — these are
# compiler-facing building blocks; training goes through the autograd
# Functions (or post-grad graphs, which are already differentiated).


@torch.library.custom_op("torch_vendor_quack::gemm_add", mutates_args=(), device_types="cuda")
def _gemm_add_functional(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    C: Tensor,  # (M, N) or (L, M, N)
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tensor:
    """Functional D = alpha * A @ B + beta * C [+ bias]."""
    if A.stride(-1) != 1:
        A = A.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    return gemm_add(
        A,
        B,
        C,
        bias=bias,
        alpha=_merge_tensor(alpha, alpha_tensor),
        beta=_merge_tensor(beta, beta_tensor),
        out_dtype=out_dtype,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
    )


@_gemm_add_functional.register_fake
def _(
    A,
    B,
    C,
    bias=None,
    alpha=1.0,
    beta=1.0,
    alpha_tensor=None,
    beta_tensor=None,
    out_dtype=None,
    dynamic_scheduler=False,
    tuned=True,
):
    return torch.empty_like(C, dtype=out_dtype if out_dtype is not None else A.dtype)


@torch.library.custom_op("torch_vendor_quack::gemm", mutates_args=(), device_types="cuda")
def _gemm_functional(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tensor:
    """Functional D = alpha * A @ B [+ bias]."""
    if A.stride(-1) != 1:
        A = A.contiguous()
    return gemm(
        A,
        B,
        bias=bias,
        alpha=_merge_tensor(alpha, alpha_tensor),
        out_dtype=out_dtype,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
    )


@_gemm_functional.register_fake
def _(
    A,
    B,
    bias=None,
    alpha=1.0,
    alpha_tensor=None,
    out_dtype=None,
    dynamic_scheduler=False,
    tuned=True,
):
    return A.new_empty(
        (*A.shape[:-1], B.shape[-1]), dtype=out_dtype if out_dtype is not None else A.dtype
    )


def gemm_add_inplace(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    # For blockscaled: a BlockScaledOperand container - see gemm().
    A: Tensor | BlockScaledOperand,
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
    split_k: Optional[int] = 1,  # K-dim CTAs per tile; None = let the autotuner choose
    split_k_mode: int = SplitKMode.SERIAL,  # see SplitKMode: SERIAL/SEPARATE deterministic, PARALLEL fastest but arrival-order
) -> None:
    """In-place GEMM with addition: out = alpha * A @ B + beta * out [+ bias].
    Args:
        A: (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k - input tensor
        B: (K, N) or (L, K, N) or (total_K, N) if varlen_k - input tensor
        out: (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k - tensor to accumulate into (modified in-place)
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for out
        cu_seqlens_m: Optional cumulative sequence lengths for variable M
        cu_seqlens_k: Optional cumulative sequence lengths for variable K
        dynamic_scheduler: Whether to use dynamic scheduler
        tuned: Whether to use autotuned configuration
    """
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(
            SFA,
            SFB,
            opA.data.ndim == 3 or cu_seqlens_m is not None or cu_seqlens_k is not None,
        )
    _gemm_add_inplace_parts(
        opA.data,
        opB.data,
        out,
        bias=bias,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=_fold_per_tensor_scales(alpha, opA, opB),
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        concat_layout=",".join(concat_layout)
        if isinstance(concat_layout, tuple)
        else concat_layout,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


def _gemm_add_inplace_parts(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    *,
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    batch_idx_permute: Optional[Tensor] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    """gemm_add_inplace on pre-unpacked operand parts. Also used internally by
    gemm_add's add-to-output path, which passes flat parts instead of rebuilding
    operand containers (the operands were already unpacked and validated)."""
    alpha_tensor = alpha if not isinstance(alpha, float) else None
    alpha = alpha if isinstance(alpha, float) else 1.0
    beta_tensor = beta if not isinstance(beta, float) else None
    beta = beta if isinstance(beta, float) else 1.0
    # Empty-input fast path: out += alpha * A@B with K=0 reduces to out *= beta.
    # The matmul contributes zero, so use the helper with C=out.
    if out.numel() == 0:
        return
    if A.numel() == 0:
        if beta != 1.0 or beta_tensor is not None:
            out.mul_(_merge_tensor(beta, beta_tensor))
        if bias is not None:
            out += bias if bias.ndim == 1 else bias.unsqueeze(-2)
        return
    gemm_add_inplace_op(
        A,
        B,
        out,
        bias,
        alpha,
        beta,
        alpha_tensor,
        beta_tensor,
        cu_seqlens_m,
        cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        concat_layout=concat_layout,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


@torch.library.custom_op(
    "torch_vendor_quack::gemm_add_inplace",
    mutates_args=("out",),
    device_types="cuda",
    # We have to split out alpha and alpha_tensor since torch.library requires
    # each argument to have a fixed type
    # schema="(Tensor A, Tensor B, Tensor(a2!) out, float alpha=1.0, float beta=1.0, Tensor? alpha_tensor=None, Tensor? beta_tensor=None, Tensor? cu_seqlens_m=None, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_add_inplace_op(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = _merge_tensor(alpha, alpha_tensor)
    beta = _merge_tensor(beta, beta_tensor)
    add_to_output = isinstance(beta, float) and beta == 1.0 and cu_seqlens_m is None
    # Use out as both input bias and output
    fn(
        A,
        B,
        out,
        out if not add_to_output else None,
        bias=bias,
        alpha=alpha,
        beta=beta,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        dynamic_scheduler=dynamic_scheduler,
        concat_layout=_parse_concat_layout(concat_layout),
        SFA=SFA,
        SFB=SFB,
        split_k=split_k,
        split_k_mode=split_k_mode,
    )


def gemm_act(
    # For blockscaled: a BlockScaledOperand container - see gemm().
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor | BlockScaledOperand,
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    alpha: float | Tensor = 1.0,  # pre-activation accumulator scale
    preact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    store_preact: bool = True,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    config: Optional[GemmConfig] = None,  # explicit pin (eager only); overrides tuned
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with activation (or gated activation) and optional output tensors.

    ``alpha`` scales the accumulator before the activation
    (``postact = act(alpha * A @ B + C + bias)``). It is applied in fp32 and
    may be a float or a 1-element CUDA tensor; 1.0 keeps the alpha-free epilogue.
    """
    _reserve_blockscaled_out(out_dtype)
    _reserve_blockscaled_out(postact_dtype)
    _check_split_k_unsupported("gemm_act", split_k)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    alpha = _fold_per_tensor_scales(alpha, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(SFA, SFB, A.ndim == 3 or cu_seqlens_m is not None)
    is_gated = activation in gated_to_pytorch_fn_map
    default_dtype = torch.bfloat16 if SFA is not None else A.dtype
    out_dtype = default_dtype if out_dtype is None else out_dtype
    postact_dtype = default_dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    # Determine output shape based on gather_A
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1])
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2) if is_gated else out_shape
    if preact_out is None and store_preact:
        preact_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    # Empty-input fast path. For M=0 or N=0 the outputs are empty; for K=0
    # (A@B == 0) the no-bias / no-C surface yields preact=0 and act(0)=0 for
    # every supported activation, so both outputs are zero.
    if postact_out.numel() == 0 or A.numel() == 0:
        if preact_out is not None:
            _empty_k_matmul_into(preact_out)
        _empty_k_matmul_into(postact_out)
        return preact_out, postact_out
    if torch.compiler.is_compiling():
        if config is not None:
            raise NotImplementedError("gemm_act: explicit config under torch.compile")
        if SFA is not None:
            SFA, SFB = _sf_encode(SFA), _sf_encode(SFB)
        concat_str = ",".join(concat_layout) if concat_layout else None
        op = gemm_gated_out if is_gated else gemm_act_out
        kwargs = {"concat_layout": concat_str} if is_gated else {}
        op(
            A,
            B,
            preact_out,
            postact_out,
            C,
            bias,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            alpha=alpha if isinstance(alpha, float) else 1.0,
            alpha_tensor=alpha if not isinstance(alpha, float) else None,
            **kwargs,
        )
        return preact_out, postact_out
    _gemm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        bias,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=concat_layout if is_gated else None,
        dynamic_scheduler=dynamic_scheduler,
        alpha=alpha,
        tuned=tuned,
        config=config,
    )
    return preact_out, postact_out


gemm_gated = gemm_act


@torch.library.custom_op(
    "torch_vendor_quack::gemm_act_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, Tensor? bias=None, str? activation=None, Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=False, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None, float alpha=1.0, Tensor? alpha_tensor=None) -> ()",
)
def gemm_act_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: ActActivation = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
) -> None:
    """GEMM with activation and pre-allocated output tensors."""
    _gemm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        bias,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        dynamic_scheduler=dynamic_scheduler,
        alpha=_merge_tensor(alpha, alpha_tensor),
        tuned=tuned,
    )


def gemm_act_ref(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    alpha: float | Tensor = 1.0,  # pre-activation accumulator scale
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tuple[Optional[Tensor], Tensor]:
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    if C is None:
        preact = gemm_ref(
            A,
            B,
            bias=bias,
            alpha=alpha,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            concat_layout=concat_layout,
        )
    else:
        preact = gemm_add_ref(
            A,
            B,
            C,
            bias=bias,
            alpha=alpha,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            concat_layout=concat_layout,
        )
    if is_gated:
        # With concat=("B",), gemm_ref already interleaves the output columns,
        # so we always use the interleaved gate/up split.
        gate = preact[..., ::2]
        up = preact[..., 1::2]
        postact = gated_to_pytorch_fn_map[activation](gate, up).to(postact_dtype)
    else:
        postact = act_to_pytorch_fn_map[activation](preact).to(postact_dtype)
    return preact.to(out_dtype) if store_preact else None, postact


gemm_gated_ref = gemm_act_ref


def gemm_dact(
    # For blockscaled: a BlockScaledOperand container - see gemm().
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor | BlockScaledOperand,
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m; or (M, 2*N) for dgated
    activation: Activation = None,
    dx_out: Optional[
        Tensor
    ] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m; double for gated
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    config: Optional[GemmConfig] = None,  # explicit pin (eager only); overrides tuned
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
):
    """GEMM with activation (or gated activation) gradient and optional output tensors."""
    _reserve_blockscaled_out(out_dtype)
    _reserve_blockscaled_out(postact_dtype)
    _check_split_k_unsupported("gemm_dact", split_k)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    if opA.per_tensor_scale is not None or opB.per_tensor_scale is not None:
        raise NotImplementedError(
            "gemm_dact does not support NVFP4 per-tensor scales yet (the dact/dgated "
            "epilogues have no alpha to fold them into)"
        )
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(SFA, SFB, A.ndim == 3 or cu_seqlens_m is not None)
    is_dgated = activation in gated_to_pytorch_fn_map
    default_dtype = torch.bfloat16 if SFA is not None else A.dtype
    out_dtype = default_dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-1] * 2) if is_dgated else (total_m, B.shape[-1])
    elif A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1] * 2) if is_dgated else (A.shape[0], B.shape[-1])
    else:
        n = B.shape[-1] * 2 if is_dgated else B.shape[-1]
        out_shape = (A.shape[0], A.shape[-2], n)
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2) if is_dgated else out_shape
    if dx_out is None:
        dx_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    # Empty-input fast path: M=0 / N=0 → outputs are empty; K=0 (A.numel()==0)
    # makes the upstream GEMM gradient zero, so dx is zero regardless of activation.
    if dx_out.numel() == 0 or A.numel() == 0:
        _empty_k_matmul_into(dx_out)
        _empty_k_matmul_into(postact_out)
        results = [dx_out, postact_out]
        if colvec_reduce:
            colvec_shape = (*out_shape[:-1],)
            results.append(torch.zeros(colvec_shape, dtype=torch.float32, device=A.device))
        return tuple(results)
    if torch.compiler.is_compiling():
        if config is not None:
            raise NotImplementedError("gemm_dact: explicit config under torch.compile")
        if SFA is not None:
            SFA, SFB = _sf_encode(SFA), _sf_encode(SFB)
        out_op = gemm_dgated_out if is_dgated else gemm_dact_out
        colvec_reduce_final = out_op(
            A,
            B,
            PreAct,
            dx_out,
            postact_out,
            colvec_scale,
            activation,
            colvec_reduce,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
        )
    else:
        colvec_reduce_final = _gemm_dact_call(
            A,
            B,
            PreAct,
            dx_out,
            postact_out,
            activation=activation,
            colvec_scale=colvec_scale,
            colvec_reduce=colvec_reduce,
            cu_seqlens_m=cu_seqlens_m,
            A_idx=A_idx,
            SFA=_sf_decode(SFA, bs_format_a),
            SFB=_sf_decode(SFB, bs_format_b),
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            config=config,
        )
    results = [dx_out, postact_out]
    if colvec_reduce:
        results.append(colvec_reduce_final)
    return tuple(results)


gemm_dgated = gemm_dact


@torch.library.custom_op(
    "torch_vendor_quack::gemm_dact_out",
    mutates_args=("dx_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor PreAct, Tensor(a3!) dx_out, Tensor(a4!) postact_out, Tensor? colvec_scale=None, str? activation=None, bool colvec_reduce=False, Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=True, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None) -> Tensor",
)
def gemm_dact_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    dx_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: ActActivation = None,
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> Tensor:
    """GEMM with activation gradient and pre-allocated output tensors."""
    result = _gemm_dact_call(
        A,
        B,
        PreAct,
        dx_out,
        postact_out,
        activation=activation,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
    )
    if result is None:  # Have to return a tensor, not None, to make torch compile happy
        return torch.empty(0, device=A.device, dtype=torch.float32)
    return result


def gemm_dact_ref(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N); or (M, 2*N) for dgated
    activation: Activation = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference implementation for GEMM with activation (or gated activation) gradient."""
    is_dgated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    dout = gemm_ref(A, B, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx).to(out_dtype)
    if is_dgated:
        gate = PreAct[..., ::2]
        up = PreAct[..., 1::2]
        gate_requires_grad, up_requires_grad = gate.requires_grad, up.requires_grad
        gate.requires_grad_(True)
        up.requires_grad_(True)
        postact = gated_to_pytorch_fn_map[activation](gate, up)
        dgate, dup = torch.autograd.grad(postact, [gate, up], dout, create_graph=False)
        gate.requires_grad_(gate_requires_grad)
        up.requires_grad_(up_requires_grad)
        dx = torch.stack([dgate, dup], dim=-1).reshape(PreAct.shape)
        return dx.to(out_dtype), postact.to(postact_dtype)
    else:
        postact = act_to_pytorch_fn_map[activation](PreAct)
        if activation is None:
            dx = dout
        else:
            PreAct_requires_grad = PreAct.requires_grad
            PreAct.requires_grad_(True)
            postact_for_grad = act_to_pytorch_fn_map[activation](PreAct)
            dx = torch.autograd.grad(postact_for_grad, PreAct, dout, create_graph=False)[0]
            PreAct.requires_grad_(PreAct_requires_grad)
        return dx.to(out_dtype), postact.to(postact_dtype)


gemm_dgated_ref = gemm_dact_ref


def _symmetric_gemm_config(sm: int) -> tuple[int, int, int, bool]:
    configs = {
        8: (128, 128, 1, False),
        9: (128, 256, 2, False),
        10: (256, 256, 2, False),
        11: (256, 256, 2, False),
        12: (128, 128, 1, True),
    }
    if sm not in configs:
        raise NotImplementedError(
            "gemm_symmetric is only supported on SM8x, SM90, SM100, SM110, and SM120"
        )
    return configs[sm]


@torch.library.custom_op(
    "torch_vendor_quack::gemm_symmetric_out",
    mutates_args=("out",),
    device_types="cuda",
    # alpha/beta split into float + Tensor pair because torch.library requires
    # each schema arg to have a fixed type. See gemm_add_out for the pattern.
)
def gemm_symmetric_out(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, M) or (L, K, M)
    out: Tensor,  # (M, M) or (L, M, M)
    C: Optional[Tensor] = None,  # (M, M) or (L, M, M)
    dynamic_scheduler: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> None:
    """GEMM with guaranteed symmetric output."""
    alpha = _merge_tensor(alpha, alpha_tensor)
    beta = _merge_tensor(beta, beta_tensor)
    _gemm_symmetric_execute(
        A,
        B,
        out,
        C,
        dynamic_scheduler=dynamic_scheduler,
        alpha=alpha,
        beta=beta,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
    )


def _symmetric_cold(canon, semaphore, config, dynamic_scheduler, ctx):
    t = canon.tensors
    # We want square tile per cluster
    capacity = get_device_capacity(t["A"].device)[0]
    tile_m, tile_n, cluster_m, pingpong = _symmetric_gemm_config(capacity)
    return gemm_symmetric_dispatch(
        t["A"],
        t["B"],
        t["out"],
        t["C"],
        semaphore,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=1,
        pingpong=pingpong,
        persistent=True,
        is_dynamic_persistent=capacity >= 10,
        max_swizzle_size=8,
        alpha=ctx["alpha"],
        beta=ctx["beta"],
        SFA=ctx["SFA"],
        SFB=ctx["SFB"],
        bs_format_a=ctx["bs_format_a"],
        bs_format_b=ctx["bs_format_b"],
    )


def _symmetric_warm(plan, canon, semaphore, ctx):
    # The semaphore is never consumed here (is_dynamic_persistent implies
    # SM100+, whose scheduler uses CLC), so it is ignored.
    t = canon.tensors
    run_gemm_symmetric_plan(
        plan,
        t["A"],
        t["B"],
        t["out"],
        t["C"],
        alpha=ctx["alpha"],
        beta=ctx["beta"],
        SFA=ctx["SFA"],
        SFB=ctx["SFB"],
    )


_SYMMETRIC_SPEC = VariantSpec(
    name="symmetric",
    tensor_roles=(("A", "a"), ("B", "b"), ("out", "mn"), ("C", "mn")),
    cold=_symmetric_cold,
    warm=_symmetric_warm,
    # The symmetric dispatch takes B operand-shaped (m, k): always relabel.
    b_kn_rule=lambda sm90_plus, varlen_m, swap_ab, ctx: False,
    semaphore=lambda dynamic, capacity, device, warm: (
        torch.zeros(1, dtype=torch.int32, device=device) if dynamic and not warm else None
    ),
)


def _gemm_symmetric_execute(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor],
    *,
    dynamic_scheduler: bool,
    alpha: float | Tensor,
    beta: float | Tensor,
    SFA: Optional[Tensor] = None,  # decoded (real-dtype) scale factors
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,  # BlockScaledFormat names (see quack.gemm.gemm)
    bs_format_b: Optional[str] = None,
    dispatch_plan=None,
):
    """Launch the symmetric GEMM through the generic variant engine."""
    return run_variant(
        _SYMMETRIC_SPEC,
        dict(A=A, B=B, out=out, C=C),
        config=None,
        dynamic_scheduler=dynamic_scheduler,
        ctx=dict(
            alpha=alpha,
            beta=beta,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
        ),
        dispatch_plan=dispatch_plan,
    )


_gemm_symmetric_iface_plan_cache: dict[tuple, IfacePlan] = {}


def gemm_symmetric(
    # For blockscaled: a BlockScaledOperand container - see gemm(). B must be
    # A.mT of the same quantized tensor (that is what makes D symmetric).
    A: Tensor | BlockScaledOperand,  # (M, K) or (L, M, K)
    B: Tensor | BlockScaledOperand,  # (K, M) or (L, K, M)
    C: Optional[Tensor] = None,  # (M, M) or (L, M, M)
    out: Optional[Tensor] = None,  # (M, M) or (L, M, M)
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with symmetric output."""
    _reserve_blockscaled_out(out_dtype)
    _check_split_k_unsupported("gemm_symmetric", split_k)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    if SFA is not None and bs_format_a != bs_format_b:
        raise ValueError(
            "gemm_symmetric requires matching A/B formats (B is A.mT of the same "
            f"quantized tensor); got {bs_format_a} / {bs_format_b}"
        )
    alpha = _fold_per_tensor_scales(alpha, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(SFA, SFB, A.ndim == 3)
    # Eager plan fast path; see gemm(). The key subsumes the dispatch key
    # (alpha/beta modes select compiled epilogues).
    plan_key = None
    if not torch.compiler.is_compiling():
        plan_key = (
            tensor_key(A),
            tensor_key(B),
            tensor_key(C),
            tensor_key(out),
            A.device,
            out_dtype,
            dynamic_scheduler,
            scalar_mode(alpha),
            scalar_mode(beta),
            tensor_key(SFA),
            tensor_key(SFB),
            bs_format_a,
            bs_format_b,
        )
        plan = _gemm_symmetric_iface_plan_cache.get(plan_key)
        if plan is not None:
            out = alloc_outputs(plan, dict(out=out), A.device)["out"]
            # No empty-input checks: empty calls return before recording below.
            plan.replay(
                dict(A=A, B=B, out=out, C=C),
                dict(
                    alpha=alpha,
                    beta=beta,
                    SFA=SFA,
                    SFB=SFB,
                    bs_format_a=bs_format_a,
                    bs_format_b=bs_format_b,
                ),
            )
            return out
    default_dtype = torch.bfloat16 if SFA is not None else A.dtype
    out_dtype = default_dtype if out_dtype is None else out_dtype
    if A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)

    # Empty-input fast path: out = alpha * A@A.T + beta * C reduces to beta * C
    # when K=0 (or just zeros / empty for M=0).
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        _empty_k_matmul_into(out, C=C, beta=beta)
        return out

    if torch.compiler.is_compiling():
        alpha_tensor = alpha if not isinstance(alpha, float) else None
        alpha_val = alpha if isinstance(alpha, float) else 1.0
        beta_tensor = beta if not isinstance(beta, float) else None
        beta_val = beta if isinstance(beta, float) else 1.0
        if SFA is not None:
            SFA, SFB = _sf_encode(SFA), _sf_encode(SFB)
        gemm_symmetric_out(
            A,
            B,
            out,
            C,
            dynamic_scheduler=dynamic_scheduler,
            alpha=alpha_val,
            beta=beta_val,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
        )
        return out
    dispatch_plan = _gemm_symmetric_execute(
        A,
        B,
        out,
        C,
        dynamic_scheduler=dynamic_scheduler,
        alpha=alpha,
        beta=beta,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
    )
    if plan_key is not None:
        _gemm_symmetric_iface_plan_cache[plan_key] = make_iface_plan(
            _SYMMETRIC_SPEC,
            dict(A=A, B=B, out=out, C=C),
            config=None,
            dynamic_scheduler=dynamic_scheduler,
            out_recipes=(("out", out_shape, out_dtype),),
            dispatch_plan=dispatch_plan,
        )
    return out


@torch.library.custom_op(
    "torch_vendor_quack::gemm_gated_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, Tensor? bias=None, str activation='swiglu', Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=False, bool tuned=True, str? concat_layout=None, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None, float alpha=1.0, Tensor? alpha_tensor=None) -> ()",
)
def gemm_gated_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
) -> None:
    """GEMM with gated activation and pre-allocated output tensors."""
    _gemm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        bias,
        activation=activation,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=_parse_concat_layout(concat_layout),
        dynamic_scheduler=dynamic_scheduler,
        alpha=_merge_tensor(alpha, alpha_tensor),
        tuned=tuned,
    )


@torch.library.custom_op(
    "torch_vendor_quack::gemm_dgated_out",
    mutates_args=("dx_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor PreAct, Tensor(a!) dx_out, Tensor(b!) postact_out, Tensor? colvec_scale=None, str activation='swiglu', bool colvec_reduce=False, Tensor? cu_seqlens_m=None, Tensor? A_idx=None, bool dynamic_scheduler=True, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None) -> Tensor",
)
def gemm_dgated_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    dx_out: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: GatedActivation = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> Tensor:
    """GEMM with gated activation gradient and pre-allocated output tensors."""
    result = _gemm_dact_call(
        A,
        B,
        PreAct,
        dx_out,
        postact_out,
        activation=activation,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
    )
    if result is None:  # Have to return a tensor, not None, to make torch compile happy
        return torch.empty(0, device=A.device, dtype=torch.float32)
    return result


def _colvec_reduce_fake(
    A: Tensor,
    colvec_reduce: bool,
    cu_seqlens_m: Optional[Tensor],
    A_idx: Optional[Tensor],
) -> Tensor:
    """Fake colvec-reduce output for the dact/dgated custom ops (empty if off)."""
    if not colvec_reduce:
        return torch.empty(0, dtype=torch.float32, device=A.device)
    if cu_seqlens_m is not None:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m,)
    elif A.ndim == 2:
        out_shape = (A.shape[0],)
    else:
        out_shape = (A.shape[0], A.shape[-2])
    return torch.empty(out_shape, dtype=torch.float32, device=A.device)


@torch.library.register_fake("torch_vendor_quack::gemm_dgated_out")
def gemm_dgated_out_fake(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    dx_out: Tensor,
    postact_out: Tensor,
    colvec_scale: Optional[Tensor] = None,
    activation: str = "swiglu",
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> Tensor:
    return _colvec_reduce_fake(A, colvec_reduce, cu_seqlens_m, A_idx)


@torch.library.register_fake("torch_vendor_quack::gemm_dact_out")
def gemm_dact_out_fake(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    dx_out: Tensor,
    postact_out: Tensor,
    colvec_scale: Optional[Tensor] = None,
    activation: Optional[str] = None,
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    dynamic_scheduler: bool = True,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> Tensor:
    return _colvec_reduce_fake(A, colvec_reduce, cu_seqlens_m, A_idx)


@gemm_add_inplace_op.register_fake
def gemm_add_inplace_fake(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    batch_idx_permute: Optional[Tensor] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: Optional[str] = None,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> None:
    # Pure no-op: the op only mutates ``out``; kernel compilation is owned
    # by jit_cache + the async compile pool at real execution time.
    return


# ---------------------------------------------------------------------------
# Shared schema-split helpers.
#
# torch.library.custom_op requires a concrete type per arg, so union-typed
# autotuned args (e.g. ``alpha: Union[float, Tensor]``, ``sr_seed: Union[int,
# Tensor]``) are split into two fixed-typed schema kwargs at the custom_op
# boundary (``alpha: float`` + ``alpha_tensor: Optional[Tensor]``). The eager
# bodies merge them back into the unified form via :func:`_merge_tensor`
# before calling the autotuned fn.
# ---------------------------------------------------------------------------


def _merge_tensor(value, tensor_value):
    """Return ``tensor_value`` if non-None, else ``value``.

    Single source of truth for the ``Union[scalar, Tensor]`` schema-split
    merge. Used both inside eager bodies (where ``value = alpha,
    tensor_value = alpha_tensor``) and inside the fake path (which derives
    the split pairs from the custom_op signature).
    """
    return tensor_value if tensor_value is not None else value


def _parse_concat_layout(value):
    """Coerce ``concat_layout`` from schema form (``Optional[str]``) to
    autotuned form (``Optional[tuple[str, ...]]``).

    custom_op schemas can't express ``tuple[str, ...]``, so callers pass a
    comma-separated string. The autotuned fn keys on a tuple (via
    ``tuple(sorted(concat_layout))``); a stray string would be iterated
    char-by-char and silently produce a wrong, never-used compile signature.
    Single source of truth used by both eager bodies and the fake path.
    """
    if value is None or isinstance(value, tuple):
        return value
    return tuple(value.split(",")) if value else None


def _register_noop_fake(custom_op):
    """Register a pure no-op fake for a mutating custom op.

    These ops only mutate their ``out`` argument, so Dynamo / AOT autograd
    need no shape effect from the fake; kernel compilation is owned by
    jit_cache + the async compile pool at real execution time.
    """

    @custom_op.register_fake
    def _fake(*args, **kwargs):
        return


_register_noop_fake(gemm_out)
_register_noop_fake(gemm_add_out)
_register_noop_fake(gemm_act_out)
_register_noop_fake(gemm_gated_out)


@gemm_symmetric_out.register_fake
def gemm_symmetric_out_fake(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    dynamic_scheduler: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    alpha_tensor: Optional[Tensor] = None,
    beta_tensor: Optional[Tensor] = None,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
) -> None:
    # Pure no-op: the op only mutates ``out``; kernel compilation is owned
    # by jit_cache + the async compile pool at real execution time.
    return


## ── gemm_rms ────────────────────────────────────────────────────────────────
# Ported to the epilogue-object surface: quack.epilogue.library.sq_reduce_mod owns
# canonicalization, plan caching, and tuning; the wrapper binds the operand
# presence pattern to the right mod and fuses the final rstd reduction
# (rms_final_reduce over the raw per-tile sq-sum partials — the same second
# kernel as before the port, so numerics are bitwise-unchanged).


def _gemm_rms_call(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor],
    norm_weight: Optional[Tensor],
    premult_out: Optional[Tensor],
    *,
    dynamic_scheduler: bool,
    tuned: bool,
    config: Optional[GemmConfig] = None,
    SFA: Optional[Tensor] = None,  # decoded (real-dtype) scale factors
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,  # BlockScaledFormat names (see quack.gemm.gemm)
    bs_format_b: Optional[str] = None,
    alpha: float | Tensor = 1.0,  # accumulator scale (folded NVFP4 per-tensor scales)
) -> Tensor:
    """Launch the sq_reduce GEMM on the epilogue object; returns the raw
    (..., n_tiles) per-tile squared-sum partials."""
    from torch._vendor.quack.epilogue.library import sq_reduce_mod

    has_alpha = scalar_mode(alpha) != 0
    mod = sq_reduce_mod(
        has_c=C is not None,
        has_rowvec=norm_weight is not None,
        has_aux=premult_out is not None,
        has_alpha=has_alpha,
    )
    outs = {"D": out}
    operands = {}
    if premult_out is not None:
        outs["mAuxOut"] = premult_out
    if norm_weight is not None:
        operands["mRowVecBroadcast"] = norm_weight
    if has_alpha:
        operands["alpha"] = alpha
    res = mod(
        A,
        B,
        C,
        out=outs,
        config=config,
        tuned=tuned,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        **operands,
    )
    return res["mColVecReduce"]


def _rms_finalize(partials: Tensor, N: int, eps: float, rstd_shape) -> Tensor:
    # Final reduction: rstd = rsqrt(sum(partials) / N + eps). The reshape
    # copies only on the cold tuned call (the winning slice of the sweep's
    # worst-case buffer is strided); warm calls allocate exact-shape partials.
    n_tiles = partials.shape[-1]
    rstd_flat = rms_final_reduce(partials.reshape(-1, n_tiles), scale=1.0 / N, eps=eps)
    return rstd_flat.reshape(rstd_shape)


@torch.library.custom_op(
    "torch_vendor_quack::gemm_rms_out",
    mutates_args=("out", "premult_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a!) out, Tensor? C=None, Tensor? norm_weight=None, Tensor(a2!)? premult_out=None, float eps=1e-6, bool dynamic_scheduler=False, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None, Tensor? alpha_tensor=None) -> Tensor",
)
def _gemm_rms_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    norm_weight: Optional[Tensor] = None,
    premult_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha_tensor: Optional[Tensor] = None,  # folded NVFP4 per-tensor scales
) -> Tensor:
    """GEMM + RMS + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    """
    partials = _gemm_rms_call(
        A,
        B,
        out,
        C,
        norm_weight,
        premult_out,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=_merge_tensor(1.0, alpha_tensor),
    )
    return _rms_finalize(partials, B.shape[-1], eps, A.shape[:-1])


@torch.library.register_fake("torch_vendor_quack::gemm_rms_out")
def _gemm_rms_out_fake(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    norm_weight: Optional[Tensor] = None,
    premult_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha_tensor: Optional[Tensor] = None,
) -> Tensor:
    rstd_shape = A.shape[:-1]
    return torch.empty(rstd_shape, dtype=torch.float32, device=A.device)


def gemm_rms_ref(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    norm_weight: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Reference: D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D = D_raw * norm_weight."""
    fn = torch.bmm if A.ndim == 3 else torch.mm
    D = fn(A, B)
    if C is not None:
        D = D + C
    rstd = torch.rsqrt(D.float().square().mean(dim=-1) + eps)
    if norm_weight is not None:
        D = D * norm_weight
    return D, rstd


def gemm_rms(
    # For blockscaled: a BlockScaledOperand container - see gemm().
    A: Tensor | BlockScaledOperand,  # (M, K) or (L, M, K)
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    norm_weight: Optional[Tensor] = None,  # (N,) or (L, N)
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    out_dtype: Optional[torch.dtype] = None,
    premult_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) — pre-norm_weight snapshot
    eps: float = 1e-6,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    config: Optional[GemmConfig] = None,  # explicit pin (eager only); overrides tuned
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> Tuple[Tensor, Tensor]:
    """GEMM + RMS statistics + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    Returns (D_out, rstd).
    """
    _reserve_blockscaled_out(out_dtype)
    _check_split_k_unsupported("gemm_rms", split_k)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    # NVFP4 per-tensor scales fold into the accumulator scale; they MUST land
    # before the sq-reduce (RMS scale invariance would otherwise hide the error
    # in D while rstd / premult_out come out wrong by the scale).
    alpha = _fold_per_tensor_scales(1.0, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(SFA, SFB, A.ndim == 3)
    default_dtype = torch.bfloat16 if SFA is not None else A.dtype
    out_dtype = default_dtype if out_dtype is None else out_dtype
    N = B.shape[-1]
    if out is None:
        out_shape = (*A.shape[:-1], N)
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    # Empty-input fast path. Skipping the kernel also avoids a torch.library
    # adinplaceorview_impl IndexError that fires on empty inputs because
    # premult_out's positional slot isn't materialized in the boxed args tuple.
    # K=0 with no C reduces the matmul to zero, so D = 0 and rstd = rsqrt(eps).
    if out.numel() == 0 or A.numel() == 0:
        _empty_k_matmul_into(out)
        if premult_out is not None:
            _empty_k_matmul_into(premult_out)
        rstd_shape = A.shape[:-1]
        if A.numel() == 0 and out.numel() > 0:
            # K=0: rstd = rsqrt(0 + eps) for every row.
            rstd = torch.full(rstd_shape, eps**-0.5, dtype=torch.float32, device=A.device)
        else:
            rstd = torch.empty(rstd_shape, dtype=torch.float32, device=A.device)
        return out, rstd
    if torch.compiler.is_compiling():
        # The opaque alias op keeps tuning at real execution time (with reduce
        # sinks, the generic quack::gemm_epi path would pin the config so the
        # partials can be graph-allocated); rms_final_reduce stays inside it.
        if config is not None:
            raise NotImplementedError("gemm_rms: explicit config under torch.compile")
        if SFA is not None:
            SFA, SFB = _sf_encode(SFA), _sf_encode(SFB)
        rstd = _gemm_rms_out(
            A,
            B,
            out,
            C=C,
            norm_weight=norm_weight,
            premult_out=premult_out,
            eps=eps,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            alpha_tensor=None if isinstance(alpha, float) else alpha,
        )
        return out, rstd
    partials = _gemm_rms_call(
        A,
        B,
        out,
        C,
        norm_weight,
        premult_out,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        config=config,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=alpha,
    )
    return out, _rms_finalize(partials, N, eps, A.shape[:-1])


## ── gemm_norm_act ─────────────────────────────────────────────────────────────
# Ported to the epilogue-object surface (see the gemm_rms note above):
# quack.epilogue.library.norm_act_mod owns canonicalization, plan caching, and tuning
# (element-mode norm_act keeps swap_ab configs via swap-at-trace; the gated
# config space never had swap_ab).


def _gemm_norm_act_call(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor],
    rstd: Optional[Tensor],
    *,
    activation,
    dynamic_scheduler: bool,
    tuned: bool,
    config: Optional[GemmConfig] = None,
    SFA: Optional[Tensor] = None,  # decoded (real-dtype) scale factors
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,  # BlockScaledFormat names (see quack.gemm.gemm)
    bs_format_b: Optional[str] = None,
    alpha: float | Tensor = 1.0,  # accumulator scale (folded NVFP4 per-tensor scales)
) -> None:
    from torch._vendor.quack.epilogue.library import norm_act_mod

    has_alpha = scalar_mode(alpha) != 0
    mod = norm_act_mod(
        activation,
        gated=activation in gated_to_pytorch_fn_map,
        has_c=C is not None,
        has_rowvec=False,
        has_colvec=rstd is not None,
        has_alpha=has_alpha,
    )
    outs = {"mAuxOut": postact_out}
    store_d = preact_out is not None
    if store_d:
        outs["D"] = preact_out
    operands = {}
    if rstd is not None:
        operands["mColVecBroadcast"] = rstd
    if has_alpha:
        operands["alpha"] = alpha
    mod(
        A,
        B,
        C,
        out=outs,
        store_d=store_d,
        config=config,
        tuned=tuned,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        **operands,
    )


@torch.library.custom_op(
    "torch_vendor_quack::gemm_norm_act_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, Tensor? rstd=None, str? activation=None, bool dynamic_scheduler=False, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None, Tensor? alpha_tensor=None) -> ()",
)
def gemm_norm_act_out(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    activation: ActActivation = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha_tensor: Optional[Tensor] = None,  # folded NVFP4 per-tensor scales
) -> None:
    _gemm_norm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        rstd,
        activation=activation,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=_merge_tensor(1.0, alpha_tensor),
    )


_register_noop_fake(gemm_norm_act_out)


@torch.library.custom_op(
    "torch_vendor_quack::gemm_norm_gated_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, Tensor? rstd=None, str activation='swiglu', bool dynamic_scheduler=False, bool tuned=True, Tensor? SFA=None, Tensor? SFB=None, str? bs_format_a=None, str? bs_format_b=None, Tensor? alpha_tensor=None) -> ()",
)
def gemm_norm_gated_out(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    activation: GatedActivation = "swiglu",
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    SFA: Optional[Tensor] = None,  # blocked scale factors, (L, rm, rk, 32, 4, 4) (see gemm_out)
    SFB: Optional[Tensor] = None,
    bs_format_a: Optional[str] = None,
    bs_format_b: Optional[str] = None,
    alpha_tensor: Optional[Tensor] = None,  # folded NVFP4 per-tensor scales
) -> None:
    _gemm_norm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        rstd,
        activation=activation,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=_merge_tensor(1.0, alpha_tensor),
    )


_register_noop_fake(gemm_norm_gated_out)


def gemm_norm_act(
    # For blockscaled: a BlockScaledOperand container - see gemm().
    A: Tensor | BlockScaledOperand,  # (M, K) or (L, M, K)
    B: Tensor | BlockScaledOperand,  # (K, N) or (L, K, N)
    rstd: Optional[Tensor] = None,  # (M,) or (L, M)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) — residual
    activation: Activation = None,
    preact_out: Optional[Tensor] = None,
    postact_out: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = False,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    config: Optional[GemmConfig] = None,  # explicit pin (eager only); overrides tuned
    split_k: Optional[int] = 1,
    split_k_mode: int = SplitKMode.SERIAL,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM + normalize + activation: PostAct = act((A @ B + C) * rstd).

    rstd is a column vector (M,).
    Returns (preact, postact) where preact is the normalized value before activation.
    """
    _reserve_blockscaled_out(out_dtype)
    _reserve_blockscaled_out(postact_dtype)
    _check_split_k_unsupported("gemm_norm_act", split_k)
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    A, B = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    # NVFP4 per-tensor scales scale the matmul product ONLY: they fold into the
    # mod's alpha, applied before the C add and the rstd/norm scales.
    alpha = _fold_per_tensor_scales(1.0, opA, opB)
    if SFA is not None:
        SFA, SFB = _sf_batch_canonicalize(SFA, SFB, A.ndim == 3)
    is_gated = activation in gated_to_pytorch_fn_map
    default_dtype = torch.bfloat16 if SFA is not None else A.dtype
    out_dtype = default_dtype if out_dtype is None else out_dtype
    postact_dtype = default_dtype if postact_dtype is None else postact_dtype
    if A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    postact_shape = (*out_shape[:-1], out_shape[-1] // 2) if is_gated else out_shape
    if preact_out is None and store_preact:
        preact_out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty(postact_shape, dtype=postact_dtype, device=A.device)
    # Empty-input fast path: skip kernel; zero both outputs (act(0)=0 for all
    # supported activations under the no-bias/no-C path of this test surface).
    if postact_out.numel() == 0 or A.numel() == 0:
        if preact_out is not None:
            _empty_k_matmul_into(preact_out)
        _empty_k_matmul_into(postact_out)
        return preact_out, postact_out
    if torch.compiler.is_compiling():
        if config is not None:
            raise NotImplementedError("gemm_norm_act: explicit config under torch.compile")
        if SFA is not None:
            SFA, SFB = _sf_encode(SFA), _sf_encode(SFB)
        op = gemm_norm_gated_out if is_gated else gemm_norm_act_out
        op(
            A,
            B,
            preact_out,
            postact_out,
            C,
            rstd,
            activation,
            dynamic_scheduler,
            tuned,
            SFA=SFA,
            SFB=SFB,
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            alpha_tensor=None if isinstance(alpha, float) else alpha,
        )
        return preact_out, postact_out
    _gemm_norm_act_call(
        A,
        B,
        preact_out,
        postact_out,
        C,
        rstd,
        activation=activation,
        dynamic_scheduler=dynamic_scheduler,
        tuned=tuned,
        config=config,
        SFA=_sf_decode(SFA, bs_format_a),
        SFB=_sf_decode(SFB, bs_format_b),
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        alpha=alpha,
    )
    return preact_out, postact_out


gemm_norm_gated = gemm_norm_act


def gemm_norm_act_ref(
    A: Tensor,
    B: Tensor,
    rstd: Optional[Tensor] = None,  # (M,) or (L, M)
    C: Optional[Tensor] = None,
    activation: Activation = None,
    store_preact: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Optional[Tensor], Tensor]:
    """Reference: preact = (A @ B + C) * rstd, postact = act(preact)."""
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    fn = torch.bmm if A.ndim == 3 else torch.mm
    D = fn(A, B)
    if C is not None:
        D = D + C
    if rstd is not None:
        D = D * rstd.unsqueeze(-1)
    preact = D.to(out_dtype) if store_preact else None
    if is_gated:
        gate = D[..., ::2]
        up = D[..., 1::2]
        postact = gated_to_pytorch_fn_map[activation](gate, up).to(postact_dtype)
    else:
        postact = act_to_pytorch_fn_map[activation](D).to(postact_dtype)
    return preact, postact


gemm_norm_gated_ref = gemm_norm_act_ref


# TODO: this is not quite right, do we need to register gemm_add not gemm_add_out?
# try:
#     from torch._inductor.fx_passes.reinplace import InplaceableOp
#     torch._inductor.fx_passes.reinplace.inplaceable_ops.update({
#         torch.ops.quack.gemm_add_out.default:
#         InplaceableOp(torch.ops.quack.gemm_add_inplace.default, mutated_arg=2)
#     })
# except ImportError:
#     pass
