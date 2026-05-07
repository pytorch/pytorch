# Copyright (c) 2025, Tri Dao
from typing import Optional, Tuple, Literal
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from .gemm_config import GemmConfig, get_all_configs

from .autotuner import autotune, AutotuneConfig
from .cute_dsl_utils import get_device_capacity
from .gemm import gemm as gemm_dispatch
from .gemm_act import gemm_act as gemm_act_dispatch
from .gemm_dact import gemm_dact as gemm_dact_dispatch
from .gemm_symmetric import gemm_symmetric as gemm_symmetric_dispatch
from .gemm_sq_reduce import gemm_sq_reduce as gemm_sq_reduce_dispatch
from .gemm_norm_act import gemm_norm_act_fn as gemm_norm_act_dispatch
from .rms_final_reduce import rms_final_reduce
from .rounding import RoundingMode


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


ActActivation = Literal[None, "silu", "silu-tanh", "relu", "relu_sq", "gelu_tanh_approx"]
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
    "swiglu",
    "swiglu-tanh",
    "swiglu_oai",
    "swiglu_oai-tanh",
    "reglu",
    "geglu",
    "glu",
]


def _concat_interleave(t):
    """Interleave halves along non-contiguous dim: [first; second] → [f0, s0, f1, ...]"""
    dim = -2 if t.stride(-1) == 1 else -1
    return t.unflatten(dim, (2, t.shape[dim] // 2)).transpose(dim - 1, dim).flatten(dim - 1, dim)


def _concat_interleave_bias(t):
    """Interleave [gate; up] along last dim for bias vectors."""
    half = t.shape[-1] // 2
    return t.unflatten(-1, (2, half)).transpose(-2, -1).flatten(-2, -1)


def default_config(device):
    cap = get_device_capacity(device)[0]
    if cap == 8:
        return GemmConfig(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            num_warps=4,
            cluster_m=1,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
            device_capacity=8,
        )
    elif cap in [10, 11]:
        return GemmConfig(
            tile_m=256,
            tile_n=256,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=True,
            device_capacity=10,
        )
    elif cap == 12:
        return GemmConfig(
            tile_m=128,
            tile_n=128,
            cluster_m=1,
            cluster_n=1,
            pingpong=True,
            is_dynamic_persistent=True,
            device_capacity=12,
        )
    else:
        return GemmConfig(
            tile_m=128,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=True,
            is_dynamic_persistent=False,
        )


def nvmmh_config(A, B, device_capacity):
    """Use nvMatmulHeuristics to pick a config for pure GEMM (no varlen/gather/epilogue).

    Returns None if unavailable, caller should fall back to default_config.
    """
    try:
        from .nvmmh_heuristic import nvmmh_default_config

        return nvmmh_default_config(A, B, device_capacity)
    except Exception:
        return None


def prune_invalid_gemm_configs(configs, named_args: dict, **kwargs):
    kwargs = named_args | kwargs
    device_capacity = get_device_capacity(kwargs["A"].device)[0]
    configs = [conf for conf in configs if conf.kwargs["config"].device_capacity == device_capacity]
    gather_A = kwargs.get("A_idx", None) is not None
    varlen_m = kwargs.get("cu_seqlens_m", None) is not None
    if varlen_m or gather_A:  # Doesn't support swap_ab
        configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    if gather_A:
        configs = [conf for conf in configs if conf.kwargs["config"].cluster_n == 1]
        if device_capacity == 9:
            configs = [conf for conf in configs if conf.kwargs["config"].tile_n != 208]
            configs = [conf for conf in configs if not conf.kwargs["config"].is_dynamic_persistent]
    # use_tma_gather only valid when gather_A is active on SM100/SM110
    if not gather_A or device_capacity not in [10, 11]:
        configs = [conf for conf in configs if not conf.kwargs["config"].use_tma_gather]
    return configs


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["dynamic_scheduler"],
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
) -> None:
    if config is None:
        # Use nvMMH heuristic for pure GEMM (no varlen, no gather, no epilogue)
        is_pure_gemm = (
            cu_seqlens_m is None
            and cu_seqlens_k is None
            and A_idx is None
            and C is None
            and bias is None
            and not add_to_output
        )
        if is_pure_gemm:
            device_capacity = get_device_capacity(A.device)[0]
            config = nvmmh_config(A, B, device_capacity)
        if config is None:
            config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    if gather_A:
        assert varlen, "gather_A requires either varlen_m or varlen_k"
        assert config.cluster_n == 1, "gather_A requires cluster_n=1"
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K) or (N, total_K)
    if B.ndim == 2 and not varlen_k:
        B = B.unsqueeze(0)  # (1, N, K)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if out.ndim == 2 and not varlen_m:
        out = out.unsqueeze(0)
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    batch_size = B.shape[0] if not varlen_k else cu_seqlens_k.shape[0] - 1
    if varlen_m:
        # If gather_A (A_idx provided), use its length; otherwise use A.shape[0]
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-2])
    else:
        out_shape = (batch_size, A.shape[-2], B.shape[-2])
    assert out.shape == out_shape, f"out shape mismatch: {out.shape} vs {out_shape}"
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
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
    _swap_map = {"A": "B", "B": "A", "out": "out", "C": "C", "mRowVecBroadcast": "mColVecBroadcast"}
    swapped_concat = (
        tuple(_swap_map.get(k, k) for k in concat_layout)
        if config.swap_ab and concat_layout
        else concat_layout
    )
    gemm_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        out if not config.swap_ab else out.mT,
        (C if not config.swap_ab else C.mT) if C is not None else None,
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
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
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
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_act_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m - None if not storing preact
    preact_out: Optional[Tensor],
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: ActActivation = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_act_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_dact_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    dx_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, N, N) or (total_M, N) if varlen_m
    activation: ActActivation = None,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if PreAct.ndim == 2 and not varlen_m:
        PreAct = PreAct.unsqueeze(0)  # (1, M, N)
    if dx_out.ndim == 2 and not varlen_m:
        D = dx_out.unsqueeze(0)
    else:
        D = dx_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_dact_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        PreAct if not config.swap_ab else PreAct.mT,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
    )


def gemm(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    alpha: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tensor:
    """GEMM with optional output tensor and tuning control."""
    if out is None:
        out_dtype = A.dtype if out_dtype is None else out_dtype
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
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
    # Empty-input fast path: skip kernel launch.
    # M=0 / N=0 — the tile scheduler's ceil_div over a zero dim divides by zero.
    # K=0 — the kernel rejects stride-0 inputs (stride must be divisible by 8);
    #       semantically the empty contraction yields a zero matrix.
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        _empty_k_matmul_into(out, bias=bias)
        return out
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
    )
    return out


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
) -> None:
    """GEMM with pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = alpha_tensor if alpha_tensor is not None else alpha
    sr_seed_arg = sr_seed_tensor if sr_seed_tensor is not None else sr_seed
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
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
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


def gemm_add(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tensor:
    """GEMM with addition and optional output tensor."""
    if out is None:
        out_dtype = A.dtype if out_dtype is None else out_dtype
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
    # K=0 reduces D = alpha*A@B + beta*C to D = beta*C.
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        if add_to_output:
            return out  # out IS C, and out += alpha * 0 is a no-op
        _empty_k_matmul_into(out, C=C, beta=beta)
        return out
    alpha_tensor = alpha if not isinstance(alpha, float) else None
    alpha = alpha if isinstance(alpha, float) else 1.0
    beta_tensor = beta if not isinstance(beta, float) else None
    beta = beta if isinstance(beta, float) else 1.0
    alpha_arg = alpha_tensor if alpha_tensor is not None else alpha
    beta_arg = beta_tensor if beta_tensor is not None else beta
    concat_str = ",".join(concat_layout) if concat_layout else None
    if add_to_output:
        gemm_add_inplace(
            A,
            B,
            out,
            alpha=alpha_arg,
            beta=beta_arg,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
            batch_idx_permute=batch_idx_permute,
            dynamic_scheduler=dynamic_scheduler,
            tuned=tuned,
            concat_layout=concat_str,
        )
    else:
        gemm_add_out(
            A,
            B,
            C,
            out,
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
        )
    return out


def gemm_add_out(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
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
) -> None:
    """GEMM with addition and pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = alpha_tensor if alpha_tensor is not None else alpha
    beta = beta_tensor if beta_tensor is not None else beta
    fn(
        A,
        B,
        out,
        C,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        dynamic_scheduler=dynamic_scheduler,
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
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
        if isinstance(alpha, float) and isinstance(beta, float):
            out = torch.addmm(C, A, B, out_dtype=out_dtype, alpha=alpha, beta=beta, out=out)
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


def gemm_add_inplace(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) or (total_K,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (L,) permutation of batch indices for scheduler
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> None:
    """In-place GEMM with addition: out = alpha * A @ B + beta * out.
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
            out.mul_(beta_tensor if beta_tensor is not None else beta)
        return
    gemm_add_inplace_op(
        A,
        B,
        out,
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
        concat_layout=",".join(concat_layout)
        if isinstance(concat_layout, tuple)
        else concat_layout,
    )


def gemm_add_inplace_op(
    # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A with varlen_m or (M, whatever) if gather_A with varlen_k
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m or (L, M, N) if varlen_k
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
) -> None:
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    alpha = alpha_tensor if alpha_tensor is not None else alpha
    beta = beta_tensor if beta_tensor is not None else beta
    add_to_output = isinstance(beta, float) and beta == 1.0 and cu_seqlens_m is None
    # Use out as both input bias and output
    fn(
        A,
        B,
        out,
        out if not add_to_output else None,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        dynamic_scheduler=dynamic_scheduler,
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
    )


def gemm_act(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
    preact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    store_preact: bool = True,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with activation (or gated activation) and optional output tensors."""
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
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
    concat_str = ",".join(concat_layout) if concat_layout else None
    if is_gated:
        gemm_gated_out(
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
            concat_layout=concat_str,
        )
    else:
        gemm_act_out(
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
        )
    return preact_out, postact_out


gemm_gated = gemm_act


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
) -> None:
    """GEMM with activation and pre-allocated output tensors."""
    fn = gemm_act_tuned if tuned else partial(gemm_act_tuned.fn, config=None)
    fn(A, B, preact_out, postact_out, C, bias, activation, cu_seqlens_m, A_idx, dynamic_scheduler)


def gemm_act_ref(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (M, total_K) if varlen_k or (whatever, K) if gather_A
    B: Tensor,  # (K, N) or (L, K, N) or (total_K, N) if varlen_k
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: Activation = None,
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
            A, B, bias=bias, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx, concat_layout=concat_layout
        )
    else:
        preact = gemm_add_ref(
            A, B, C, bias=bias, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx, concat_layout=concat_layout
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
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m; or (M, 2*N) for dgated
    activation: Activation = None,
    dx_out: Optional[
        Tensor
    ] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m; double for gated
    postact_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m (dgated only)
    colvec_reduce: bool = False,  # dgated only
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
):
    """GEMM with activation (or gated activation) gradient and optional output tensors."""
    is_dgated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
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
    # makes preact contribution zero. dact at preact=0 is also 0 for every
    # supported activation, so we can zero outputs and skip the kernel.
    if dx_out.numel() == 0 or A.numel() == 0:
        _empty_k_matmul_into(dx_out)
        _empty_k_matmul_into(postact_out)
        results = [dx_out, postact_out]
        if colvec_reduce:
            colvec_shape = (*out_shape[:-1],)
            results.append(torch.zeros(colvec_shape, dtype=torch.float32, device=A.device))
        return tuple(results)
    if is_dgated:
        colvec_reduce_final = gemm_dgated_out(
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
        )
        results = [dx_out, postact_out]
        if colvec_reduce:
            results.append(colvec_reduce_final)
        return tuple(results)
    else:
        gemm_dact_out(
            A,
            B,
            PreAct,
            dx_out,
            postact_out,
            activation,
            cu_seqlens_m,
            A_idx,
            dynamic_scheduler,
            tuned,
        )
        results = [dx_out, postact_out]
        return tuple(results)


gemm_dgated = gemm_dact


def gemm_dact_out(
    A: Tensor,  # (M, K) or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    dx_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    activation: ActActivation = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    tuned: bool = True,
) -> None:
    """GEMM with activation gradient and pre-allocated output tensors."""
    fn = gemm_dact_tuned if tuned else partial(gemm_dact_tuned.fn, config=None)
    fn(A, B, PreAct, dx_out, postact_out, activation, cu_seqlens_m, A_idx, dynamic_scheduler)


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
) -> None:
    """GEMM with guaranteed symmetric output."""
    alpha = alpha_tensor if alpha_tensor is not None else alpha
    beta = beta_tensor if beta_tensor is not None else beta
    if A.ndim == 2:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (M, K) or (L, M, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, M, K)
    if C is not None and C.ndim == 2:
        C = C.unsqueeze(0)  # (1, M, M)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    else:
        out = out
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    )
    sm = get_device_capacity(A.device)[0]
    # We want square tile per cluster
    tile_m, tile_n, cluster_m, pingpong = _symmetric_gemm_config(sm)
    gemm_symmetric_dispatch(
        A,
        B,
        out if out is not None else None,
        C if C is not None else None,
        tile_count_semaphore,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=1,
        pingpong=pingpong,
        persistent=True,
        is_dynamic_persistent=sm >= 10,
        max_swizzle_size=8,
        alpha=alpha,
        beta=beta,
    )


def gemm_symmetric(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, M) or (L, K, M)
    C: Optional[Tensor] = None,  # (M, M) or (L, M, M)
    out: Optional[Tensor] = None,  # (M, M) or (L, M, M)
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with symmetric output."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    # Determine output shape based on gather_A
    if A.ndim == 2:
        out_shape = (A.shape[0], B.shape[-1])
    else:
        out_shape = (A.shape[0], A.shape[-2], B.shape[-1])
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)

    alpha_tensor = alpha if not isinstance(alpha, float) else None
    alpha_val = alpha if isinstance(alpha, float) else 1.0
    beta_tensor = beta if not isinstance(beta, float) else None
    beta_val = beta if isinstance(beta, float) else 1.0

    # Empty-input fast path: out = alpha * A@A.T + beta * C reduces to beta * C
    # when K=0 (or just zeros / empty for M=0).
    if out.numel() == 0:
        return out
    if A.numel() == 0:
        _empty_k_matmul_into(out, C=C, beta=beta)
        return out

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
    )
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs("gated")],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_gated_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    # (M, N) or (L, M, N) or (total_M, N) if varlen_m - None if not storing preact
    preact_out: Optional[Tensor],
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2) or (total_M, N//2) if varlen_m
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    bias: Optional[Tensor] = None,  # (N,) or (L, N)
    activation: GatedActivation = "swiglu",
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
    concat_layout: tuple | None = None,  # tensors whose non-contiguous dim is concat [gate; up]
) -> None:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)  # (L, N)
    if concat_layout and "bias" in concat_layout:
        if bias is not None and bias.dtype.itemsize >= 4:
            bias_key = "mColVecBroadcast" if config.swap_ab else "mRowVecBroadcast"
            concat_layout = tuple(bias_key if k == "bias" else k for k in concat_layout)
        else:
            concat_layout = tuple(k for k in concat_layout if k != "bias")
            if bias is not None:
                bias = _concat_interleave_bias(bias)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_act_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias if not config.swap_ab else None,
        colvec_bias=bias if config.swap_ab else None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
        concat_layout=concat_layout,
    )


def prune_invalid_gemm_dgated_configs(configs, named_args: dict, **kwargs):
    kwargs = named_args | kwargs
    # if there's colvec_scale or colvec_reduce, don't swap_AB
    if kwargs.get("colvec_scale", None) is not None or kwargs.get("colvec_reduce", False):
        configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    return prune_invalid_gemm_configs(configs, named_args, **kwargs)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs("dgated")],
    key=["activation", "colvec_reduce", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_dgated_configs},
)
def gemm_dgated_tuned(
    # (M, K) or or (L, M, K) or (total_M, K) if varlen_m or (whatever, K) if gather_A with varlen_m
    A: Tensor,
    B: Tensor,  # (K, N) or (L, K, N)
    PreAct: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    dx_out: Tensor,  # (M, 2*N) or (L, M, 2*N) or (total_M, 2*N) if varlen_m
    postact_out: Tensor,  # (M, N) or (L, M, N) or (total_M, N) if varlen_m
    colvec_scale: Optional[Tensor] = None,  # (M,) or (L, M) or (total_M,) if varlen_m
    activation: GatedActivation = "swiglu",
    # whether to do colvec reduction, returning (M,) or (L, M) or (total_M) if varlen_m
    colvec_reduce: bool = False,
    cu_seqlens_m: Optional[Tensor] = None,  # (L+1), int32
    A_idx: Optional[Tensor] = None,  # (total_M,) if gather_A with varlen_m
    dynamic_scheduler: bool = True,
    config: Optional[GemmConfig] = None,
) -> Optional[Tensor]:
    if config is None:
        config = default_config(A.device)
    varlen_m = cu_seqlens_m is not None
    if varlen_m:
        assert not config.swap_ab, "Variable-length sequences not supported with swap_ab"
    og_ndim_2 = A.ndim == 2 and not varlen_m
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)  # (1, M, K)
    B = B.mT  # (N, K) or (L, N, K)
    if B.ndim == 2:
        B = B.unsqueeze(0)  # (1, N, K)
    if PreAct.ndim == 2 and not varlen_m:
        PreAct = PreAct.unsqueeze(0)  # (1, M, 2*N)
    if dx_out.ndim == 2 and not varlen_m:
        D = dx_out.unsqueeze(0)
    else:
        D = dx_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if colvec_scale is not None and colvec_scale.ndim == 1 and not varlen_m:
        colvec_scale = colvec_scale.unsqueeze(0)  # (L, N)
    if colvec_scale is not None:
        assert not config.swap_ab, "colvec_scale not supported with swap_ab"
    if colvec_reduce:
        tile_n = config.tile_n
        shape_n = (B.shape[-2] + tile_n - 1) // tile_n
        if varlen_m:
            total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
            colvec_shape = (total_m, shape_n)
        else:
            colvec_shape = (A.shape[0], A.shape[-2], shape_n)
        colvec_reduce_partial = torch.empty(colvec_shape, dtype=torch.float32, device=A.device)
    else:
        colvec_reduce_partial = None
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_dact_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        PreAct if not config.swap_ab else PreAct.mT,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        colvec_scale=colvec_scale,
        colvec_reduce=colvec_reduce_partial,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        use_tma_gather=config.use_tma_gather,
    )
    if colvec_reduce:
        colvec_reduce_final = colvec_reduce_partial.sum(dim=-1)
        if og_ndim_2:
            colvec_reduce_final = colvec_reduce_final.squeeze(0)
    else:
        colvec_reduce_final = None
    return colvec_reduce_final


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
) -> None:
    """GEMM with gated activation and pre-allocated output tensors."""
    fn = gemm_gated_tuned if tuned else partial(gemm_gated_tuned.fn, config=None)
    fn(
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
        concat_layout=tuple(concat_layout.split(",")) if concat_layout else None,
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
) -> Tensor:
    """GEMM with gated activation gradient and pre-allocated output tensors."""
    fn = gemm_dgated_tuned if tuned else partial(gemm_dgated_tuned.fn, config=None)
    result = fn(
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
    )
    if result is None:  # Have to return a tensor, not None, to make torch compile happy
        return torch.empty(0, device=A.device, dtype=torch.float32)
    return result


def _precompile_default_config(autotuned_fn, *args, **kwargs):
    """Compile the default config in COMPILE_ONLY mode.

    Checks COMPILE_ONLY flag and SymInt guard, then calls the unwrapped function with
    config=None (which selects the default config), triggering compilation (exports .o)
    without benchmarking or kernel launch.
    Tests use tuned=False which also selects the default config, so this is sufficient.
    """
    from .cache_utils import COMPILE_ONLY

    A = args[0] if args else kwargs.get("A")
    if not COMPILE_ONLY or A is None or isinstance(A.shape[0], torch.SymInt):
        return
    try:
        autotuned_fn.fn(*args, config=None, **kwargs)
    except Exception:
        pass


## ── gemm_rms ────────────────────────────────────────────────────────────────


def _prune_gemm_rms_configs(configs, named_args: dict, **kwargs):
    """ColVecReduce requires no swap_ab."""
    configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    return prune_invalid_gemm_configs(configs, named_args | kwargs)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["dynamic_scheduler"],
    prune_configs_by={"early_config_prune": _prune_gemm_rms_configs},
)
def _gemm_rms_tuned(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    out: Tensor,  # (M, N) or (L, M, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    norm_weight: Optional[Tensor] = None,  # (N,) or (L, N)
    premult_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) — pre-norm_weight snapshot
    eps: float = 1e-6,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> Tensor:
    if config is None:
        config = default_config(A.device)
    og_ndim_2 = A.ndim == 2
    N = B.shape[-1]
    if A.ndim == 2:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    if C is not None and C.ndim == 2:
        C = C.unsqueeze(0)
    if norm_weight is not None and norm_weight.ndim == 1:
        norm_weight = norm_weight.unsqueeze(0)  # (L, N)
    if premult_out is not None and premult_out.ndim == 2:
        premult_out = premult_out.unsqueeze(0)
    # Allocate partial reduction buffer
    tile_n = config.tile_n
    n_tiles = (N + tile_n - 1) // tile_n
    colvec_reduce = torch.empty(
        (A.shape[0], A.shape[1], n_tiles), dtype=torch.float32, device=A.device
    )
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_sq_reduce_dispatch(
        A,
        B,
        out,
        C,
        colvec_reduce,
        tile_count_semaphore,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        rowvec=norm_weight,
        aux_out=premult_out,
    )
    # Final reduction: rstd = rsqrt(sum(partials) / N + eps)
    scale = 1.0 / N
    flat_reduce = colvec_reduce.reshape(-1, n_tiles)
    rstd_flat = rms_final_reduce(flat_reduce, scale=scale, eps=eps)
    rstd = rstd_flat.reshape(A.shape[:-1])
    if og_ndim_2:
        rstd = rstd.squeeze(0)
    return rstd


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
) -> Tensor:
    """GEMM + RMS + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    """
    fn = _gemm_rms_tuned if tuned else partial(_gemm_rms_tuned.fn, config=None)
    return fn(
        A,
        B,
        out,
        C=C,
        norm_weight=norm_weight,
        premult_out=premult_out,
        eps=eps,
        dynamic_scheduler=dynamic_scheduler,
    )


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
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    norm_weight: Optional[Tensor] = None,  # (N,) or (L, N)
    out: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    out_dtype: Optional[torch.dtype] = None,
    premult_out: Optional[Tensor] = None,  # (M, N) or (L, M, N) — pre-norm_weight snapshot
    eps: float = 1e-6,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tuple[Tensor, Tensor]:
    """GEMM + RMS statistics + optional rowvec scaling.

    D_raw = A @ B (+ C), rstd = rsqrt(mean(D_raw^2) + eps), D_out = D_raw * norm_weight.
    If premult_out is provided, D_raw (the pre-norm_weight value) is also written to it.
    Returns (D_out, rstd).
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
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
    )
    return out, rstd


## ── gemm_norm_act ─────────────────────────────────────────────────────────────


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_norm_act_tuned(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N) — None if not storing preact
    postact_out: Tensor,  # (M, N) or (L, M, N)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    rstd: Optional[Tensor] = None,  # (M,) or (L, M)
    activation: ActActivation = None,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    if A.ndim == 2:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if C is not None and C.ndim == 2:
        C = C.unsqueeze(0)
    if preact_out is not None and preact_out.ndim == 2:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if rstd is not None and rstd.ndim == 1:
        rstd = rstd.unsqueeze(0)  # (L, M)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_norm_act_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        colvec=rstd if not config.swap_ab else None,
        rowvec=rstd if config.swap_ab else None,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs("gated")],
    key=["activation", "dynamic_scheduler"],
    prune_configs_by={"early_config_prune": prune_invalid_gemm_configs},
)
def gemm_norm_gated_tuned(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
    preact_out: Optional[Tensor],  # (M, N) or (L, M, N)
    postact_out: Tensor,  # (M, N//2) or (L, M, N//2)
    C: Optional[Tensor] = None,  # (M, N) or (L, M, N)
    rstd: Optional[Tensor] = None,  # (M,) or (L, M)
    activation: GatedActivation = "swiglu",
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = default_config(A.device)
    if A.ndim == 2:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if C is not None and C.ndim == 2:
        C = C.unsqueeze(0)
    if preact_out is not None and preact_out.ndim == 2:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if rstd is not None and rstd.ndim == 1:
        rstd = rstd.unsqueeze(0)  # (L, M)
    dynamic_scheduler = dynamic_scheduler or config.is_dynamic_persistent
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if dynamic_scheduler and get_device_capacity(A.device)[0] == 9
        else None
    )
    gemm_norm_act_dispatch(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        tile_K=config.tile_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=dynamic_scheduler,
        max_swizzle_size=config.max_swizzle_size,
        colvec=rstd if not config.swap_ab else None,
        rowvec=rstd if config.swap_ab else None,
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
) -> None:
    fn = gemm_norm_act_tuned if tuned else partial(gemm_norm_act_tuned.fn, config=None)
    fn(A, B, preact_out, postact_out, C, rstd, activation, dynamic_scheduler)


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
) -> None:
    fn = gemm_norm_gated_tuned if tuned else partial(gemm_norm_gated_tuned.fn, config=None)
    fn(A, B, preact_out, postact_out, C, rstd, activation, dynamic_scheduler)


def gemm_norm_act(
    A: Tensor,  # (M, K) or (L, M, K)
    B: Tensor,  # (K, N) or (L, K, N)
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
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM + normalize + activation: PostAct = act((A @ B + C) * rstd).

    rstd is a column vector (M,).
    Returns (preact, postact) where preact is the normalized value before activation.
    """
    is_gated = activation in gated_to_pytorch_fn_map
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
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
    if is_gated:
        gemm_norm_gated_out(
            A,
            B,
            preact_out,
            postact_out,
            C,
            rstd,
            activation,
            dynamic_scheduler,
            tuned,
        )
    else:
        gemm_norm_act_out(
            A,
            B,
            preact_out,
            postact_out,
            C,
            rstd,
            activation,
            dynamic_scheduler,
            tuned,
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
