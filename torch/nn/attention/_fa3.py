"""
PROTOTYPE!
Flash Attention 3 implementation.
For fp8: only supports forward pass right now.
@TODO: add forward and backward support for fp16/bf16.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing_extensions import TypeVarTuple, Unpack

import torch
from torch.library import Library

from . import _registry


__all__ = [
    "register_flash_attention_fa3",
]


_FA3_CUDA_FWD: Callable | None = None  # Cache for torch.ops.flash_attn_3.fwd


@dataclass
class _FA3Handle:
    library: Library | None

    def remove(self) -> None:
        self.library = None
        # Clear the C++ flag
        torch._C._set_sdp_use_fa3(False)


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa3(
    module_path: str = "flash_attn_interface",
) -> _FA3Handle:
    """
    Register FA3 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA3 implementation.
    """
    _fa3_import_module(module_path)

    # Expose FA3 registration status to C++
    torch._C._set_sdp_use_fa3(True)

    return _FA3Handle(_fa3_register_kernels())


def _fa3_import_module(module_path: str) -> None:
    importlib.import_module(module_path)
    if not hasattr(torch.ops, "flash_attn_3") or not hasattr(
        torch.ops.flash_attn_3, "fwd"
    ):
        raise RuntimeError(f"Module '{module_path}' does not expose FA3 kernels")
    global _FA3_CUDA_FWD
    _FA3_CUDA_FWD = torch.ops.flash_attn_3.fwd


def _fa3_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl("_flash_attention_forward", _fa3_flash_attention_forward_impl, "CUDA")
    lib.impl("_flash_attention_backward", _fa3_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa3_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa3_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa3_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    cum_seq_q: torch.Tensor | None,
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype != torch.float8_e4m3fn:
        return "query dtype must be float8_e4m3fn"
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) != 9:
        return "FA3 requires compute capability 9.0"
    return None


def _fa3_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    error = _fa3_common_support_error(
        query,
        (query, key, value),
        cum_seq_q,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: Unpack[Ts]) -> tuple[Unpack[Ts]]:
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _maybe_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
    """Ensure tensor is contiguous in the last dimension."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _fa3_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the FA3 forward pass by calling the C++ kernel directly.
    """
    if _FA3_CUDA_FWD is None:
        raise RuntimeError("FA3 not registered")
    # Ensure contiguous in the last dimension
    q = _maybe_contiguous(query)
    k = _maybe_contiguous(key)
    v = value.contiguous() if value.stride(-1) != 1 and value.stride(-3) != 1 else value

    cu_seqlens_q = _maybe_contiguous(cu_seq_q)
    cu_seqlens_k = _maybe_contiguous(cu_seq_k)
    seqused_k = _maybe_contiguous(seqused_k)

    out, softmax_lse, out_accum, softmax_lse_accum = _FA3_CUDA_FWD(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        None,  # qv
        out,  # out_ (pre-allocated output)
        cu_seqlens_q,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,  # seqused_k
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        None,  # page_table,
        None,  # kv_batch_idx,
        None,  # leftpad_k,
        None,  # rotary_cos,
        None,  # rotary_sin,
        None,  # seqlens_rotary,
        None,  # q_descale,
        None,  # k_descale,
        None,  # v_descale,
        scale,  # softmax_scale,
        is_causal,  # causal,
        window_size_left if window_size_left is not None else -1,  # window_size_left
        window_size_right if window_size_right is not None else -1,  # window_size_right
        0,  # attention_chunk,
        0.0,  # softcap,
        True,  # rotary_interleaved,
        None,  # scheduler_metadata,
        1,  # num_splits,
        None,  # pack_gqa,
        0,  # sm_margin,
    )
    return out, softmax_lse.contiguous()


def _fa3_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    error = _fa3_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
    )
    if error is not None:
        raise RuntimeError(f"FA3 flash_attention forward unsupported: {error}")
    out, lse = _fa3_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
        out,
    )
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
    debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return out, lse, rng_state, philox_offset, debug_mask


def _fa3_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    rng_state: torch.Tensor,
    unused: torch.Tensor,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
):
    raise RuntimeError(
        "FA3 does not support backward pass. Either:\n"
        "  1. Use torch.no_grad() for inference\n"
        "  2. Unregister FA3 before training:restore_flash_attention_impl"
    )


def _fa3_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    error = _fa3_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA3 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    # Pre-allocate output with query's strides (BHSD layout), then create
    # a BSHD view for the kernel. This ensures the returned output has
    # the same memory layout as the input query.
    out_bhsd = torch.empty_like(query, dtype=torch.bfloat16)
    out_bshd = out_bhsd.transpose(1, 2)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    _, lse, rng_state, philox_offset, debug_mask = _fa3_flash_attention_forward_impl(
        q,
        k,
        v,
        None,
        None,
        max_q_flash,
        max_k_flash,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        out=out_bshd,
    )
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out_bhsd,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa3_scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    raise RuntimeError(
        "FA3 does not support backward pass. Either:\n"
        "  1. Use torch.no_grad() for inference\n"
        "  2. Unregister FA3 before training:restore_flash_attention_impl"
    )


_registry.register_flash_attention_impl("FA3", register_fn=register_flash_attention_fa3)
