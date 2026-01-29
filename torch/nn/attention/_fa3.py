"""
PROTOTYPE!
Flash Attention 3 implementation.
For fp8: only supports forward pass right now.
For fp16/bf16: supports forward and backward pass.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
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
_FA3_CUDA_BWD: Callable | None = None  # Cache for torch.ops.flash_attn_3.bwd


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
    if not hasattr(torch.ops, "flash_attn_3"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA3 kernels")
    if not hasattr(torch.ops.flash_attn_3, "fwd"):
        raise RuntimeError(
            f"Module '{module_path}' does not expose FA3 forward kernels"
        )
    if not hasattr(torch.ops.flash_attn_3, "bwd"):
        raise RuntimeError(
            f"Module '{module_path}' does not expose FA3 backward kernels"
        )
    global _FA3_CUDA_FWD, _FA3_CUDA_BWD
    _FA3_CUDA_FWD = torch.ops.flash_attn_3.fwd
    _FA3_CUDA_BWD = torch.ops.flash_attn_3.bwd


def _fa3_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl(
        "_flash_attention_forward.quantized", _fa3_flash_attention_forward_impl, "CUDA"
    )
    lib.impl(
        "_scaled_dot_product_flash_attention.quantized",
        _fa3_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_flash_attention_forward", _fa3_flash_attention_forward_impl_default, "CUDA"
    )
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa3_scaled_dot_product_flash_attention_forward_impl_default,
        "CUDA",
    )

    lib.impl("_flash_attention_backward", _fa3_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa3_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa3_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"

    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype == torch.float8_e4m3fn and (
        q_descale is None or k_descale is None or v_descale is None
    ):
        warnings.warn(
            "When using SDPA with fp8, descale tensor should always be used"
            " for accurate dequantization. Please use "
            "_scaled_dot_product_attention_quantized and "
            "provide the descale tensors.",
            UserWarning,
        )
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
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> str | None:
    if return_debug_mask:
        return "return_debug_mask must be False"
    if alibi_slopes is not None:
        return "alibi_slopes not supported"
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return "seqused_k must be int32"
        if not seqused_k.is_cuda:
            return "seqused_k must be CUDA"
    supported_dtypes = (torch.float8_e4m3fn, torch.float16, torch.bfloat16)
    if not all(t.dtype in supported_dtypes for t in {query, key, value}):
        return f"inputs must be one of {supported_dtypes}"
    if len({t.dtype for t in {query, key, value}}) != 1:
        return "all inputs must have the same dtype"
    error = _fa3_common_support_error(
        query,
        (query, key, value),
        dropout_p,
        cum_seq_q,
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa3_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    window_size_left: int | None,
    window_size_right: int | None,
) -> str | None:
    # FA3 backward ONLY supports fp16/bf16, NOT fp8
    if query.dtype == torch.float8_e4m3fn:
        return (
            "FA3 backward does not support fp8 - use inference only (torch.no_grad())"
        )
    if logsumexp.dtype != torch.float32:
        return "logsumexp dtype must be float32"
    supported_dtypes = (torch.float16, torch.bfloat16)
    if not all(t.dtype in supported_dtypes for t in {grad_out, query, key, value, out}):
        return f"inputs must be one of {supported_dtypes}"
    if len({t.dtype for t in {grad_out, query, key, value, out}}) != 1:
        return "all inputs must have the same dtype"
    error = _fa3_common_support_error(
        query,
        (grad_out, query, key, value, out, logsumexp),
        dropout_p,
        cum_seq_q,
        None,
        None,
        None,
    )
    if error is not None:
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
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the FA3 forward pass by calling the C++ kernel directly.

    When k_cache and v_cache are provided, the key and value tensors are treated
    as new keys/values to append to the cache. The cache tensors become the main
    k and v inputs to the kernel.
    """
    if _FA3_CUDA_FWD is None:
        raise RuntimeError("FA3 not registered")

    use_kv_cache = k_cache is not None and v_cache is not None

    if use_kv_cache:
        if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
            raise RuntimeError("k_cache and v_cache must be contiguous in the last dim")
        k = k_cache
        v = v_cache
        k_new = _maybe_contiguous(key)
        v_new = _maybe_contiguous(value)
        # cache_seqlens overrides seqused_k for the cache lookup
        effective_seqused_k = (
            _maybe_contiguous(cache_seqlens) if cache_seqlens is not None else seqused_k
        )
        cu_seqlens_k = None
        cu_seqlens_k_new = _maybe_contiguous(cu_seq_k)
    else:
        k = _maybe_contiguous(key)
        v = (
            value.contiguous()
            if value.dtype == torch.float8_e4m3fn
            and value.stride(-1) != 1
            and value.stride(-3) != 1
            else _maybe_contiguous(value)
        )
        k_new = None
        v_new = None
        effective_seqused_k = _maybe_contiguous(seqused_k)
        cu_seqlens_k = _maybe_contiguous(cu_seq_k)
        cu_seqlens_k_new = None

    # Ensure contiguous in the last dimension
    q = _maybe_contiguous(query)
    cu_seqlens_q = _maybe_contiguous(cu_seq_q)

    page_table, kv_batch_idx = [
        _maybe_contiguous(x) for x in (page_table, cache_batch_idx)
    ]

    out, softmax_lse, out_accum, softmax_lse_accum = _FA3_CUDA_FWD(
        q,
        k,
        v,
        k_new,  # k_new (new keys for kv cache)
        v_new,  # v_new (new values for kv cache)
        None,  # qv
        out,  # out_ (pre-allocated output)
        cu_seqlens_q,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k
        cu_seqlens_k_new,  # cu_seqlens_k_new
        None,  # seqused_q
        effective_seqused_k,  # seqused_k
        max_seqlen_q,  # max_seqlen_q
        max_seqlen_k,  # max_seqlen_k
        page_table,  # page_table
        kv_batch_idx,  # kv_batch_idx
        None,  # leftpad_k,
        None,  # rotary_cos,
        None,  # rotary_sin,
        None,  # seqlens_rotary,
        q_descale,  # q_descale,
        k_descale,  # k_descale,
        v_descale,  # v_descale,
        scale,  # softmax_scale,
        is_causal,  # causal,
        window_size_left if window_size_left is not None else -1,  # window_size_left
        window_size_right if window_size_right is not None else -1,  # window_size_right
        0,  # attention_chunk,
        0.0,  # softcap,
        True,  # rotary_interleaved,
        None,  # scheduler_metadata,
        1 if torch.are_deterministic_algorithms_enabled() else 0,  # num_splits,
        None,  # pack_gqa,
        torch._C._get_sm_carveout_experimental() or 0,  # sm_margin,
    )
    return out, softmax_lse.contiguous()


def _fa3_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FA3_CUDA_BWD is None:
        raise RuntimeError("FA3 not registered")

    # Ensure contiguous
    dout = _maybe_contiguous(grad_out)
    q = query.contiguous() if query.stride(-1) != 1 else query
    k = key.contiguous() if key.stride(-1) != 1 else key
    v = value.contiguous() if value.stride(-1) != 1 else value
    o = _maybe_contiguous(out)
    lse = _maybe_contiguous(logsumexp)

    # Pre-allocate gradient tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _FA3_CUDA_BWD(
        dout,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        cu_seq_q,
        cu_seq_k,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        0.0,
        deterministic,
        torch._C._get_sm_carveout_experimental() or 0,
    )
    return dq, dk, dv


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
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
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
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        raise RuntimeError(f"FA3 flash_attention forward unsupported: {error}")
    out, lse = _fa3_run_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left,
        window_size_right,
        seqused_k,
        out,
        q_descale,
        k_descale,
        v_descale,
        k_cache,
        v_cache,
        cache_seqlens,
        cache_batch_idx,
        page_table,
    )
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
    debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)
    return out, lse, rng_state, philox_offset, debug_mask


def _fa3_flash_attention_forward_impl_default(
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
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
):
    return _fa3_flash_attention_forward_impl(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        None,
        None,
        None,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        page_table=page_table,
    )


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
    window_size_left: int | None = None,
    window_size_right: int | None = None,
):
    """FA3 implementation of _flash_attention_backward."""
    error = _fa3_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
        window_size_left,
        window_size_right,
    )

    if error is not None:
        raise RuntimeError(f"FA3 flash_attention backward unsupported: {error}")

    deterministic = torch.are_deterministic_algorithms_enabled()

    dq, dk, dv = _fa3_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        scale,
        is_causal,
        window_size_left if window_size_left is not None else -1,
        window_size_right if window_size_right is not None else -1,
        deterministic,
    )
    return dq, dk, dv


def _fa3_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
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
        q_descale,
        k_descale,
        v_descale,
    )
    if error is not None:
        raise RuntimeError(f"FA3 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    # Pre-allocate output with query's strides (BHSD layout), then create
    # a BSHD view for the kernel. This ensures the returned output has
    # the same memory layout as the input query.
    out_dtype = torch.bfloat16 if query.dtype == torch.float8_e4m3fn else query.dtype
    out_bhsd = torch.empty_like(query, dtype=out_dtype)
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
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
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


def _fa3_scaled_dot_product_flash_attention_forward_impl_default(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    return _fa3_scaled_dot_product_flash_attention_forward_impl(
        query,
        key,
        value,
        None,
        None,
        None,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
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
    """FA3 implementation of _scaled_dot_product_flash_attention_backward."""
    error = _fa3_backward_support_error(
        grad_out, query, key, value, out, logsumexp, dropout_p, None, None, None
    )
    if error is not None:
        raise RuntimeError(f"FA3 SDPA backward unsupported: {error}")

    # SDPA uses BHSD layout, FA3 uses BSHD - transpose
    grad_out_t, q_t, k_t, v_t, out_t = _transpose_dense(
        grad_out, query, key, value, out
    )

    dq, dk, dv = _fa3_flash_attention_backward_impl(
        grad_out_t,
        q_t,
        k_t,
        v_t,
        out_t,
        logsumexp,
        None,  # cum_seq_q (dense attention)
        None,  # cum_seq_k
        max_q,  # max_seqlen_q
        max_k,  # max_seqlen_k
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )

    # Transpose gradients back to BHSD layout
    dq_out, dk_out, dv_out = _transpose_dense(dq, dk, dv)
    return dq_out, dk_out, dv_out


_registry.register_flash_attention_impl("FA3", register_fn=register_flash_attention_fa3)
