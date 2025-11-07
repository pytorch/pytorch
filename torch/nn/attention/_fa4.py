# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library


__all__ = [
    "register_flash_attention_fa4",
    "flash_attention_fa4_status",
]


@dataclass
class _FA4FlashAttentionState:
    module_path: str = "flash_attn.cute.interface"
    module: ModuleType | None = None
    registered: bool = False
    last_error: str | None = None


_FA4_STATE = _FA4FlashAttentionState()
_FA4_LIBRARY: Library | None = None


def _flash_attention_forward_fallback(*args, **kwargs):
    return torch.ops.aten._flash_attention_forward.default(*args, **kwargs)


def _flash_attention_backward_fallback(*args, **kwargs):
    return torch.ops.aten._flash_attention_backward.default(*args, **kwargs)


def _sdpa_forward_fallback(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_flash_attention.default(*args, **kwargs)


def _sdpa_backward_fallback(*args, **kwargs):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        *args, **kwargs
    )


def register_flash_attention_fa4(
    module_path: str = "flash_attn.cute.interface",
) -> None:
    module = _fa4_import_module(module_path)
    _FA4_STATE.module_path = module_path
    _FA4_STATE.module = module
    _FA4_STATE.last_error = None
    if _FA4_STATE.registered:
        return
    _fa4_register_kernels()
    _FA4_STATE.registered = True


def flash_attention_fa4_status() -> dict[str, Any]:
    return {
        "registered": _FA4_STATE.registered,
        "module_loaded": _FA4_STATE.module is not None,
        "module_path": _FA4_STATE.module_path,
        "last_error": _FA4_STATE.last_error,
    }


def _fa4_import_module(module_path: str) -> ModuleType:
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        message = f"Module '{module_path}' not found"
        _FA4_STATE.last_error = message
        raise ImportError(message)
    module = importlib.import_module(module_path)
    if not hasattr(module, "_flash_attn_fwd") or not hasattr(module, "_flash_attn_bwd"):
        message = f"Module '{module_path}' does not expose FA4 kernels"
        _FA4_STATE.last_error = message
        raise RuntimeError(message)
    return module


def _fa4_register_kernels() -> None:
    global _FA4_LIBRARY
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl("_flash_attention_forward", _fa4_flash_attention_forward_impl, "CUDA")
    lib.impl("_flash_attention_backward", _fa4_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa4_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa4_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    _FA4_LIBRARY = lib


def _fa4_forward_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
) -> bool:
    if dropout_p != 0.0:
        return False
    if return_debug_mask:
        return False
    if alibi_slopes is not None:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if seqused_k is not None:
        if seqused_k.dtype != torch.int32:
            return False
        if not seqused_k.is_cuda:
            return False
    if cum_seq_q is None and query.dim() != 4:
        return False
    if cum_seq_q is not None and query.dim() != 3:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(query.device)
    return major in (9, 10)


def _fa4_backward_supported(
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
) -> bool:
    if dropout_p != 0.0:
        return False
    if window_size_left is not None or window_size_right is not None:
        return False
    tensors = (grad_out, query, key, value, out, logsumexp)
    if not all(t.is_cuda for t in tensors):
        return False
    if len({t.device for t in tensors}) != 1:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if logsumexp.dtype != torch.float32:
        return False
    if cum_seq_q is None and query.dim() != 4:
        return False
    if cum_seq_q is not None and query.dim() != 3:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(query.device)
    return major in (9, 10)


def _fa4_prepare_dense(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous()


def _fa4_restore_dense(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).contiguous()


def _fa4_as_int32(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dtype == torch.int32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.int32).contiguous()


def _fa4_zero_seed(device: torch.device) -> torch.Tensor:
    return torch.zeros((2,), dtype=torch.uint64, device=device)


def _fa4_zero_offset(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.uint64, device=device)


def _fa4_empty_debug_mask(query: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, dtype=query.dtype, device=query.device)


def _fa4_to_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _fa4_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _FA4_STATE.module
    if module is None:
        raise RuntimeError("FA4 module not loaded")
    dense = cum_seq_q is None
    if dense:
        q = _fa4_prepare_dense(query)
        k = _fa4_prepare_dense(key)
        v = _fa4_prepare_dense(value)
    else:
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
    kwargs: dict[str, Any] = {
        "softmax_scale": scale,
        "causal": is_causal,
        "window_size_left": _fa4_to_optional_int(window_size_left),
        "window_size_right": _fa4_to_optional_int(window_size_right),
        "return_lse": True,
        "cu_seqlens_q": _fa4_as_int32(cum_seq_q),
        "cu_seqlens_k": _fa4_as_int32(cum_seq_k),
        "seqused_k": seqused_k.contiguous() if seqused_k is not None else None,
    }
    out, lse = module._flash_attn_fwd(q, k, v, **kwargs)
    if dense:
        out = _fa4_restore_dense(out)
    return out, lse.contiguous()


def _fa4_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor | None,
    cum_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _FA4_STATE.module
    if module is None:
        raise RuntimeError("FA4 module not loaded")
    dense = cum_seq_q is None
    if dense:
        q = _fa4_prepare_dense(query)
        k = _fa4_prepare_dense(key)
        v = _fa4_prepare_dense(value)
        o = _fa4_prepare_dense(out)
        do = _fa4_prepare_dense(grad_out)
    else:
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        o = out.contiguous()
        do = grad_out.contiguous()
    dq, dk, dv = module._flash_attn_bwd(
        q,
        k,
        v,
        o,
        do,
        logsumexp.contiguous(),
        softmax_scale=scale,
        causal=is_causal,
        cu_seqlens_q=_fa4_as_int32(cum_seq_q),
        cu_seqlens_k=_fa4_as_int32(cum_seq_k),
    )
    if dense:
        dq = _fa4_restore_dense(dq)
        dk = _fa4_restore_dense(dk)
        dv = _fa4_restore_dense(dv)
    return dq, dk, dv


def _fa4_flash_attention_forward_impl(
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
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
):
    if not _FA4_STATE.registered:
        return _flash_attention_forward_fallback(
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
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            seqused_k=seqused_k,
            alibi_slopes=alibi_slopes,
        )
    if not _fa4_forward_supported(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
    ):
        return _flash_attention_forward_fallback(
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
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            seqused_k=seqused_k,
            alibi_slopes=alibi_slopes,
        )
    out, lse = _fa4_run_forward(
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
    )
    rng_state = _fa4_zero_seed(query.device)
    philox_offset = _fa4_zero_offset(query.device)
    debug_mask = _fa4_empty_debug_mask(query)
    return out, lse, rng_state, philox_offset, debug_mask


def _fa4_flash_attention_backward_impl(
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
    if not _FA4_STATE.registered:
        return _flash_attention_backward_fallback(
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
            dropout_p,
            is_causal,
            rng_state,
            unused,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
    if not _fa4_backward_supported(
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
    ):
        return _flash_attention_backward_fallback(
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
            dropout_p,
            is_causal,
            rng_state,
            unused,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
    return _fa4_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        scale,
        is_causal,
    )


def _fa4_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    if not _FA4_STATE.registered:
        return _sdpa_forward_fallback(
            query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale
        )
    if not _fa4_forward_supported(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
    ):
        return _sdpa_forward_fallback(
            query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale
        )
    out, lse = _fa4_run_forward(
        query,
        key,
        value,
        None,
        None,
        scale,
        is_causal,
        None,
        None,
        None,
    )
    rng_state = _fa4_zero_seed(query.device)
    philox_offset = _fa4_zero_offset(query.device)
    debug_mask = _fa4_empty_debug_mask(query)
    max_q = query.size(2)
    max_k = key.size(2)
    return (
        out,
        lse,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        philox_offset,
        debug_mask,
    )


def _fa4_scaled_dot_product_flash_attention_backward_impl(
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
    if not _FA4_STATE.registered:
        return _sdpa_backward_fallback(
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
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )
    if not _fa4_backward_supported(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        None,
        None,
        None,
    ):
        return _sdpa_backward_fallback(
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
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )
    return _fa4_run_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        None,
        None,
        scale,
        is_causal,
    )
