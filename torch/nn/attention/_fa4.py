"""UBER PROTOTYPE!!!"""
# mypy: allow-untyped-defs

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import cache, partial
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVarTuple

from . import _registry
from ._utils import _empty_with_matching_layout, _FP8_DTYPES


if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library


__all__ = [
    "register_flash_attention_fa4",
]


_FA4_MODULE_PATH: str | None = None


@dataclass
class _FA4Handle:
    library: Library | None

    def remove(self) -> None:
        self.library = None
        torch._C._set_sdp_use_fa4(False)


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_flash_attention_fa4(
    module_path: str = "flash_attn.cute.interface",
) -> _FA4Handle:
    """
    Register FA4 flash attention kernels with the PyTorch dispatcher.

    Args:
        module_path: Python module path to the FA4 implementation.
    """
    global _FA4_MODULE_PATH
    _ = _fa4_import_module(module_path)
    _FA4_MODULE_PATH = module_path
    handle = _FA4Handle(_fa4_register_kernels())
    torch._C._set_sdp_use_fa4(True)
    return handle


@cache
def _fa4_import_module(module_path: str) -> ModuleType:
    module = importlib.import_module(module_path)
    if not hasattr(module, "_flash_attn_fwd") or not hasattr(module, "_flash_attn_bwd"):
        raise RuntimeError(f"Module '{module_path}' does not expose FA4 kernels")
    return module


@cache
def _fa4_tensorwise_scales_cls(module: ModuleType) -> type | None:
    """FA4's ``TensorwiseScales`` if ``_flash_attn_fwd`` takes ``scales=``, None if it takes ``q_descale=``."""
    parameters = inspect.signature(module._flash_attn_fwd).parameters
    if "scales" in parameters:
        # TensorwiseScales lives where _flash_attn_fwd is defined, also when `module` only re-exports it.
        return importlib.import_module(
            module._flash_attn_fwd.__module__
        ).TensorwiseScales
    if "q_descale" in parameters:
        return None
    raise RuntimeError(
        f"{module.__name__}._flash_attn_fwd takes no fp8 descales; fp8 needs flash-attn-4 >= 4.0.0b10"
    )


def _fa4_register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")
    lib.impl("_flash_attention_forward", _fa4_flash_attention_forward_impl, "CUDA")
    # The flash .quantized ops take the descales right after return_debug_mask, as these impls do,
    # so one impl serves both overloads.
    lib.impl(
        "_flash_attention_forward.quantized",
        partial(_fa4_flash_attention_forward_impl, quantized=True),
        "CUDA",
    )
    lib.impl(
        "_flash_attention_forward_no_dropout_inplace",
        _fa4_flash_attention_forward_no_dropout_inplace_impl,
        "CUDA",
    )
    lib.impl(
        "_flash_attention_forward_no_dropout_inplace.quantized",
        partial(_fa4_flash_attention_forward_no_dropout_inplace_impl, quantized=True),
        "CUDA",
    )
    lib.impl("_flash_attention_backward", _fa4_flash_attention_backward_impl, "CUDA")
    lib.impl(
        "_scaled_dot_product_flash_attention",
        _fa4_scaled_dot_product_flash_attention_forward_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention.quantized",
        _fa4_scaled_dot_product_flash_attention_forward_quantized_impl,
        "CUDA",
    )
    lib.impl(
        "_scaled_dot_product_flash_attention_backward",
        _fa4_scaled_dot_product_flash_attention_backward_impl,
        "CUDA",
    )
    return lib


def _fa4_common_support_error(
    query: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    cum_seq_q: torch.Tensor | None,
    require_fp32: tuple[tuple[str, torch.Tensor], ...] = (),
    query_dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16),
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if query.dtype not in query_dtypes:
        names = " or ".join(str(dtype).removeprefix("torch.") for dtype in query_dtypes)
        return f"query dtype must be {names}"
    for name, tensor in require_fp32:
        if tensor.dtype != torch.float32:
            return f"{name} dtype must be float32"
    if cum_seq_q is None and query.dim() != 4:
        return "dense query must be 4D"
    if cum_seq_q is not None and query.dim() != 3:
        return "ragged query must be 3D"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(query.device) not in (9, 10):
        return "FA4 requires compute capability 9.0 or 10.0"
    return None


def _fa4_forward_support_error(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    return_debug_mask: bool,
    alibi_slopes: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    cum_seq_q: torch.Tensor | None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
    quantized: bool = False,
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
    major = _get_device_major(query.device)
    if block_table is not None and major != 10:
        return f"paged KV (block_table) not supported on SM {major}0"
    if num_splits is not None and num_splits > 1 and major != 10:
        return f"SplitKV (num_splits > 1) not supported on SM {major}0"
    if quantized and major != 10:
        return f"fp8 not supported on SM {major}0"
    if quantized and not query.dtype == key.dtype == value.dtype:
        return "query, key and value must share one fp8 dtype"
    error = _fa4_common_support_error(
        query,
        (query, key, value),
        cum_seq_q,
        query_dtypes=_FP8_DTYPES if quantized else (torch.float16, torch.bfloat16),
    )
    if error is not None:
        if error == "inputs must share device":
            return "query, key, value must be on same device"
        return error
    return None


def _fa4_backward_support_error(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_p: float,
    cum_seq_q: torch.Tensor | None,
    grad_logsumexp: torch.Tensor | None = None,
) -> str | None:
    if dropout_p != 0.0:
        return "dropout_p must be 0"
    tensors = (grad_out, query, key, value, out, logsumexp)
    require_fp32 = (("logsumexp", logsumexp),)
    if grad_logsumexp is not None:
        if grad_logsumexp.shape != logsumexp.shape:
            return "grad_logsumexp must have the shape of logsumexp"
        tensors += (grad_logsumexp,)
        require_fp32 += (("grad_logsumexp", grad_logsumexp),)
    error = _fa4_common_support_error(
        query, tensors, cum_seq_q, require_fp32=require_fp32
    )
    if error is not None:
        return error
    return None


def _aten_to_fa4_window_size(val: int | None) -> int | None:
    """need to convert -1 to None for FA4"""
    return None if val == -1 else val


Ts = TypeVarTuple("Ts")


def _transpose_dense(*tensors: *Ts) -> tuple[*Ts]:
    return tuple(t.transpose(1, 2) for t in tensors)  # type: ignore[attr-defined]


def _fa4_run_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    max_q: int | None,
    max_k: int | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    if _get_device_major(query.device) != 10:
        # SplitKV is SM100-only; the support check already rejects num_splits > 1
        num_splits = 1
    elif num_splits is None:
        # FA4 runs its SplitKV heuristic when num_splits < 1
        num_splits = 1 if torch.are_deterministic_algorithms_enabled() else 0

    kwargs: dict[str, Any] = {
        "softmax_scale": scale,
        "causal": is_causal,
        "window_size_left": _aten_to_fa4_window_size(window_size_left),
        "window_size_right": _aten_to_fa4_window_size(window_size_right),
        "return_lse": True,
        "cu_seqlens_q": cu_seq_q,
        "cu_seqlens_k": cu_seq_k,
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
        "seqused_k": seqused_k.contiguous() if seqused_k is not None else None,
        "page_table": block_table,
        "num_splits": num_splits,
        "out": out,
    }
    if any(t is not None for t in (q_descale, k_descale, v_descale)):
        tensorwise_scales_cls = _fa4_tensorwise_scales_cls(module)
        if tensorwise_scales_cls is not None:
            kwargs["scales"] = tensorwise_scales_cls(q_descale, k_descale, v_descale)
        else:
            # flash-attn-4 releases before 4.0.0b39 take the descales as separate arguments.
            kwargs.update(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
    out, lse, *_ = module._flash_attn_fwd(query, key, value, **kwargs)
    return out, lse.contiguous()


def _fa4_run_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cu_seq_q: torch.Tensor | None,
    cu_seq_k: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    deterministic: bool = False,
    grad_logsumexp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FA4_MODULE_PATH is None:
        raise RuntimeError("FA4 not registered")
    module = _fa4_import_module(_FA4_MODULE_PATH)
    dq, dk, dv = module._flash_attn_bwd(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp.contiguous(),
        softmax_scale=scale,
        causal=is_causal,
        window_size_left=_aten_to_fa4_window_size(window_size_left),
        window_size_right=_aten_to_fa4_window_size(window_size_right),
        cu_seqlens_q=cu_seq_q,
        cu_seqlens_k=cu_seq_k,
        deterministic=deterministic,
        # Older FA4 releases have no dlse argument.
        **({"dlse": grad_logsumexp} if grad_logsumexp is not None else {}),
    )
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
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    compute_auxiliary: bool = True,
    num_splits: int | None = None,
    quantized: bool = False,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        alibi_slopes,
        seqused_k,
        cum_seq_q,
        block_table,
        num_splits,
        quantized,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention forward unsupported: {error}")
    out, lse = _fa4_run_forward(
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
        block_table,
        num_splits,
        q_descale,
        k_descale,
        v_descale,
    )
    if compute_auxiliary:
        rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
        philox_offset = torch.zeros((), dtype=torch.uint64, device=query.device)
        debug_mask = torch.empty(0, dtype=out.dtype, device=query.device)
    else:
        rng_state = None
        philox_offset = None
        debug_mask = None
    return out, lse, rng_state, philox_offset, debug_mask


def _fa4_flash_attention_forward_no_dropout_inplace_impl(
    out: torch.Tensor,
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
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    seqused_k: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
    quantized: bool = False,
):
    _, lse, _, _, _ = _fa4_flash_attention_forward_impl(
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
        out=out,
        block_table=block_table,
        compute_auxiliary=False,
        num_splits=num_splits,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        quantized=quantized,
    )
    return lse


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
    grad_logsumexp: torch.Tensor | None = None,
):
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        cum_seq_q,
        grad_logsumexp,
    )
    if error is not None:
        raise RuntimeError(f"FA4 flash_attention backward unsupported: {error}")
    if (
        grad_logsumexp is not None
        and query.size(-1) == value.size(-1) == 256
        and _get_device_major(query.device) == 10
    ):
        raise NotImplementedError(
            "FA4 flash_attention backward does not support grad_logsumexp with head "
            "dim 256 on SM100"
        )
    deterministic = torch.are_deterministic_algorithms_enabled()
    dq, dk, dv = _fa4_run_backward(
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
        window_size_left,
        window_size_right,
        deterministic,
        grad_logsumexp,
    )
    return dq, dk, dv


def _fa4_scaled_dot_product_flash_attention_forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    quantized: bool = False,
):
    error = _fa4_forward_support_error(
        query,
        key,
        value,
        dropout_p,
        return_debug_mask,
        None,
        None,
        None,
        quantized=quantized,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA forward unsupported: {error}")
    q, k, v = _transpose_dense(query, key, value)

    out_bhsd = _empty_with_matching_layout(
        query,
        (*query.shape[:-1], value.size(-1)),
        dtype=torch.bfloat16 if quantized else None,
    )
    out_bshd = out_bhsd.transpose(1, 2)

    max_q_flash = q.size(1)
    max_k_flash = k.size(1)
    _, lse, rng_state, philox_offset, debug_mask = _fa4_flash_attention_forward_impl(
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
        quantized=quantized,
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


def _fa4_scaled_dot_product_flash_attention_forward_quantized_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
):
    """``.quantized`` SDPA overload: fp8 query, key and value; writes a bf16 output.

    Its schema puts the descales before ``dropout_p``, so it can't be a ``partial`` of the default impl.
    """
    return _fa4_scaled_dot_product_flash_attention_forward_impl(
        query,
        key,
        value,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        quantized=True,
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
    error = _fa4_backward_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        None,
    )
    if error is not None:
        raise RuntimeError(f"FA4 SDPA backward unsupported: {error}")
    q, k, v, o, go = _transpose_dense(query, key, value, out, grad_out)
    max_q = query.size(2)
    max_k = key.size(2)
    dq, dk, dv = _fa4_flash_attention_backward_impl(
        go,
        q,
        k,
        v,
        o,
        logsumexp,
        None,
        None,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )
    dq, dk, dv = _transpose_dense(dq, dk, dv)
    return dq, dk, dv


_registry.register_flash_attention_impl("FA4", register_fn=register_flash_attention_fa4)
