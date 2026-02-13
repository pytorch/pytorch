"""CuteDSL LayerNorm dispatcher override."""
# mypy: allow-untyped-defs

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
from torch.library import Library

from . import _registry
from ._kernels.layernorm import cutedsl_layernorm_bwd, cutedsl_layernorm_fwd


__all__ = [
    "register_cutedsl_layernorm",
]


@dataclass
class _CuteDSLLayerNormHandle:
    library: Library | None

    def remove(self) -> None:
        self.library = None


@cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def register_cutedsl_layernorm() -> _CuteDSLLayerNormHandle:
    """
    Register CuteDSL LayerNorm kernels with the PyTorch dispatcher.
    """
    return _CuteDSLLayerNormHandle(_register_kernels())


def _register_kernels() -> Library:
    lib = Library("aten", "IMPL", "CUDA")  # noqa: TOR901
    lib.impl("native_layer_norm", _cutedsl_layer_norm_impl, "CUDA")
    lib.impl(
        "native_layer_norm_backward", _cutedsl_layer_norm_backward_impl, "CUDA"
    )
    return lib


def _support_error(
    input: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input dtype must be float16, bfloat16, or float32"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(input.device) not in (9, 10):
        return "CuteDSL LayerNorm requires compute capability 9.0 or 10.0"
    return None


def _cutedsl_layer_norm_impl(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
):
    tensors: tuple[torch.Tensor, ...] = (input,)
    if weight is not None:
        tensors = (*tensors, weight)
    if bias is not None:
        tensors = (*tensors, bias)
    error = _support_error(input, tensors)
    if error is not None:
        raise RuntimeError(f"CuteDSL LayerNorm forward unsupported: {error}")

    output, mean, rstd = cutedsl_layernorm_fwd(
        input, weight, bias, normalized_shape, eps
    )
    return output, mean, rstd


def _cutedsl_layer_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_mask: list[bool],
):
    tensors: tuple[torch.Tensor, ...] = (grad_out, input, mean, rstd)
    if weight is not None:
        tensors = (*tensors, weight)
    if bias is not None:
        tensors = (*tensors, bias)
    error = _support_error(input, tensors)
    if error is not None:
        raise RuntimeError(f"CuteDSL LayerNorm backward unsupported: {error}")

    grad_input, grad_weight, grad_bias = cutedsl_layernorm_bwd(
        grad_out, input, mean, rstd, weight, bias, normalized_shape
    )

    if not output_mask[0]:
        grad_input = None
    if not output_mask[1]:
        grad_weight = None
    if not output_mask[2]:
        grad_bias = None
    return grad_input, grad_weight, grad_bias


_registry.register_norm_impl(
    "cutedsl_layernorm", register_fn=register_cutedsl_layernorm
)
