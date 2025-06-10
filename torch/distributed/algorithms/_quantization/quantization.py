from __future__ import annotations

import functools
from enum import Enum
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.distributed as dist


TORCH_HALF_MIN = torch.finfo(torch.float16).min
TORCH_HALF_MAX = torch.finfo(torch.float16).max


class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """

    FP16 = ("fp16",)
    BFP16 = "bfp16"

    def __str__(self) -> str:
        return self.value


def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def _quantize_tensor(tensor: torch.Tensor, qtype: DQuantType) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _quantize_tensor_list(
    tensor_list: list[torch.Tensor], qtype: DQuantType
) -> list[torch.Tensor]:
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list


def _dequantize_tensor(
    tensor: torch.Tensor, qtype: DQuantType, quant_loss: float | None = None
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        if quant_loss is None:
            return tensor.float()
        else:
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


def _dequantize_tensor_list(
    tensor_list: list[torch.Tensor], qtype: DQuantType, quant_loss: float | None = None
) -> list[torch.Tensor]:
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_dequantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    dequantized_tensor_list = [_dequantize_tensor(t, qtype) for t in tensor_list]
    return dequantized_tensor_list


def auto_quantize(
    func: Callable[..., Any], qtype: DQuantType, quant_loss: float | None = None
) -> Callable[..., Any]:
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        group = kwargs.get("group", None)
        async_op = kwargs.get("async_op", False)
        if async_op is True:
            raise RuntimeError("The async_op=True mode is not supported yet.")
        if func == dist.all_gather:
            tensors = args[0]
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        elif func == dist.all_to_all:
            tensors = args[0]
            input_tensor_list = _quantize_tensor_list(args[1], qtype)
            out_tensor_list = _quantize_tensor_list(tensors, qtype)
            dist.all_to_all(
                out_tensor_list, input_tensor_list, group=group, async_op=async_op
            )
            dequantized_tensors = _dequantize_tensor_list(
                out_tensor_list, qtype, quant_loss=quant_loss
            )
            for i, t in enumerate(dequantized_tensors):
                tensors[i] = t

        elif func == dist.all_to_all_single:
            tensors = args[0]
            out_splits = kwargs.get("out_splits", None)
            in_splits = kwargs.get("in_splits", None)
            # Quantizing the input/output tensor
            input_tensor = _quantize_tensor(args[1], qtype)
            out_tensor = _quantize_tensor(tensors, qtype)
            dist.all_to_all_single(
                out_tensor, input_tensor, out_splits, in_splits, group=group
            )
            dequantized_tensor = _dequantize_tensor(
                out_tensor, qtype, quant_loss=quant_loss
            )
            tensors.copy_(dequantized_tensor)
        else:
            raise RuntimeError(f"The collective op {func} is not supported yet")

    return wrapper
