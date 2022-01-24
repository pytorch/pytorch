from typing import List

import torch
import tensorrt as trt

from .types import Shape, TRTDataType


def torch_dtype_to_trt(dtype: torch.dtype) -> TRTDataType:
    """
    Convert PyTorch data types to TensorRT data types.

    Args:
        dtype (torch.dtype): A PyTorch data type.

    Returns:
        The equivalent TensorRT data type.
    """
    if trt.__version__ >= "7.0" and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype: TRTDataType) -> torch.dtype:
    """
    Convert TensorRT data types to PyTorch data types.

    Args:
        dtype (TRTDataType): A TensorRT data type.

    Returns:
        The equivalent PyTorch data type.
    """
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= "7.0" and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def get_dynamic_dims(shape: Shape) -> List[int]:
    """
    This function finds the dynamic dimensions in the given
    shape. A dimension is dynamic if it's -1.

    Args:
        shape (Shape): A sequence of integer that represents
            the shape of a tensor.

    Returns:
        A list of integers contains all the dynamic dimensions
        in the given shape
    """
    dynamic_dims = []

    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)

    return dynamic_dims
