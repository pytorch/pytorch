import torch
from typing import Type, Union

from ._C import DataType

NumberTypeType = Union[Type[bool], Type[int], Type[float], Type[complex]]
    
_torch_dtype_to_nvfuser_dtype_map = {
    torch.cdouble: DataType.ComplexDouble,
    torch.cfloat: DataType.ComplexFloat,
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.bfloat16: DataType.BFloat16,
    torch.long: DataType.Int,
    torch.int: DataType.Int32,
    torch.uint8: DataType.Int32,
    torch.bool: DataType.Bool,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}

def torch_dtype_to_nvfuser_dtype(dtype: Union[torch.dtype, NumberTypeType]):
    """
    Translates from torch.dtype to nvFuser's DataType enum
    """
    return _torch_dtype_to_nvfuser_dtype_map[dtype]


