from enum import Enum
from typing import Optional

import torch
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.variable.type import ScalarDtype


class Attribute(str, Enum):
    OPTIONAL = "optional"
    LENGTH = "len"
    DTYPE = "dtype"
    RANK = "rank"
    SIZE = "size"
    VALUE = "value"

    @staticmethod
    def hierarchy(argtype: ArgType):
        if argtype.is_tensor_list():
            if argtype == ArgType.TensorOptList:
                return [
                    Attribute.LENGTH,
                    Attribute.OPTIONAL,
                    Attribute.DTYPE,
                    Attribute.RANK,
                    Attribute.SIZE,
                    Attribute.VALUE,
                ]
            else:
                return [
                    Attribute.LENGTH,
                    Attribute.DTYPE,
                    Attribute.RANK,
                    Attribute.SIZE,
                    Attribute.VALUE,
                ]
        opt = [Attribute.OPTIONAL] if argtype.is_optional() else []
        if argtype.is_tensor():
            return opt + [
                Attribute.DTYPE,
                Attribute.RANK,
                Attribute.SIZE,
                Attribute.VALUE,
            ]
        elif argtype.is_scalar():
            return opt + [Attribute.DTYPE, Attribute.VALUE]
        elif argtype.is_list():
            return opt + [Attribute.LENGTH, Attribute.VALUE]
        else:
            return opt + [Attribute.VALUE]

    def get_vtype(
        self,
        argtype: Optional[ArgType] = None,
        scalar_dtype: Optional[ScalarDtype] = None,
    ):
        if self == Attribute.OPTIONAL:
            return bool
        if self == Attribute.DTYPE:
            if argtype is None:
                raise ValueError(f"Attribute {self} requires an argtype")
            if argtype.is_scalar():
                return ScalarDtype
            return torch.dtype
        if self in [Attribute.LENGTH, Attribute.RANK, Attribute.SIZE]:
            return int
        if self == Attribute.VALUE:
            if argtype is None:
                raise ValueError(f"Attribute {self} requires an argtype")
            if argtype.has_integer_value():
                return int
            if argtype.is_bool():
                return bool
            if argtype.is_float():
                return float
            if argtype.is_string():
                return str
            if argtype.is_memory_format():
                return str
            if argtype.is_scalar():
                if scalar_dtype is None:
                    raise ValueError(
                        "Attribute value for argtype scalar requires a scalar_dtype"
                    )
                return scalar_dtype.value
            if argtype.is_scalar_type():
                return torch.dtype
        return float

    def get_custom_limits(self, argtype: Optional[ArgType] = None):
        RANK_MAX = 6
        SIZE_MAX = 8
        TL_LEN_MAX = 6
        LIST_LEN_MAX = 8
        VALUE_LENGTH_MIN = -9
        VALUE_LENGTH_MAX = 9
        VALUE_MIN = -20
        VALUE_MAX = 20

        if self == Attribute.LENGTH:
            if argtype is None:
                raise ValueError(f"Attribute {self} requires an argtype")
            if argtype.is_tensor_list():
                return (0, TL_LEN_MAX)
            if argtype.is_shape():
                return (0, RANK_MAX)
            return (0, LIST_LEN_MAX)
        elif self == Attribute.RANK:
            return (0, RANK_MAX)
        elif self == Attribute.SIZE:
            return (0, SIZE_MAX)
        elif self == Attribute.VALUE:
            if argtype is None:
                raise ValueError(f"Attribute {self} requires an argtype")
            if argtype.is_shape():
                return (-SIZE_MAX, SIZE_MAX)
            if argtype.is_length() or argtype.is_length_list():
                return (VALUE_LENGTH_MIN, VALUE_LENGTH_MAX)
        return None
