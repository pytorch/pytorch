from enum import Enum


class ArgType(str, Enum):
    Tensor = "Tensor"
    TensorOpt = "Tensor?"

    TensorList = "Tensor[]"
    TensorOptList = "Tensor?[]"

    Scalar = "Scalar"
    ScalarOpt = "Scalar?"

    ScalarType = "ScalarType"
    ScalarTypeOpt = "ScalarType?"

    Dim = "Dim"
    DimOpt = "Dim?"
    DimList = "Dim[]"
    DimListOpt = "Dim[]?"

    Shape = "Shape"
    Index = "Index"
    IndexOpt = "Index?"
    Length = "Length"
    LengthOpt = "Length?"
    LengthList = "Length[]"

    Bool = "Bool"
    Int = "Integer"
    IntOpt = "Integer?"
    Float = "Float"
    FloatOpt = "Float?"
    String = "String"
    StringOpt = "String?"
    MemoryFormat = "MemoryFormat"

    def is_tensor(self):
        return self in [ArgType.Tensor, ArgType.TensorOpt]

    def is_tensor_list(self):
        return self in [ArgType.TensorList, ArgType.TensorOptList]

    def is_scalar(self):
        return self in [ArgType.Scalar, ArgType.ScalarOpt]

    def is_scalar_type(self):
        return self in [ArgType.ScalarType, ArgType.ScalarTypeOpt]

    def is_dim(self):
        return self in [ArgType.Dim, ArgType.DimOpt]

    def is_dim_list(self):
        return self in [ArgType.DimList, ArgType.DimListOpt]

    def is_shape(self):
        return self in [ArgType.Shape]

    def is_index(self):
        return self in [ArgType.Index, ArgType.IndexOpt]

    def is_length(self):
        return self in [ArgType.Length, ArgType.LengthOpt]

    def is_length_list(self):
        return self in [ArgType.LengthList]

    def is_bool(self):
        return self in [ArgType.Bool]

    def is_int(self):
        return self in [ArgType.Int, ArgType.IntOpt]

    def is_float(self):
        return self in [ArgType.Float, ArgType.FloatOpt]

    def is_string(self):
        return self in [ArgType.String, ArgType.StringOpt]

    def is_memory_format(self):
        return self in [ArgType.MemoryFormat]

    def is_optional(self):
        return self in [
            ArgType.TensorOpt,
            ArgType.ScalarOpt,
            ArgType.ScalarTypeOpt,
            ArgType.DimOpt,
            ArgType.DimListOpt,
            ArgType.FloatOpt,
            ArgType.IndexOpt,
            ArgType.IntOpt,
            ArgType.LengthOpt,
        ]

    def is_list(self):
        return self in [
            ArgType.TensorList,
            ArgType.TensorOptList,
            ArgType.DimList,
            ArgType.DimListOpt,
            ArgType.LengthList,
            ArgType.Shape,
        ]

    def has_integer_value(self):
        return self in [
            ArgType.Dim,
            ArgType.DimOpt,
            ArgType.DimList,
            ArgType.DimListOpt,
            ArgType.Shape,
            ArgType.Index,
            ArgType.IndexOpt,
            ArgType.Length,
            ArgType.LengthOpt,
            ArgType.LengthList,
            ArgType.Int,
            ArgType.IntOpt,
        ]

    def has_dtype(self):
        return (
            self.is_tensor()
            or self.is_tensor_list()
            or self.is_scalar()
            or self.is_scalar_type()
        )
