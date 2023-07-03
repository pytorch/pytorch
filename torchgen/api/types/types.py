"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""
from dataclasses import dataclass
from typing import Dict, TypeVar

from torchgen.model import BaseTy, ScalarType

from .types_base import (
    BaseCppType,
    BaseCType,
    boolT,
    byteT,
    charT,
    CType,
    doubleT,
    floatT,
    int32T,
    longT,
    shortT,
)

_T = TypeVar("_T")

TENSOR_LIST_LIKE_CTYPES = [
    "at::TensorList",
    "const c10::List<c10::optional<at::Tensor>> &",
    "const at::ITensorListRef &",
]


halfT = BaseCppType("at", "Half")
complexHalfT = BaseCppType(
    "c10", "complex<c10::Half>"
)  # stuffing template param here is an abuse
complexFloatT = BaseCppType("c10", "complex<float>")
complexDoubleT = BaseCppType("c10", "complex<double>")
bfloat16T = BaseCppType("at", "BFloat16")
float8_e5m2T = BaseCppType("at", "Float8_e5m2")
float8_e4m3fnT = BaseCppType("at", "Float8_e4m3fn")
stringT = BaseCppType("c10", "string_view")
generatorT = BaseCppType("at", "Generator")
scalarTypeT = BaseCppType("at", "ScalarType")
tensorT = BaseCppType("at", "Tensor")
optionalTensorRefT = BaseCppType("at", "OptionalTensorRef")
tensorListT = BaseCppType("at", "TensorList")
iTensorListRefT = BaseCppType("at", "ITensorListRef")
iOptTensorListRefT = BaseCppType("at", "IOptTensorListRef")
dimnameT = BaseCppType("at", "Dimname")
dimnameListT = BaseCppType("at", "DimnameList")
dimVectorT = BaseCppType("at", "DimVector")
layoutT = BaseCppType("at", "Layout")
deviceT = BaseCppType("at", "Device")
scalarT = BaseCppType("at", "Scalar")
optionalScalarRefT = BaseCppType("at", "OptionalScalarRef")
memoryFormatT = BaseCppType("at", "MemoryFormat")
qschemeT = BaseCppType("at", "QScheme")
storageT = BaseCppType("at", "Storage")
streamT = BaseCppType("at", "Stream")
intArrayRefT = BaseCppType("at", "IntArrayRef")
optionalIntArrayRefT = BaseCppType("at", "OptionalIntArrayRef")
optionalSymIntArrayRefT = BaseCppType("at", "OptionalSymIntArrayRef")
tensorOptionsT = BaseCppType("at", "TensorOptions")
typeAndSizeT = BaseCppType("torch::autograd::generated", "TypeAndSize")
tensorGeometryT = BaseCppType("at", "TensorGeometry")
SymIntT = BaseCppType("c10", "SymInt")
symIntArrayRefT = BaseCppType("c10", "SymIntArrayRef")

# Types representing template parameters.  Technically, we probably shouldn't
# represent them this way in codegen, but it was pretty convenient.
scalar_t = BaseCppType("", "scalar_t")
opmath_t = BaseCppType("", "opmath_t")

ScalarTypeToCppMapping: Dict[ScalarType, BaseCppType] = {
    ScalarType.Byte: byteT,
    ScalarType.Char: charT,
    ScalarType.Short: shortT,
    ScalarType.Int: int32T,
    ScalarType.Long: longT,
    ScalarType.Half: halfT,
    ScalarType.Float: floatT,
    ScalarType.Double: doubleT,
    ScalarType.ComplexHalf: complexHalfT,
    ScalarType.ComplexFloat: complexFloatT,
    ScalarType.ComplexDouble: complexDoubleT,
    ScalarType.Bool: boolT,
    ScalarType.Float8_e5m2: float8_e5m2T,
    ScalarType.Float8_e4m3fn: float8_e4m3fnT,
}

BaseTypeToCppMapping: Dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Dimname: dimnameT,
    BaseTy.DimVector: dimVectorT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
    BaseTy.SymInt: SymIntT,
}

# CTypes encode C++ type structure as needed for translation.


@dataclass(frozen=True)
class OptionalCType(CType):
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"c10::optional<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"c10::optional<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ListCType(CType):
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"c10::List<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"c10::List<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return ListCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"at::ArrayRef<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"ArrayRef<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return ArrayRefCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class VectorizedCType(CType):
    # This template is explicitly specialized, so the only valid
    # elems are those we have specializations for (e.g., float, double, ...)
    # scalar_t is also a common argument here (when we are codegen in
    # a templated context)
    elem: BaseCType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"at::vec::Vectorized<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        raise NotImplementedError

    def remove_const_ref(self) -> "CType":
        return self
