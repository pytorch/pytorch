from __future__ import annotations

from dataclasses import dataclass

from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    CType,
    doubleT,
    Expr,
    longT,
    MutRefCType,
    NamedCType,
)
from torchgen.model import BaseTy


halfT = BaseCppType("torch::executor", "Half")
bfloat16T = BaseCppType("torch::executor", "BFloat16")
stringT = BaseCppType("torch::executor", "string_view")
scalarTypeT = BaseCppType("torch::executor", "ScalarType")
tensorT = BaseCppType("torch::executor", "Tensor")
tensorListT = BaseCppType("torch::executor", "TensorList")
scalarT = BaseCppType("torch::executor", "Scalar")
memoryFormatT = BaseCppType("torch::executor", "MemoryFormat")
intArrayRefT = BaseCppType("torch::executor", "IntArrayRef")
optionalT = BaseCppType("torch::executor", "optional")
contextT = BaseCppType("torch::executor", "KernelRuntimeContext")

contextExpr = Expr(
    expr="context",
    type=NamedCType(name="context", type=MutRefCType(BaseCType(contextT))),
)

contextArg = Binding(
    name="context",
    nctype=contextExpr.type,
    argument=None,  # type: ignore[arg-type]
    default=None,
)

BaseTypeToCppMapping: dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
}


@dataclass(frozen=True)
class OptionalCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"torch::executor::optional<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::optional<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> CType:
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"torch::executor::ArrayRef<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::ArrayRef<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> CType:
        return ArrayRefCType(self.elem.remove_const_ref())
