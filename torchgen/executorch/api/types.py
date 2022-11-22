from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from torchgen.api.types_base import    (
    ArgName,
    BaseCppType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    Expr,
    MutRefCType,
    NamedCType,
    SpecialArgName,
)
from torchgen.model import BaseTy, FunctionSchema, NativeFunction

# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType("", "uint8_t")
charT = BaseCppType("", "int8_t")
shortT = BaseCppType("", "int16_t")
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType("", "int32_t")
longT = BaseCppType("", "int64_t")
halfT = BaseCppType("torch::executor", "Half")
doubleT = BaseCppType("", "double")
floatT = BaseCppType("", "float")
boolT = BaseCppType("", "bool")
bfloat16T = BaseCppType("torch::executor", "BFloat16")
voidT = BaseCppType("", "void")
stringT = BaseCppType("torch::executor", "string_view")
scalarTypeT = BaseCppType("torch::executor", "ScalarType")
tensorT = BaseCppType("torch::executor", "Tensor")
tensorListT = BaseCppType("torch::executor", "TensorList")
scalarT = BaseCppType("torch::executor", "Scalar")
memoryFormatT = BaseCppType("torch::executor", "MemoryFormat")
intArrayRefT = BaseCppType("torch::executor", "IntArrayRef")
optionalT = BaseCppType("torch::executor", "optional")


BaseTypeToCppMapping: Dict[BaseTy, BaseCppType] = {
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
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"torch::executor::optional<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::optional<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"torch::executor::ArrayRef<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::ArrayRef<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return ArrayRefCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class TupleCType(CType):
    elems: List["CType"]

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::tuple<{",".join([e.cpp_type() for e in self.elems])}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{",".join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    def remove_const_ref(self) -> "CType":
        return TupleCType([e.remove_const_ref() for e in self.elems])


@dataclass(frozen=True)
class VectorCType(CType):
    elem: "CType"

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::vector<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"::std::vector<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> "CType":
        return VectorCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayCType(CType):
    elem: "CType"
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::array<{self.elem.cpp_type()},{self.size}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>"

    def remove_const_ref(self) -> "CType":
        return ArrayCType(self.elem.remove_const_ref(), self.size)


# This signature is merely a CppSignature with Executorch types. The inline definition of RuntimeSignature is generated in Functions.h and it's used by unboxing functions.
@dataclass(frozen=True)
class CppSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: Set[str]

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    def arguments(self) -> List[Binding]:
        return cpp.arguments(
            self.func.arguments,
            faithful=True,  # always faithful, out argument at the end
            symint=False,  # symint not supported
            method=False,  # method not supported
            cpp_no_default_args=self.cpp_no_default_args,
        )

    def name(self) -> str:
        return self.prefix + cpp.name(
            self.func,
            faithful_name_for_out_overloads=True,
        )

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None) -> str:
        args = [a.defn() for a in self.arguments()]
        args_str = ", ".join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def returns_type(self) -> CType:
        return cpp.returns_type(self.func.returns, symint=False)

    @staticmethod
    def from_native_function(f: NativeFunction, *, prefix: str = "") -> "CppSignature":
        return CppSignature(
            func=f.func, prefix=prefix, cpp_no_default_args=f.cpp_no_default_args
        )


@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    symint: bool

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ", ".join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})"

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func, symint=self.symint)



from torchgen.executorch.api import cpp, native
