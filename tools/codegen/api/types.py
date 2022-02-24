from tools.codegen.model import (Argument, FunctionSchema, NativeFunction,
                                 BackendIndex,
                                 SelfArgument, TensorOptionsArguments, BaseTy)
from dataclasses import dataclass
from typing import Optional, Union, Sequence, TypeVar, List, Set, Dict
from enum import Enum

_T = TypeVar('_T')

TENSOR_LIST_LIKE_CTYPES = [
    'at::TensorList',
    'const c10::List<c10::optional<at::Tensor>> &',
    'const at::ITensorList &',
]

# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occr.

SpecialArgName = Enum('SpecialArgName', (
    'possibly_redundant_memory_format',
))
ArgName = Union[str, SpecialArgName]

# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: Optional[str]
    name: str

    def __str__(self) -> str:
        if self.ns is None or self.ns == '':
            return self.name
        return f"{self.ns}::{self.name}"

# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType('', 'uint8_t')
charT = BaseCppType('', 'int8_t')
shortT = BaseCppType('', 'int16_t')
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType('', 'int32_t')
longT = BaseCppType('', 'int64_t')
halfT = BaseCppType('at', 'Half')
doubleT = BaseCppType('', 'double')
floatT = BaseCppType('', 'float')
complexHalfT = BaseCppType('c10', 'complex<c10::Half>')  # stuffing template param here is an abuse
complexFloatT = BaseCppType('c10', 'complex<float>')
complexDoubleT = BaseCppType('c10', 'complex<double>')
boolT = BaseCppType('', 'bool')
bfloat16T = BaseCppType('at', 'BFloat16')
voidT = BaseCppType('', 'void')
stringT = BaseCppType('c10', 'string_view')
generatorT = BaseCppType('at', 'Generator')
scalarTypeT = BaseCppType('at', 'ScalarType')
tensorT = BaseCppType('at', 'Tensor')
optionalTensorRefT = BaseCppType('at', 'OptionalTensorRef')
tensorListT = BaseCppType('at', 'TensorList')
iTensorListT = BaseCppType('at', 'ITensorList')
iOptTensorRefListT = BaseCppType('at', 'IOptTensorRefList')
dimnameT = BaseCppType('at', 'Dimname')
dimnameListT = BaseCppType('at', 'DimnameList')
dimVectorT = BaseCppType('at', 'DimVector')
layoutT = BaseCppType('at', 'Layout')
deviceT = BaseCppType('at', 'Device')
scalarT = BaseCppType('at', 'Scalar')
optionalScalarRefT = BaseCppType('at', 'OptionalScalarRef')
memoryFormatT = BaseCppType('at', 'MemoryFormat')
qschemeT = BaseCppType('at', 'QScheme')
storageT = BaseCppType('at', 'Storage')
streamT = BaseCppType('at', 'Stream')
intArrayRefT = BaseCppType('at', 'IntArrayRef')
tensorOptionsT = BaseCppType('at', 'TensorOptions')
typeAndSizeT = BaseCppType('torch::autograd::generated', 'TypeAndSize')
tensorGeometryT = BaseCppType('at', 'TensorGeometry')

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
}

# CTypes encode C++ type structure as needed for translation.

@dataclass(frozen=True)
class BaseCType:
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return str(self.type).replace('at::', '')

    def remove_const_ref(self) -> 'CType':
        return self

@dataclass(frozen=True)
class ConstRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'const {self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'const {self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class MutRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'{self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'{self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class OptionalCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::optional<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::optional<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return OptionalCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ListCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::List<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::List<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ListCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'at::ArrayRef<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'ArrayRef<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayRefCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class VectorCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::vector<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::vector<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return VectorCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayCType:
    elem: 'CType'
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::array<{self.elem.cpp_type()},{self.size}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayCType(self.elem.remove_const_ref(), self.size)

@dataclass(frozen=True)
class TupleCType:
    elems: List['CType']

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::tuple<{",".join([e.cpp_type() for e in self.elems])}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{",".join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    def remove_const_ref(self) -> 'CType':
        return TupleCType([e.remove_const_ref() for e in self.elems])

CType = Union[
    BaseCType,
    OptionalCType,
    ConstRefCType,
    MutRefCType,
    ListCType,
    ArrayRefCType,
    ArrayCType,
    VectorCType,
    TupleCType
]

# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#

@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return self.type.cpp_type_registration_declarations()

    def remove_const_ref(self) -> 'NamedCType':
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> 'NamedCType':
        return NamedCType(name, self.type)

# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See tools.codegen.api.translate)

@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Union[Argument, TensorOptionsArguments, SelfArgument]
    # TODO: maybe don't represent default here
    default: Optional[str] = None

    @property
    def type(self) -> str:
        return self.nctype.cpp_type()

    def no_default(self) -> 'Binding':
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def decl_registration_declarations(self) -> str:
        type_s = self.nctype.cpp_type_registration_declarations()
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{type_s} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

    def with_name(self, name: str) -> 'Binding':
        return Binding(
            name=name,
            nctype=self.nctype,
            argument=self.argument,
            default=self.default
        )

# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.

@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType

# A CppSignature represents a single overload in the C++ API.  For
# any given function schema, there may be multiple CppSignatures
# corresponding to it, based on how we desugar to C++.  See also
# CppSignatureGroup.
@dataclass(frozen=True)
class CppSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: Set[str]

    # [Note: Structured Type Override]
    # We override Tensor[] for structured kernels in both the dispatcher
    # and in the C++ API.
    # This is a step towards enabling the new API: ITensorList.
    # See [Note: ITensorList]

    # Should the arguments be overriden with structured types?
    structured_type_override: bool

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementors access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments, faithful=self.faithful,
            method=self.method, cpp_no_default_args=self.cpp_no_default_args,
            structured_type_override=self.structured_type_override)

    def name(self) -> str:
        n = cpp.name(self.func, faithful_name_for_out_overloads=self.faithful)
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} (*)({args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} ({args_types_str})'


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: Optional[CppSignature]

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    @staticmethod
    def from_native_function(f: NativeFunction, *, method: bool, fallback_binding: bool = False) -> 'CppSignatureGroup':
        func = f.func
        faithful_signature: Optional[CppSignature]
        if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
            faithful_signature = CppSignature(
                func=func,
                faithful=True,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args,
                structured_type_override=f.part_of_structured_group
            )
        else:
            faithful_signature = None
        signature = CppSignature(
            func=func,
            faithful=False,
            method=method,
            fallback_binding=fallback_binding,
            cpp_no_default_args=f.cpp_no_default_args,
            structured_type_override=f.part_of_structured_group
        )
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
        )

@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # See [Note: Structured Type Override]
    # Should the arguments be overriden with structured types?
    structured_type_override: bool

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    def arguments(self) -> List[Binding]:
        return dispatcher.arguments(self.func, structured_type_override=self.structured_type_override)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None, *, is_redispatching_fn: bool = False) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ['c10::DispatchKeySet dispatchKeySet'] + args
        args_str = ', '.join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> List[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema, *, structured_type_override: bool, prefix: str = '') -> 'DispatcherSignature':
        return DispatcherSignature(func, structured_type_override, prefix)

@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # See [Note: Structured Type Override]
    # Should the arguments be overriden with structured types?
    structured_type_override: bool

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ', '.join(a.defn() for a in self.arguments())
        return f'{native.returns_type(self.func.returns).cpp_type()} (*)({args_str})'

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func, structured_type_override=self.structured_type_override)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns)

    def dispatcher_exprs(self) -> List[Expr]:
        args = self.arguments()
        dispatcher_args = dispatcher.arguments(self.func, structured_type_override=self.structured_type_override)
        return translate.translate(args, dispatcher_args, method=False)

@dataclass(frozen=True)
class ViewInverseSignature:
    # The NativeFunction this signature is derived from
    f: NativeFunction

    def name(self) -> str:
        return functionalization.name(self.f, functional_op=self.f, is_reverse=True, include_namespace=False)

    def decl(self) -> str:
        return_type = functionalization.returns_type(self.f.func)
        decls = [a.decl() for a in functionalization.inner_arguments(self.f, is_reverse=True)]
        return f"static {return_type.cpp_type()} {self.name()}({', '.join(decls)});"

    @staticmethod
    def from_func(f: NativeFunction) -> 'ViewInverseSignature':
        # Some assertions: lambdas are only used for view ops
        assert f.is_view_op
        assert not f.func.name.name.inplace  # only functional view ops need an inverse (e.g. not transpose_())
        return ViewInverseSignature(f)

@dataclass(frozen=True)
class FunctionalizationLambda:
    # The NativeFunction this signature is derived from
    f: NativeFunction

    # The corresponding out-of-place variant of the above NativeFunction
    # This only really matters for inplace-view ops.
    # e.g. transpose_() -> transpose().
    functional_op: NativeFunction

    # are we generating the forward lambda or the reverse lambda?
    is_reverse: bool

    def captures(self) -> List[Expr]:
        # The lambda lives inside of a kernel following the dispatcher API, so its outer context is the dispatcher arguments
        outer_ctx = dispatcher.arguments(self.f.func, structured_type_override=self.f.part_of_structured_group)
        capture_bindings = functionalization.capture_arguments(self.f, is_reverse=self.is_reverse)
        # allow_expensive_conversions is set because we want to convert
        # some reference types (IntArrayRef) to value types (vector<int64_t>).
        capture_exprs = translate.translate(outer_ctx, capture_bindings, method=False, allow_expensive_conversions=True)
        return capture_exprs

    def decl(self) -> str:
        return_type = functionalization.returns_type(self.f.func)
        capture_str = ', '.join(f'{val.type.name} = {val.expr}' for val in self.captures())
        decls = [a.decl() for a in functionalization.outer_arguments(is_reverse=self.is_reverse)]
        return f"[{capture_str}]({', '.join(decls)}) -> {return_type.cpp_type()}"

    def inner_call(self) -> str:
        inner_call_name = functionalization.name(
            self.f, functional_op=self.functional_op, is_reverse=self.is_reverse, include_namespace=True)

        arg_ctx = functionalization.outer_arguments(is_reverse=self.is_reverse)
        capture_ctx = functionalization.capture_arguments(self.f, is_reverse=self.is_reverse)
        full_ctx = arg_ctx + capture_ctx

        call_bindings = functionalization.inner_arguments(self.f, is_reverse=self.is_reverse)
        maybe_index = functionalization.inner_call_index(self.f.func)
        call_exprs = [e.expr for e in translate.translate(full_ctx, call_bindings, method=False)]
        if not self.is_reverse and maybe_index is not None:
            return f'{inner_call_name}({", ".join(call_exprs)})[{maybe_index.name}];'
        else:
            return f'{inner_call_name}({", ".join(call_exprs)});'

    @staticmethod
    def from_func(f: NativeFunction, *, functional_op: NativeFunction, is_reverse: bool) -> 'FunctionalizationLambda':
        # Some assertions: lambdas are only used for view ops
        assert f.is_view_op
        assert functional_op.is_view_op
        # functional_op corresponds to the functional-variant of f, and is only actually used if f itself is an inplace_view op.
        assert f.func.signature() == functional_op.func.signature()
        return FunctionalizationLambda(f, functional_op, is_reverse)


# Helper functions

def kernel_signature(
        f: NativeFunction, backend_index: BackendIndex, *, prefix: str = '') -> Union['NativeSignature', 'DispatcherSignature']:
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group, prefix=prefix)
    else:
        return NativeSignature(f.func, structured_type_override=f.part_of_structured_group, prefix=prefix)

# Functions only, no types
from tools.codegen.api import cpp, dispatcher, native, translate, functionalization
