from tools.codegen.model import *
from dataclasses import dataclass
from typing import Optional, Union, Sequence, Tuple, TypeVar, List

_T = TypeVar('_T')

# ------------------------------------------------------------------- #

#                           cpp types

# ------------------------------------------------------------------- #

# Describe a single argument (e.g., the x in "f(int x)") in the C++ API.
@dataclass(frozen=True)
class CppArgument:
    # C++ type, e.g., int
    type: str
    # C++ name, e.g., x
    name: str
    # Only used by the header, but we work it out in all cases anyway
    default: Optional[str]
    # The JIT argument(s) this formal was derived from.  May
    # correspond to multiple arguments if this is TensorOptions!
    argument: Union[Argument, TensorOptionsArguments]

    # Default string representation prints the most elaborated form
    # of the formal
    def __str__(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

    # Return a copy of CppArgument with defaults removed
    def no_default(self) -> 'CppArgument':
        return CppArgument(
            type=self.type,
            name=self.name,
            default=None,
            argument=self.argument,
        )

    # However, you might also find the version with no default useful
    def str_no_default(self) -> str:
        return f"{self.type} {self.name}"

# An argument pack groups several CppArguments together into
# a semantically meaningful unit.  Don't let the packing
# deceive you: if you look at these arguments in C++, they're
# always packing (in analogy to how parameter packs in C++
# templates actually turn into separate arguments when you
# unpack them).
@dataclass(frozen=True)
class CppArgumentPackIface:
    # Return this argument pack, but with default stripped
    def no_default(self: _T) -> _T:
        raise NotImplementedError

    # Unpack the pack into a sequence of arguments, discarding
    # semantic information, and also discarding the implicit this
    # argument that doesn't actually show up in declarations
    def explicit_arguments(self) -> Sequence[CppArgument]:
        raise NotImplementedError

# Lifts a single CppArgument into a pack.
@dataclass(frozen=True)
class CppSingleArgumentPack(CppArgumentPackIface):
    this: CppArgument

    def no_default(self) -> 'CppSingleArgumentPack':
        return CppSingleArgumentPack(this=self.this.no_default())

    @property
    def type(self) -> str:
        return self.this.type

    def explicit_arguments(self) -> Sequence[CppArgument]:
        return [self.this]

# Describe an implicit this argument (*this) on methods in the C++ API.
# We don't use CppSingleArgumentPack because these never show up
# in the explicit arguments list
@dataclass(frozen=True)
class CppThisArgumentPack(CppArgumentPackIface):
    # The grouped JIT argument this formal was derived from
    argument: SelfArgument

    # C++ type, e.g., Tensor&
    type: str

    # this arguments are never defaulted
    def no_default(self) -> 'CppThisArgumentPack':
        return self

    # The this argument is implicit, so it's not included in the
    # explicit arguments list.
    def explicit_arguments(self) -> Sequence[CppArgument]:
        return []

# Semantically represents a bundle of CppArguments that collectively
# represent a TensorOptions.  If you don't care about TensorOptions
# processing, think of this as just a list of four CppArguments; however
# if you need to bundle these arguments back into a single
# TensorOptions, it will be easiest to operate on this struct as a
# whole.
#
# NOTE: this does NOT represent a 'const TensorOptions&' argument.
# If you have one of those, it will be CppSingleArgumentPack
@dataclass(frozen=True)
class CppTensorOptionsArgumentPack(CppArgumentPackIface):
    argument: TensorOptionsArguments
    dtype: CppArgument
    layout: CppArgument
    device: CppArgument
    pin_memory: CppArgument

    # Remove the defaults from each of the constituent arguments
    # representing the TensorOptions
    def no_default(self) -> 'CppTensorOptionsArgumentPack':
        return CppTensorOptionsArgumentPack(
            argument=self.argument,
            dtype=self.dtype.no_default(),
            layout=self.layout.no_default(),
            device=self.device.no_default(),
            pin_memory=self.pin_memory.no_default(),
        )

    # Flatten the TensorOptions into individual CppArguments
    def explicit_arguments(self) -> Sequence[CppArgument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]

# Use this instead of CppArgumentPackIface, as this is a closed union
CppArgumentPack = Union[
    CppSingleArgumentPack,
    CppThisArgumentPack,
    CppTensorOptionsArgumentPack,
]

@dataclass(frozen=True)
class CppExpr:
    type: str
    expr: str

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

    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[CppArgument]:
        return [sub_a for a in self.argument_packs() for sub_a in a.explicit_arguments()]

    # Return the packed argument structure of this signature.  This preserves
    # high-level structure of the arguments so you may find it easier to do
    # translations working with this representation.
    def argument_packs(self) -> Sequence[CppArgumentPack]:
        arguments: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
        if self.faithful:
            arguments.extend(self.func.arguments.non_out)
            arguments.extend(self.func.arguments.out)
        else:
            arguments.extend(self.func.arguments.out)
            arguments.extend(self.func.arguments.non_out)

        if self.faithful:
            # After this, manually do overload disambiguation, by
            # dropping defaults from the faithful signature.  In
            # principle, we should be able to do this at some later
            # point in time with other overload disambiguation
            argument_packs = tuple(
                cpp.argument_faithful(a, method=self.method).no_default() for a in arguments
            )
        else:
            argument_packs = tuple(
                cpp.argument(a, method=self.method) for a in arguments
            )
        return argument_packs

    def name(self) -> str:
        n = cpp.name(self.func, faithful_name_for_out_overloads=self.faithful)
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(self) -> str:
        returns_type = cpp.returns_type(self.func.returns)
        cpp_args_str = ', '.join(map(str, self.arguments()))
        return f"{returns_type} {self.name()}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(self, *, prefix: str = "") -> str:
        returns_type = cpp.returns_type(self.func.returns)
        cpp_args_str = ', '.join(a.str_no_default() for a in self.arguments())
        name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: Optional[CppSignature]

    @staticmethod
    def from_schema(func: FunctionSchema, *, method: bool, fallback_binding: bool = False) -> 'CppSignatureGroup':
        faithful_signature: Optional[CppSignature]
        if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
            faithful_signature = CppSignature(func=func, faithful=True, method=method, fallback_binding=fallback_binding)
        else:
            faithful_signature = None
        signature = CppSignature(func=func, faithful=False, method=method, fallback_binding=fallback_binding)
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
        )

# ------------------------------------------------------------------- #

#                           dispatcher types

# ------------------------------------------------------------------- #

@dataclass(frozen=True)
class DispatcherExpr:
    type: str
    expr: str

@dataclass(frozen=True)
class DispatcherArgument:
    type: str
    name: str
    # dispatcher NEVER has defaults
    argument: Union[Argument, TensorOptionsArguments]
    # TensorOptionsArguments can occur when not using full c10 dispatch

    def __str__(self) -> str:
        return f"{self.type} {self.name}"

@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Note to self: if we ever need to reassemble tensor options, we may need to
    # also preserve grouping with DispatcherTensorOptionsArguments.  This should
    # be an unlikely situation, however, since the general direction we are
    # headed is to make native:: take everything in expanded form, so you
    # shouldn't need to reassemble
    _arguments: Tuple[DispatcherArgument, ...]
    _returns_type: str

    def arguments(self) -> Tuple[DispatcherArgument, ...]:
        return self._arguments

    def name(self) -> str:
        return dispatcher.name(self.func)

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(map(str, self.arguments()))
        if name is None:
            name = self.name()
        return f"{self._returns_type} {name}({args_str})"

    def exprs(self) -> Sequence[DispatcherExpr]:
        return dispatcher.exprs(self.arguments())

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self._arguments)
        return f'{self._returns_type} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema) -> 'DispatcherSignature':
        arguments = dispatcher.arguments(func)
        returns_type = dispatcher.returns_type(func.returns)

        return DispatcherSignature(
            func=func,
            _arguments=arguments,
            _returns_type=returns_type,
        )

# ------------------------------------------------------------------- #

#                    native types (NativeFunctions.h)

# ------------------------------------------------------------------- #

# NB: the "native" here is not to be confused with the native in
# native_functions.yaml

@dataclass(frozen=True)
class NativeExpr:
    type: str
    expr: str

@dataclass(frozen=True)
class NativeArgument:
    type: str
    name: str
    # Native function arguments have defaults to make it a little
    # easier to call them directly to bypass dispatch.
    default: Optional[str]
    argument: Union[Argument, TensorOptionsArguments]

    # Convention here is swapped because arguably NativeFunctions.h
    # shouldn't have defaults (they should be handled during dispatching).
    # The defaults are a mild convenience, however, for people who directly
    # call native:: functions
    def __str__(self) -> str:
        return f"{self.type} {self.name}"

    def str_with_default(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    _arguments: Tuple[NativeArgument, ...]
    _returns_type: str

    def name(self) -> str:
        return native.name(self.func)

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(map(str, self.arguments()))
        if name is None:
            name = self.name()
        return f"{self._returns_type} {name}({args_str})"

    def ptr_type(self) -> str:
        args_str = ', '.join(map(str, self.arguments()))
        return f'{self._returns_type} (*)({args_str})'

    def arguments(self) -> Tuple[NativeArgument, ...]:
        return self._arguments

    def dispatcher_exprs(self) -> Sequence['DispatcherExpr']:
        return dispatcher.nativearguments_exprs(self.arguments())

    @staticmethod
    def from_schema(func: FunctionSchema) -> 'NativeSignature':
        arguments = native.arguments(func)
        returns_type = native.returns_type(func.returns)

        return NativeSignature(
            func=func,
            _arguments=arguments,
            _returns_type=returns_type,
        )

# ------------------------------------------------------------------- #

#                           meta api

# ------------------------------------------------------------------- #

@dataclass(frozen=True)
class MetaArgument:
    type: str
    name: str
    # structured kernels (for which MetaArgument matters) always will
    # be use_c10_dispatcher full.  That means JIT arguments and
    # meta arguments are always in 1:1 correspondence.  If this is ever not true
    # we will have to do something more fancy here.
    argument: Argument

    def __str__(self) -> str:
        return f"{self.type} {self.name}"

# Functions only, no types
from tools.codegen.api import cpp, dispatcher, native
