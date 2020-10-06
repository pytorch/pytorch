from tools.codegen.model import *
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence, Tuple

# ------------------------------------------------------------------- #

#                       Grouping arguments

# ------------------------------------------------------------------- #

# Represents the implicit *this argument for method calls in C++ API
@dataclass(frozen=True)
class ThisArgument:
    argument: Argument

# Bundle of arguments that represent a TensorOptions in the C++ API.
@dataclass(frozen=True)
class TensorOptionsArguments:
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]

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

    # List of CppArguments that this structure explicitly represents
    def explicit_arguments(self) -> Sequence['CppArgument']:
        return [self]

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

# Describe an implicit this argument (*this) on methods in the C++ API
@dataclass(frozen=True)
class CppThisArgument:
    # The grouped JIT argument this formal was derived from
    argument: ThisArgument

    # C++ type, e.g., Tensor&
    type: str

    # this arguments are never defaulted
    def no_default(self) -> 'CppThisArgument':
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
@dataclass(frozen=True)
class CppTensorOptionsArguments:
    argument: TensorOptionsArguments
    dtype: CppArgument
    layout: CppArgument
    device: CppArgument
    pin_memory: CppArgument

    # Remove the defaults from each of the constituent arguments
    # representing the TensorOptions
    def no_default(self) -> 'CppTensorOptionsArguments':
        return CppTensorOptionsArguments(
            argument=self.argument,
            dtype=self.dtype.no_default(),
            layout=self.layout.no_default(),
            device=self.device.no_default(),
            pin_memory=self.pin_memory.no_default(),
        )

    # Flatten the TensorOptions into individual CppArguments
    def explicit_arguments(self) -> Sequence[CppArgument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]

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

    # Enough information about the C++ types to generate a full
    # C++ type signature for this signature.  I'm not too sure
    # if these are the right representations, so for now this
    # is intended to be more abstract.  Prefer using
    # cpp_grouped_arguments() to access this
    _cpp_grouped_arguments: Tuple[Union[CppArgument, CppTensorOptionsArguments, CppThisArgument], ...]
    _cpp_returns_type: str

    # WARNING: This is probably NOT what you want
    #
    # Return the flattened argument structure of this signature,
    # discarding information about grouping.  If you are planning
    # to translate these arguments into expressions on another
    # API, you probably want cpp_grouped_arguments instead (which
    # will handle TensorOptions translations correctly)
    def cpp_ungrouped_arguments(self) -> Sequence[CppArgument]:
        return [sub_a for a in self._cpp_grouped_arguments for sub_a in a.explicit_arguments()]

    # Return the grouped argument structure of this signature.  This
    # preserves high-level structure of the arguments so you may
    # find it easier to do translations working with this
    # representation.
    def cpp_grouped_arguments(self) -> Sequence[Union[CppArgument, CppTensorOptionsArguments, CppThisArgument]]:
        return self._cpp_grouped_arguments

    # Render the C++ declaration for this signature
    def decl(self) -> str:
        cpp_args_str = ', '.join(map(str, self.cpp_ungrouped_arguments()))
        return f"{self._cpp_returns_type} {cpp.name(self.func)}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(self, prefix: str = "") -> str:
        cpp_args_str = ', '.join(a.str_no_default() for a in self.cpp_ungrouped_arguments())
        return f"{self._cpp_returns_type} {prefix}{cpp.name(self.func)}({cpp_args_str})"

    # NB: This constructor knows how to disambiguate defaults when
    # faithful is True.  Ideally this would live as an external process
    # see https://github.com/pytorch/pytorch/pull/45666
    @staticmethod
    def _from_grouped_arguments(
        func: FunctionSchema,
        arguments: Sequence[Union[Argument, TensorOptionsArguments, ThisArgument]],
        *,
        faithful: bool
    ) -> 'CppSignature':
        if faithful:
            # Immediately, manually do overload disambiguation, by
            # dropping defaults from the faithful signature.  In
            # principle, we should be able to do this at some later
            # point in time with other overload disambiguation
            cpp_grouped_arguments = tuple(
                cpp.argument_faithful(a).no_default() for a in arguments
            )
        else:
            cpp_grouped_arguments = tuple(
                cpp.argument(a) for a in arguments
            )
        return CppSignature(
            func=func,
            _cpp_grouped_arguments=cpp_grouped_arguments,
            _cpp_returns_type=cpp.returns_type(func.returns),
        )

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
    def from_schema(func: FunctionSchema, *, method: bool) -> 'CppSignatureGroup':
        grouped_arguments = cpp.group_arguments(func, method=method)
        faithful_signature: Optional[CppSignature]
        if any(isinstance(a, TensorOptionsArguments) for a in grouped_arguments):
            faithful_signature = CppSignature._from_grouped_arguments(func, grouped_arguments, faithful=True)
        else:
            faithful_signature = None
        signature = CppSignature._from_grouped_arguments(func, grouped_arguments, faithful=False)
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
        )

# ------------------------------------------------------------------- #

#                   dispatcher/legacy_dispatcher types

# ------------------------------------------------------------------- #

@dataclass(frozen=True)
class DispatcherExpr:
    type: str
    expr: str

@dataclass(frozen=True)
class LegacyDispatcherExpr:
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
class LegacyDispatcherArgument:
    type: str
    name: str
    # Legacy dispatcher arguments have defaults for some reasons (e.g.,
    # the function prototypes in CPUType.h are defaulted).  There isn't
    # really any good reason to do this, as these functions are only
    # ever called from a context where all defaulted arguments are
    # guaranteed to be given explicitly.
    # TODO: Remove this
    default: Optional[str]
    argument: Union[Argument, TensorOptionsArguments]

    # Convention here is swapped because arguably legacy
    # dispatcher shouldn't have defaults...
    def __str__(self) -> str:
        return f"{self.type} {self.name}"

    def str_with_default(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

# Functions only, no types
import tools.codegen.api.cpp as cpp
