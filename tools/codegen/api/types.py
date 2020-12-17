from tools.codegen.model import *
from dataclasses import dataclass
from typing import Optional, Union, TypeVar, List
from enum import Enum

_T = TypeVar('_T')

# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occr.

SpecialArgName = Enum('SpecialArgName', (
    'possibly_redundant_memory_format',
))
ArgName = Union[str, SpecialArgName]

# A CType is short for C++ semantic type.  A CType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#
# CTypes encode C++ type structure as needed for translation.  Right now we
# track references and optional, but don't, for example, track ArrayRef.  If
# you need trnsnlations that know about these types, beef up this data
# structure.

@dataclass(frozen=True)
class BaseCType:
    type: str
    name: ArgName

    def cpp_type(self) -> str:
        return self.type

@dataclass(frozen=True)
class ConstRefCType:
    elem: 'CType'

    def cpp_type(self) -> str:
        return f'const {self.elem.cpp_type()} &'

    @property
    def name(self) -> ArgName:
        return self.elem.name

@dataclass(frozen=True)
class MutRefCType:
    elem: 'CType'

    def cpp_type(self) -> str:
        return f'{self.elem.cpp_type()} &'

    @property
    def name(self) -> ArgName:
        return self.elem.name

@dataclass(frozen=True)
class OptionalCType:
    elem: 'CType'

    def cpp_type(self) -> str:
        return f'c10::optional<{self.elem.cpp_type()}>'

    @property
    def name(self) -> ArgName:
        return self.elem.name

CType = Union[BaseCType, OptionalCType, ConstRefCType, MutRefCType]

# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See tools.codegen.api.translate)

@dataclass(frozen=True)
class Binding:
    name: str
    ctype: CType
    argument: Union[Argument, TensorOptionsArguments, SelfArgument]
    # TODO: maybe don't represent default here
    default: Optional[str] = None

    @property
    def type(self) -> str:
        return self.ctype.cpp_type()

    def no_default(self) -> 'Binding':
        return Binding(
            name=self.name,
            ctype=self.ctype,
            default=None,
            argument=self.argument,
        )

    def decl(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.

@dataclass(frozen=True)
class Expr:
    expr: str
    type: CType

@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    def arguments(self) -> List[Binding]:
        return dispatcher.arguments(self.func)

    def name(self) -> str:
        return dispatcher.name(self.func)

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type()} {name}({args_str})"

    def exprs(self) -> List[Expr]:
        return [Expr(a.name, a.ctype) for a in self.arguments()]

    def returns_type(self) -> str:
        return dispatcher.returns_type(self.func.returns)

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type()} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema) -> 'DispatcherSignature':
        return DispatcherSignature(func)

@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns)} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ', '.join(a.defn() for a in self.arguments())
        return f'{native.returns_type(self.func.returns)} (*)({args_str})'

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func)

    def dispatcher_exprs(self) -> List[Expr]:
        return translate.translate(self.arguments(), dispatcher.arguments(self.func), method=False)

# Functions only, no types
from tools.codegen.api import dispatcher, native, translate
