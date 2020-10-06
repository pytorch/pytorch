from tools.codegen.model import *
from dataclasses import dataclass
from typing import Optional, Union, Sequence

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

# Describe a argument (e.g., the x in "f(int x)") in the C++ API
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
    # May also correspond to the implicit *this argument!
    argument: Union[Argument, TensorOptionsArguments, ThisArgument]

    # Default string representation prints the most elaborated form
    # of the formal
    def __str__(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

    # However, you might also find the version with no default useful
    def str_no_default(self) -> str:
        return f"{self.type} {self.name}"

@dataclass(frozen=True)
class CppExpr:
    type: str
    expr: str

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
