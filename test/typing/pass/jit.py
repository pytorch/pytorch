from enum import Enum

from typing import Callable, Type, TypeVar

from torch import jit, nn, ScriptDict, ScriptFunction, ScriptList, Tensor
from typing_extensions import assert_never, assert_type, ParamSpec

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


# Script Enum
assert_type(jit.script(Color), Type[Color])

# ScriptDict
assert_type(jit.script({1: 1}), ScriptDict)

# ScriptList
assert_type(jit.script([0]), ScriptList)

# ScriptModule
assert_type(jit.script(nn.Linear(2, 2)), jit.RecursiveScriptModule)

# ScripFunction
relu: Callable[[Tensor, bool], Tensor] = nn.functional.relu  # forget argument names
assert_type(jit.script(relu), ScriptFunction[[Tensor, bool], Tensor])

# can't script nn.Module class
assert_never(jit.script(nn.Linear))
