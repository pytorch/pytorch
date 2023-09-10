from enum import Enum
from typing import Type, TypeVar

from typing_extensions import ParamSpec, assert_never, assert_type

from torch import ScriptDict, ScriptFunction, ScriptList, Tensor, jit, nn

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
# NOTE: can't use assert_type because of parameter names
relu: ScriptFunction[[Tensor, bool], Tensor] = jit.script(nn.functional.relu)

# can't script nn.Module class
assert_never(jit.script(nn.Linear))
