from enum import Enum
from typing import Type

from torch import jit, nn, ScriptDict, ScriptFunction, ScriptList

from typing_extensions import assert_never, assert_type


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
assert_type(jit.script(nn.functional.relu), ScriptFunction)

# can't script nn.Module class
assert_never(jit.script(nn.Linear))
