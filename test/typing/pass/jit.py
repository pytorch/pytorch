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
scripted_module = jit.script(nn.Linear(2, 2))
assert_type(scripted_module, jit.RecursiveScriptModule)

# ScriptMethod
# NOTE: Generic usage only possible with Python 3.9
forward: ScriptMethod = scripted_module.forward

# ScripFunction
# NOTE: can't use assert_type because of parameter names
# NOTE: Generic usage only possible with Python 3.9
relu: ScriptFunction = jit.script(nn.functional.relu)

# can't script nn.Module class
assert_never(jit.script(nn.Linear))
