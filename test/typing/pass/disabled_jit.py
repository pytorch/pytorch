from enum import Enum
from typing import Type, TypeVar
from typing_extensions import assert_never, assert_type, ParamSpec

import pytest

from torch import jit, nn, ScriptDict, ScriptFunction, ScriptList

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

# ScripFunction
# NOTE: can't use assert_type because of parameter names
# NOTE: Generic usage only possible with Python 3.9
relu: ScriptFunction = jit.script(nn.functional.relu)

# can't script nn.Module class
with pytest.raises(RuntimeError):
    assert_never(jit.script(nn.Linear))
