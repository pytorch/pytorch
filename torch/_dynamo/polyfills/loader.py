# Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

import functools
import importlib
from typing import TYPE_CHECKING

import torch.utils._pytree as python_pytree

from .. import polyfills, trace_rules


if TYPE_CHECKING:
    from types import ModuleType


# See also the TYPE_CHECKING block in torch/_dynamo/polyfills/__init__.py
POLYFILLED_MODULE_NAMES: tuple[str, ...] = (
    "_collections",
    "builtins",
    "functools",
    "itertools",
    "operator",
    "os",
    "struct",
    "sys",
    "fx",
    "tensor",
    "torch_c_nn",
    "traceback",
)
if python_pytree._cxx_pytree_dynamo_traceable:
    POLYFILLED_MODULE_NAMES += ("pytree",)


@functools.cache
def get_polyfilled_modules() -> tuple["ModuleType", ...]:
    modules = tuple(
        importlib.import_module(f".{submodule}", package=polyfills.__name__)
        for submodule in POLYFILLED_MODULE_NAMES
    )
    # Unregister the builtin functions from _builtin_function_ids to let them to be
    # dispatched with the appropriate VariableTracker type. Otherwise, they will be
    # dispatched with BuiltinVariable if present in _builtin_function_ids.
    for polyfill_module in modules:
        for polyfill_name in polyfill_module.__all__:
            trace_rules._builtin_function_ids.remove(
                id(getattr(polyfill_module, polyfill_name).__torch_dynamo_original__)
            )
    return modules
