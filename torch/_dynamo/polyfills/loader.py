# Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

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

POLYFILLED_MODULES: tuple["ModuleType", ...] = tuple(
    importlib.import_module(f".{submodule}", package=polyfills.__name__)
    for submodule in POLYFILLED_MODULE_NAMES
)


# Unregister the builtin functions from _builtin_function_ids to let them to be
# dispatched with the appropriate VariableTracker type. Otherwise, they will be
# dispatched with BuiltinVariable if present in _builtin_function_ids.
for polyfill_module in POLYFILLED_MODULES:
    for polyfill_name in polyfill_module.__all__:
        polyfill_handler = getattr(polyfill_module, polyfill_name)
        original_fn = polyfill_handler.__torch_dynamo_original__
        trace_rules._builtin_function_ids.remove(id(original_fn))
