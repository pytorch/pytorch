# Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

import importlib
import sys
from typing import TYPE_CHECKING

import torch.utils._pytree as python_pytree

from .. import polyfills, trace_rules


if TYPE_CHECKING:
    from types import ModuleType


_loaded_polyfill_module_names: set[str] = set()


def _load_polyfill_module(submodule: str) -> "ModuleType":
    if submodule in _loaded_polyfill_module_names:
        raise AssertionError(f"Polyfill module {submodule} has already been loaded")

    polyfill_module = importlib.import_module(
        f".{submodule}", package=polyfills.__name__
    )
    _loaded_polyfill_module_names.add(submodule)

    # Unregister the builtin functions from _builtin_function_ids to let them to be
    # dispatched with the appropriate VariableTracker type. Otherwise, they will be
    # dispatched with BuiltinVariable if present in _builtin_function_ids.
    for polyfill_name in polyfill_module.__all__:
        polyfill_handler = getattr(polyfill_module, polyfill_name)
        original_fn = polyfill_handler.__torch_dynamo_original__
        trace_rules._builtin_function_ids.remove(id(original_fn))

    return polyfill_module


def _load_pytree_polyfill_module() -> None:
    global POLYFILLED_MODULES
    if "pytree" in _loaded_polyfill_module_names:
        return
    POLYFILLED_MODULES = (*POLYFILLED_MODULES, _load_polyfill_module("pytree"))


# See also the TYPE_CHECKING block in torch/_dynamo/polyfills/__init__.py
POLYFILLED_MODULE_NAMES: tuple[str, ...] = (
    "_collections",
    "builtins",
    "copy",
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
    if "optree" in sys.modules:
        POLYFILLED_MODULE_NAMES += ("pytree",)
    else:
        trace_rules.add_module_init_func("optree", _load_pytree_polyfill_module)

POLYFILLED_MODULES: tuple["ModuleType", ...] = tuple(
    _load_polyfill_module(submodule) for submodule in POLYFILLED_MODULE_NAMES
)
