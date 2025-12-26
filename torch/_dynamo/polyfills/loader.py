# Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

import builtins
import functools
import heapq
import importlib
import itertools
import operator
import os
import struct
import sys
import traceback
from typing import Any, TYPE_CHECKING

import torch.utils._pytree as python_pytree

from .. import polyfills, trace_rules


if TYPE_CHECKING:
    from types import ModuleType


# See also the TYPE_CHECKING block in torch/_dynamo/polyfills/__init__.py
POLYFILLED_MODULE_NAMES: tuple[str, ...] = (
    "_collections",
    "builtins",
    "functools",
    "heapq",
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

# Track which polyfill modules have been loaded
_loaded_polyfill_modules: set[str] = set()


def _get_polyfill_module(module_name: str) -> "ModuleType":
    """Load a polyfill module by name."""
    return importlib.import_module(f".{module_name}", package=polyfills.__name__)


def _unregister_from_builtin_ids(polyfill_module: "ModuleType") -> None:
    """Unregister polyfilled functions from _builtin_function_ids."""
    for polyfill_name in polyfill_module.__all__:
        polyfill_handler = getattr(polyfill_module, polyfill_name)
        original_fn = polyfill_handler.__torch_dynamo_original__
        fn_id = id(original_fn)
        if fn_id in trace_rules._builtin_function_ids:
            trace_rules._builtin_function_ids.remove(fn_id)


def _load_polyfill_module(module_name: str) -> "ModuleType":
    """Load a polyfill module and register its handlers."""
    if module_name in _loaded_polyfill_modules:
        return _get_polyfill_module(module_name)

    _loaded_polyfill_modules.add(module_name)
    polyfill_module = _get_polyfill_module(module_name)
    _unregister_from_builtin_ids(polyfill_module)
    return polyfill_module


# Registry mapping id(original_fn) -> polyfill module name
# This is built without importing polyfill modules
_LAZY_POLYFILL_REGISTRY: dict[int, str] = {}


def _build_lazy_registry() -> None:
    """
    Build a registry mapping original function IDs to their polyfill module names.
    This allows lazy loading of polyfill modules only when they are actually used.
    """
    # Mapping from polyfill module name to (source_module, function_names)
    # This defines which functions each polyfill module handles
    polyfill_specs: list[tuple[str, Any, list[str]]] = [
        # (polyfill_module_name, source_module, function_names)
        ("builtins", builtins, ["all", "any", "enumerate", "sum"]),
        ("functools", functools, ["reduce"]),
        (
            "heapq",
            heapq,
            [
                "_heapify_max",
                "_heappop_max",
                "_heapreplace_max",
                "heapify",
                "heappop",
                "heappush",
                "heappushpop",
                "heapreplace",
                "merge",
                "nlargest",
                "nsmallest",
            ],
        ),
        (
            "itertools",
            itertools,
            [
                "accumulate",
                "chain",
                "compress",
                "cycle",
                "dropwhile",
                "filterfalse",
                "islice",
                "tee",
                "zip_longest",
                "pairwise",
            ],
        ),
        ("operator", operator, ["attrgetter", "itemgetter", "methodcaller", "countOf"]),
        ("os", os, ["fspath"]),
        ("struct", struct, ["pack", "unpack"]),
        ("sys", sys, ["intern", "getrecursionlimit"]),
        ("traceback", traceback, ["extract_tb", "clear_frames"]),
    ]

    # Handle conditional sys functions
    if hasattr(sys, "get_int_max_str_digits"):
        polyfill_specs.append(("sys", sys, ["get_int_max_str_digits"]))

    # Handle itertools.chain.from_iterable specially
    if hasattr(itertools.chain, "from_iterable"):
        _LAZY_POLYFILL_REGISTRY[id(itertools.chain.from_iterable)] = "itertools"

    # Build the registry
    for polyfill_module_name, source_module, function_names in polyfill_specs:
        for fn_name in function_names:
            fn = getattr(source_module, fn_name, None)
            if fn is not None:
                _LAZY_POLYFILL_REGISTRY[id(fn)] = polyfill_module_name


# Build the registry at import time
_build_lazy_registry()


def _ensure_polyfill_loaded(fn: Any) -> bool:
    """
    Check if a function has a polyfill and load it if needed.
    Returns True if a polyfill was loaded, False otherwise.
    """
    fn_id = id(fn)
    if fn_id in _LAZY_POLYFILL_REGISTRY:
        module_name = _LAZY_POLYFILL_REGISTRY[fn_id]
        if module_name not in _loaded_polyfill_modules:
            _load_polyfill_module(module_name)
            return True
    return False


def get_polyfilled_modules() -> tuple["ModuleType", ...]:
    """
    Get all polyfill modules, loading them lazily.
    This is provided for backwards compatibility.
    """
    # Load all modules that haven't been loaded yet
    for module_name in POLYFILLED_MODULE_NAMES:
        if module_name not in _loaded_polyfill_modules:
            _load_polyfill_module(module_name)
    return tuple(
        _get_polyfill_module(module_name) for module_name in POLYFILLED_MODULE_NAMES
    )


# These modules need to be loaded eagerly because they handle
# PyTorch internal functions that require immediate registration:
# - fx: _fx_map_arg, _fx_map_aggregate
# - tensor: torch.Tensor._make_subclass
# - torch_c_nn: torch._C._nn functions
# - _collections: _collections._count_elements (optional, may not exist)
# - pytree: torch.utils._pytree functions (conditional)
_EAGER_LOAD_MODULES: frozenset[str] = frozenset(
    {"fx", "tensor", "torch_c_nn", "_collections"}
)
if python_pytree._cxx_pytree_dynamo_traceable:
    _EAGER_LOAD_MODULES = _EAGER_LOAD_MODULES | frozenset({"pytree"})


def _load_eager_modules() -> None:
    """Load polyfill modules that must be loaded eagerly."""
    for module_name in _EAGER_LOAD_MODULES:
        if module_name in POLYFILLED_MODULE_NAMES:
            _load_polyfill_module(module_name)


# Load eager modules at import time
_load_eager_modules()

# Keep POLYFILLED_MODULES for backwards compatibility
# It will lazily load all modules when accessed via __getattr__
POLYFILLED_MODULES: tuple["ModuleType", ...] = ()


def __getattr__(name: str) -> Any:
    if name == "POLYFILLED_MODULES":
        return get_polyfilled_modules()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
