from typing import Callable

import torch
from .function_dispatcher import _ATENLIB_FUNCTIONS, _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE

CUSTOM_OP_OVERLOADS = []


def custom_op_overload(schema: str):
    def inner(f: Callable):
        # TODO: Refactor the Library API so this is less rage inducing
        # TODO: Perhaps the namespace should be directly based on Python
        # module
        if "::" in schema:
            ns = schema.split("::", 2)[0]
        else:
            ns = "contrib"
        # TODO: Library doesn't allow FRAGMENT, need to allow it
        lib = torch.library.Library(ns, "DEF")
        name = lib.define(schema)
        if "::" in name:
            name = name.split("::", 2)[1]
        lib.impl(name, f, "CompositeExplicitAutograd")
        CUSTOM_OP_OVERLOADS.append(lib)
        return getattr(getattr(torch.ops, ns), name)

    return inner


def _register_custom_op_overload(
    op_overload: torch._ops.OpOverload, exporter_key: str, exporter: Callable
) -> None:
    assert op_overload not in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
    assert exporter_key not in _ATENLIB_FUNCTIONS
    _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[op_overload] = exporter_key
    _ATENLIB_FUNCTIONS[exporter_key] = exporter
