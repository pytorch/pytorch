from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

import torch.library


P = ParamSpec("P")
R = TypeVar("R")

_OpOverrideFn = Callable[Concatenate[torch.DispatchKeySet, P], R]
_OpReplaceFn = Callable[P, R]

_OpFn = _OpOverrideFn | _OpReplaceFn


libs = {}


def _get_library(lib_symbol: str, dispatch_key: str) -> torch.library.Library:
    """
    Return a `torch.library.Library` instance unique to the passed
    (lib_symbol, dispatch_key) pair. Create a new instance if necessary.
    """
    global libs

    if (lib_symbol, dispatch_key) not in libs:
        libs[(lib_symbol, dispatch_key)] = torch.library.Library(
            lib_symbol, "IMPL", dispatch_key
        )

    return libs[(lib_symbol, dispatch_key)]


def _register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    impl: _OpOverrideFn | _OpReplaceFn,
    *,
    allow_multiple_override=False,
    unconditional_override=False,
) -> None:
    """
    Register a passed override function to the dispatcher, based on the
    passed lib and op symbols, and the dispatch key.

    lib_symbol: str - library yourve overriding symbols in (generally "aten")
    op_symbol: str - name of the op you're overriding
    dispatch_key: str - dispatch key to override
    impl: Fn - implementation for the override
    allow_multiple_override: bool - allow overriding an existing override
    unconditional_override: bool - Impl doesn't have a fallback, and doesn't require
                                   torch.DispatchKeySet as the first argument.
    """
    lib = _get_library(lib_symbol, dispatch_key)

    lib.impl(
        op_symbol,
        impl,
        dispatch_key,
        with_keyset=(not unconditional_override),
        allow_override=allow_multiple_override,
    )
