import contextlib
import os
import warnings
from functools import cache
from typing import cast

import torch

# This handles collecting registration of all native ops
# Also need to import DSL utils to make sure DSL registration is ok
from . import (
    cutedsl_utils,
    dsl_registry,
    flydsl_utils,
    helion_utils,
    ops,
    registry,
    triton_utils,
)


@cache
def get_user_ordering_fn() -> registry.UserOrderingFn | None:
    """
    Get a user-supplied graph-ordering function if specified.

    Pass in a `package.submodule.fn` string to the env variable
    `TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN` that implements the
    calling API described in `torch/_native/README.md`. This function
    must be part of an importable path.

    Return either the imported function or `None`
    """
    env_var = os.getenv("TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN")

    if not env_var:
        return None

    try:
        import importlib

        # Split into "package.submodule.fn_name
        module_name, fn_name = env_var.rsplit(".", 1)

        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)

        if not callable(fn):
            raise TypeError(f"{env_var} does not describe a callable function")

        # Cast needed: getattr returns object, but we've verified fn is callable with correct signature
        return cast(registry.UserOrderingFn, fn)
    except Exception as e:
        raise ValueError(
            f"Could not resolve {env_var} into an importable & callable function"
        ) from e


user_order_fn = get_user_ordering_fn()
if user_order_fn:
    registry.reorder_graphs_from_user_function(user_order_fn)


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators,  other operators may also be overridden\\.",
        category=UserWarning,
    )
    registry._register_all_overrides()


@cache
def _native_aot_embedded() -> bool:
    """True iff this build ships AOT kernels for its declared ops.

    Answers what the build contains, not whether the kernels may run: a build made
    without stage 2 has none, and every op falls back to its ordinary aten
    implementation. Says nothing about set_aot_enabled(), which can mask kernels
    that are present. Touches no device.
    """
    try:
        return any(
            schema.name.startswith("_native_aot::")
            for schema in torch._C._jit_get_all_schemas()
        )
    except Exception:
        return False


def set_aot_enabled(enabled: bool) -> None:
    """Turn the embedded AOT kernels off (or back on) for this process.

    With them off every op computes the same answers through its ordinary aten
    implementation, so this is the switch to use when comparing against stock
    PyTorch. Takes effect immediately, on calls already in flight.

    Does NOT reach ops whose kernels ARE the implementation rather than a faster
    route to the same answer; use _unconditional_masked() for one of those."""
    torch._C._set_native_aot_enabled(enabled)


def aot_enabled() -> bool:
    return torch._C._get_native_aot_enabled()


def _apply_env_opt_out() -> None:
    """Honour TORCH_DISABLE_NATIVE_AOT=1 at import, once.

    Separate from _native_aot_embedded(), which is cached and answers a question about
    the build; this reads configuration and changes state. Gated on that answer so the
    switch keeps its default on a build that ships no kernels, where it means nothing.
    """
    if os.getenv("TORCH_DISABLE_NATIVE_AOT") == "1" and _native_aot_embedded():
        set_aot_enabled(False)


_apply_env_opt_out()


@contextlib.contextmanager
def _unconditional_masked():
    """PRIVATE, for reference computations only: also mask the overrides and AOT
    kernels that are exempt from the user-facing switches (declaration UNCONDITIONAL
    or register_op_override(unconditional_override=True)).

    Those two mechanisms otherwise have no off state, leaving a numerics test no way
    to obtain stock aten values. Not a user knob: masking an unconditional override
    changes what the op computes. Combine with python_native.<dsl>.disabled(), which
    masks everything else; this only lifts the exemptions."""
    from torch._native.registry import _set_mask_unconditional

    # One flag serves both mechanisms (see registry._unconditional_is_masked), and
    # setting it also rebuilds the routers, so overrides and AOT kernels cannot
    # disagree about whether the exemption is lifted.
    previous = _set_mask_unconditional(True)
    try:
        yield
    finally:
        _set_mask_unconditional(previous)
