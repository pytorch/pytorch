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
    """True iff this libtorch_cuda was linked with AOT kernel artifacts.

    The artifacts' static initializers registered the stub kernels and
    the _native_aot covers custom ops at load time (every shipped
    artifact set registers at least one covers op -- the in-process
    marker probed here). A build whose stage 2 never ran has no
    artifacts, no registrations, and the generated wrapper stubs fall
    through to the stock impls. Nothing here initializes CUDA: kernel
    cubins load lazily on first use.

    TORCH_DISABLE_NATIVE_AOT=1 masks the kernels by flipping the
    Context switch every stub consultation checks; gated coverage then
    declines, so covered calls keep their JIT route.
    """
    try:
        embedded = any(
            schema.name.startswith("_native_aot::")
            for schema in torch._C._jit_get_all_schemas()
        )
    except Exception:
        embedded = False
    if embedded and os.getenv("TORCH_DISABLE_NATIVE_AOT") == "1":
        torch._C._set_native_aot_enabled(False)
    return embedded


_native_aot_embedded()


def set_aot_enabled(enabled: bool) -> None:
    """Toggle at::globalContext().allowNativeAot(): the switch every
    generated stub consultation checks, so False gives stock-aten
    behavior even with the AOT kernel library loaded.

    Does NOT reach ops whose declaration is UNCONDITIONAL: their kernels
    are the implementation rather than a faster route to the same answer.
    Use _unconditional_masked() if you need stock aten for one of those."""
    torch._C._set_native_aot_enabled(enabled)


def aot_enabled() -> bool:
    return torch._C._get_native_aot_enabled()


@contextlib.contextmanager
def _unconditional_masked():
    """PRIVATE, for reference computations only: also mask the overrides and
    AOT kernels that are exempt from the user-facing switches (declaration
    UNCONDITIONAL / register_op_override(unconditional_override=True)).

    Exists because those two mechanisms otherwise have no off state, which
    would leave a numerics test with no way to obtain stock aten values for
    the op -- and a reference silently computed through the override under
    test is worse than no reference. Not a user knob: masking an
    unconditional override changes what the op computes.

    Combine with python_native.<dsl>.disabled(), which masks everything
    else; this only lifts the exemptions."""
    from torch._native.registry import _set_mask_unconditional, _unconditional_is_masked

    # Read both previous values BEFORE mutating either: setting one and then
    # failing to read the other would leave the mask latched on for the rest
    # of the process, silently masking unconditional overrides everywhere.
    previous_jit = _unconditional_is_masked()
    previous_aot = torch._C._get_native_aot_unconditional_masked()
    try:
        _set_mask_unconditional(True)
        torch._C._set_native_aot_unconditional_masked(True)
        yield
    finally:
        torch._C._set_native_aot_unconditional_masked(previous_aot)
        _set_mask_unconditional(previous_jit)
