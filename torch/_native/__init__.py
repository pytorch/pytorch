import os
import warnings
from functools import cache
from typing import cast

import torch

# This handles collecting registration of all native ops
# Also need to import DSL utils to make sure DSL registration is ok
from . import cutedsl_utils, dsl_registry, ops, registry, triton_utils


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
def _load_native_aot_lib() -> str | None:
    """Load the native-AOT kernel library, if present.

    The library's static initializers register kernels on the
    at::native DispatchStubs in the generated NativeAotStubs.h; without
    it the stubs have no kernel and the generated wrappers run the
    stock impls. Loading must not initialize CUDA: kernel cubins load
    lazily on first use.

    Search order: $TORCH_NATIVE_AOT_LIB, torch/lib (installed),
    build/native_aot (development). Returns the loaded path or None.
    """
    if os.getenv("TORCH_DISABLE_NATIVE_AOT") == "1":
        return None
    torch_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = "libtorch_native_aot_cuda.so"
    candidates = [
        os.getenv("TORCH_NATIVE_AOT_LIB"),
        os.path.join(torch_dir, "lib", name),
        os.path.join(os.path.dirname(torch_dir), "build", "native_aot", name),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                # Registers the lib in torch.ops.loaded_libraries as well
                # as dlopen-ing it (static initializers do the stub
                # registration; there are no TORCH_LIBRARY defs inside).
                torch.ops.load_library(path)
            except OSError as e:
                warnings.warn(f"Failed to load native-AOT library {path}: {e}")
                return None
            return path
    return None


_load_native_aot_lib()


def set_aot_enabled(enabled: bool) -> None:
    """Toggle at::globalContext().allowNativeAot(): the switch every
    generated stub consultation checks, so False gives stock-aten
    behavior even with the AOT kernel library loaded."""
    torch._C._set_native_aot_enabled(enabled)


def aot_enabled() -> bool:
    return torch._C._get_native_aot_enabled()
