import functools
import logging
import sys
import warnings
from collections.abc import Callable
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .dsl_registry import dsl_registry, DSLModuleProtocol
from .registry import (
    _OpCondFn,
    _OpImplFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)

_pending_cuda_registrations: list[Callable[[], None]] = []


def register_cuda_overrides(
    register: Callable[[], None], arch_majors: tuple[int, ...]
) -> None:
    r"""Install a CuTeDSL registration group after supported CUDA devices initialize."""
    if not runtime_available() or check_native_jit_disabled() or not _version_is_ok():
        return

    import torch

    from . import registry

    def install() -> None:
        native = sys.modules["torch._native"]
        if not native._registration_complete:
            _pending_cuda_registrations.append(install)
            return
        if not any(
            torch.cuda.get_device_capability(i)[0] in arch_majors
            for i in range(torch.cuda.device_count())
        ):
            return
        before = {key: len(graph) for key, graph in registry._graphs.items()}
        register()
        keys = {
            key
            for key, graph in registry._graphs.items()
            if len(graph) != before.get(key, 0)
        }
        if native.user_order_fn is not None:
            registry._apply_graph_transformation(
                native.user_order_fn, keys_to_process=keys
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Warning only once for all operators.*",
                category=UserWarning,
            )
            for key in keys:
                registry._register_overrides_from_graph(
                    *key, registry._graphs[key], filter_state=registry._filter_state
                )

    torch.cuda._lazy_call(install)


def _flush_cuda_registrations() -> None:
    pending = list(_pending_cuda_registrations)
    _pending_cuda_registrations.clear()
    for install in pending:
        install()


_CUTEDSL_DSL_NAME = "cutedsl"
# Reductions require FastDivmodDivisorV2 from 4.6.2; Version.from_part needs packaging >26.
_CUTEDSL_REQUIRED_VERSIONS: set[Version] = {
    Version("4.6.2"),
    Version("4.7.1"),
    Version("4.8.0"),
    Version("4.8.1"),
}


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    # CuTeDSL kernels require genuine NVIDIA GPUs; on ROCm/HIP builds the
    # CUDA compatibility layer reports is_built()=True but cuInit() will fail
    # because no NVIDIA driver is present.
    import torch

    if torch.version.hip is not None:
        return (False, None)

    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        # info, not warning: missing optional deps is the common case on stock
        # builds and we don't want to spam stderr on `import torch`. Surface
        # it via TORCH_LOGS=+native_dsl when diagnosing why an override is
        # silent.
        log.info(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl` and `apache-tvm-ffi`; "
            "%s",
            reason,
        )
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | Version:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_ok() -> bool:
    _, version = _check_runtime_available()
    if check_native_version_skip() or (version in _CUTEDSL_REQUIRED_VERSIONS):
        return True

    log.info(
        "cutedsl version %s is not known-good (ok: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _CUTEDSL_REQUIRED_VERSIONS,
    )
    return False


def deregister_op_overrides() -> None:
    """
    Deregister all ops through cuteDSL
    """
    _deregister_op_overrides_impl(disable_dsl_names=_CUTEDSL_DSL_NAME)


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    """
    See torch/_native/registry.py for the underlying implementation
    and arguments. This is a thin, DSL-checking wrapper over
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_ok():
        return

    _register_op_override_impl(
        _CUTEDSL_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry
# Note: Import-time registration ensures DSL is available when module is loaded
dsl_registry.register_dsl("cutedsl", cast(DSLModuleProtocol, sys.modules[__name__]))
