import functools
import importlib.machinery
import importlib.util
import logging
import sys
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
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


_FLYDSL_DSL_NAME = "flydsl"
_FLYDSL_REQUIRED_VERSIONS: set[Version] = {
    Version("0.2.2"),
}


def _flydsl_runtime_unavailable_reason() -> str | None:
    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl` (importlib.util.find_spec(flydsl) failed)"

    # importlib.util.find_spec("flydsl._mlir") imports the parent package as a
    # side effect. Query the package paths directly so `import torch` stays lazy.
    mlir_spec = importlib.machinery.PathFinder.find_spec(
        "_mlir",
        list(flydsl_spec.submodule_search_locations),
    )
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir` (runtime bindings are not built)"

    return None


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if flydsl is available.

    NOTE: Doesn't import flydsl at this point.
    """
    # Skip all checks if running on CPU-only binary.
    if not _cuda.is_built():
        return (False, None)

    import torch

    if torch.version.hip is None:
        return (False, None)

    reason = _flydsl_runtime_unavailable_reason()
    if reason is None:
        available = True
        version = _available_version("flydsl")
    else:
        # info, not warning: missing optional deps is the common case on stock
        # builds and we don't want to spam stderr on `import torch`. Surface
        # it via TORCH_LOGS=+native_dsl when diagnosing why an override is
        # silent.
        log.info("FlyDSL operators require optional package `flydsl`; %s", reason)
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
    if check_native_version_skip():
        return True
    if version in _FLYDSL_REQUIRED_VERSIONS:
        return True
    if version is not None and Version(version.base_version) in _FLYDSL_REQUIRED_VERSIONS:
        return True

    log.info(
        "flydsl version %s is not known-good (ok: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _FLYDSL_REQUIRED_VERSIONS,
    )
    return False


def deregister_op_overrides() -> None:
    """
    Deregister all ops through FlyDSL.
    """
    _deregister_op_overrides_impl(disable_dsl_names=_FLYDSL_DSL_NAME)


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
    _register_op_override_impl.
    """
    available, _ = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_ok():
        return

    _register_op_override_impl(
        _FLYDSL_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry.
# Note: Import-time registration ensures DSL is available when module is loaded.
dsl_registry.register_dsl("flydsl", cast(DSLModuleProtocol, sys.modules[__name__]))
