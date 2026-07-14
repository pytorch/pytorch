"""Runtime and dispatcher helpers for optional FlyDSL native operators."""

from __future__ import annotations

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

# RMSNorm backward first shipped in FlyDSL 0.2.3. The copied kernel is also
# validated against the current 0.2.4 API. Unknown versions fall back to ATen
# unless a developer explicitly sets TORCH_NATIVE_SKIP_VERSION_CHECK=1.
_FLYDSL_REQUIRED_VERSIONS: set[Version] = {
    Version("0.2.3"),
    Version("0.2.4"),
}


def _flydsl_runtime_unavailable_reason() -> str | None:
    flydsl_spec = importlib.util.find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Looking up ``flydsl._mlir`` directly imports the parent package. Search
    # the package paths instead so ``import torch`` remains fork-safe and lazy.
    mlir_spec = importlib.machinery.PathFinder.find_spec(
        "_mlir", list(flydsl_spec.submodule_search_locations)
    )
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir` (runtime is not built)"
    return None


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """Check FlyDSL availability without importing or initializing the GPU."""

    if not _cuda.is_built():
        return False, None

    import torch

    if torch.version.hip is None:
        return False, None

    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.info("FlyDSL native operators are disabled: %s", reason)
        return False, None
    return True, _available_version("flydsl")


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> Version | None:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_ok() -> bool:
    _, version = _check_runtime_available()
    if check_native_version_skip():
        return True
    if version in _FLYDSL_REQUIRED_VERSIONS:
        return True
    if (
        version is not None
        and Version(version.base_version) in _FLYDSL_REQUIRED_VERSIONS
    ):
        return True

    log.info(
        "FlyDSL version %s is not known-good (supported: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 only for local experiments",
        version,
        _FLYDSL_REQUIRED_VERSIONS,
    )
    return False


def deregister_op_overrides() -> None:
    """Temporarily deregister all FlyDSL overrides."""

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
    """Register an override only when the known-good FlyDSL runtime exists."""

    available, _ = _check_runtime_available()
    if not available or check_native_jit_disabled() or not _version_is_ok():
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


dsl_registry.register_dsl("flydsl", cast(DSLModuleProtocol, sys.modules[__name__]))
