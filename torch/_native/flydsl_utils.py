"""Runtime and dispatcher helpers for optional FlyDSL native operators."""

import functools
import logging
import sys
from os import environ as _environ
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

_FLYDSL_DSL_NAME = "flydsl"

# The kernels this gate protects -- see ops/norm/flydsl_rmsnorm_fwd.py -- are
# written against the FlyDSL 0.3.x flydsl.expr.gpu.shuffle_xor interface. Other
# versions fall back to ATen unless a developer explicitly sets
# TORCH_NATIVE_SKIP_VERSION_CHECK=1.
_FLYDSL_SUPPORTED_RELEASES = ((0, 3),)


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """Check FlyDSL availability without importing or initializing the GPU."""

    if not _cuda.is_built():
        return False, None

    # FlyDSL targets ROCm only; CUDA builds use the ATen implementation.
    import torch

    if torch.version.hip is None:
        return False, None

    reason = _unavailable_reason([("flydsl", "flydsl")])
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
    available, version = _check_runtime_available()
    if not available:
        # _check_runtime_available already logged why, if there was anything to
        # say. Falling through would report the absent version as the problem
        # and send the reader after a package that is not even installed.
        return False
    if check_native_version_skip():
        return True
    if (
        version is not None
        and version.release[:2] in _FLYDSL_SUPPORTED_RELEASES
        and not version.is_prerelease
    ):
        return True

    supported = ", ".join(
        ".".join(map(str, release)) + ".x" for release in _FLYDSL_SUPPORTED_RELEASES
    )
    log.info(
        "FlyDSL version %s is not supported (supported stable releases: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        supported,
    )
    return False


@functools.cache
def _get_flydsl_device_arch(device_index: int) -> str | None:
    """Return the cached ROCm architecture reported for a device."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_index)
            arch = getattr(props, "gcnArchName", None)
            if arch:
                return str(arch).split(":", 1)[0]
    except Exception:
        log.debug("Could not determine FlyDSL GPU arch", exc_info=True)
    return None


def _resolve_rocm_arch(device_index: int) -> str | None:
    """Return the gfx name to compile for, or None if it cannot be determined.

    FLYDSL_GPU_ARCH wins, then HSA_OVERRIDE_GFX_VERSION, then the device's
    cached gcnArchName. Environment overrides are read on every call.
    """
    env = _environ.get("FLYDSL_GPU_ARCH")
    if env:
        return env.split(":", 1)[0]

    hsa = _environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa:
        if hsa.startswith("gfx"):
            return hsa
        if hsa.count(".") == 2:
            major, minor, stepping = hsa.split(".")
            try:
                return f"gfx{major}{minor}{int(stepping):x}"
            except ValueError:
                log.debug("Ignoring invalid HSA_OVERRIDE_GFX_VERSION=%s", hsa)

    return _get_flydsl_device_arch(device_index)


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
