"""Runtime and dispatcher helpers for optional FlyDSL native operators."""

import functools
import logging
import sys
from importlib.machinery import PathFinder as _PathFinder
from importlib.util import find_spec as _find_spec
from os import environ as _environ
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

# The kernels this gate protects -- see ops/norm/flydsl_rmsnorm_fwd.py -- are
# written against the FlyDSL 0.3.x flydsl.expr.gpu.shuffle_xor interface. Other
# versions fall back to ATen unless a developer explicitly sets
# TORCH_NATIVE_SKIP_VERSION_CHECK=1.
_FLYDSL_SUPPORTED_RELEASE = (0, 3)


def _flydsl_runtime_unavailable_reason() -> str | None:
    # find_spec raises ValueError when `flydsl` sits in sys.modules without a
    # usable __spec__. This runs during `import torch`, so an unusable install
    # has to read as "unavailable" rather than escape as an exception.
    try:
        flydsl_spec = _find_spec("flydsl")
    except (ImportError, ValueError):
        flydsl_spec = None
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Looking up ``flydsl._mlir`` directly imports the parent package. Search
    # the package paths instead so ``import torch`` remains fork-safe and lazy.
    # A bad entry in submodule_search_locations reaches a path entry finder,
    # which can raise for the same reason.
    try:
        mlir_spec = _PathFinder.find_spec(
            "_mlir", list(flydsl_spec.submodule_search_locations)
        )
    except (ImportError, ValueError):
        mlir_spec = None
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir` (runtime is not built)"
    return None


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """Check FlyDSL availability without importing or initializing the GPU."""

    if not _cuda.is_built():
        return False, None

    # FlyDSL targets ROCm only; CUDA builds use the ATen implementation.
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
    available, version = _check_runtime_available()
    if not available:
        # _check_runtime_available already logged why, if there was anything to
        # say. Falling through would report the absent version as the problem
        # and send the reader after a package that is not even installed.
        return False
    if check_native_version_skip():
        return True
    # FlyDSL currently ships as dev tags (0.3.0.dev765 at the time of writing).
    # Its 0.3.x line is API-compatible with the kernels under ops/.
    if version is not None and version.release[:2] == _FLYDSL_SUPPORTED_RELEASE:
        return True

    supported = ".".join(map(str, _FLYDSL_SUPPORTED_RELEASE))
    if version is None:
        # Importable but with no installed distribution -- a source checkout on
        # PYTHONPATH looks like this. Saying "version None" would read as a
        # version mismatch and send the reader looking for the wrong problem.
        log.info(
            "FlyDSL version metadata is missing (expected %s.x); "
            "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to use it anyway",
            supported,
        )
    else:
        log.info(
            "FlyDSL version %s is not supported (expected %s.x); "
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
