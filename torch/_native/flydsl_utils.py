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

# The vendored kernels are validated against the FlyDSL 0.3.x API. 0.3.0
# removed flydsl.expr.vector, which the RMSNorm kernel imported from, so
# earlier versions cannot load it at all. They fall back to ATen unless a
# developer explicitly sets TORCH_NATIVE_SKIP_VERSION_CHECK=1.
_FLYDSL_SUPPORTED_RELEASE = (0, 3)


def _flydsl_runtime_unavailable_reason() -> None | str:
    flydsl_spec = _find_spec("flydsl")
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Looking up ``flydsl._mlir`` directly imports the parent package. Search
    # the package paths instead so ``import torch`` remains fork-safe and lazy.
    mlir_spec = _PathFinder.find_spec(
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
    _, version = _check_runtime_available()
    if check_native_version_skip():
        return True
    # FlyDSL currently ships as dev tags (0.3.0.dev765 at the time of writing).
    # Its 0.3.x line is API-compatible with these vendored kernels.
    if version is not None and version.release[:2] == _FLYDSL_SUPPORTED_RELEASE:
        return True

    log.info(
        "FlyDSL version %s is not supported (expected %s.x); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        ".".join(map(str, _FLYDSL_SUPPORTED_RELEASE)),
    )
    return False


def _arch_from_hsa_override(value: str) -> str | None:
    """Parse HSA_OVERRIDE_GFX_VERSION, which is either a gfx name or M.m.s.

    The stepping is hexadecimal: 9.0.10 is gfx90a, not gfx9010.
    """
    if value.startswith("gfx"):
        return value.split(":", 1)[0]
    parts = value.split(".")
    if len(parts) != 3:
        return None
    major, minor, stepping = parts
    try:
        return f"gfx{major}{minor}{int(stepping):x}"
    except ValueError:
        log.debug("Ignoring invalid HSA_OVERRIDE_GFX_VERSION=%s", value)
        return None


@functools.lru_cache
def _resolve_rocm_arch(device_index: int) -> str | None:
    """Return the gfx name FlyDSL should compile for on ``device_index``.

    Shared so the eager overrides and the Inductor FlyDSL templates cannot
    disagree about what they are compiling for -- they run in different
    processes and only one of them can import flydsl, so neither can rely on
    the runtime's own resolver. Deliberately reads only the environment and
    torch's device properties.
    """
    env = _environ.get("FLYDSL_GPU_ARCH")
    if env:
        return env.split(":", 1)[0]

    hsa = _environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa:
        arch = _arch_from_hsa_override(hsa)
        if arch is not None:
            return arch

    import torch

    # gcnArchName carries feature flags, e.g. "gfx950:sramecc+:xnack-".
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
        return props.gcnArchName.split(":", 1)[0]
    except Exception:
        log.debug("Could not determine FlyDSL GPU arch", exc_info=True)
        return None


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
