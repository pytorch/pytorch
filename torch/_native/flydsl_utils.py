"""Runtime and dispatcher helpers for optional FlyDSL native operators."""

import contextlib as _contextlib
import functools
import logging
import sys
from collections.abc import Iterator as _Iterator
from os import environ as _environ
from threading import RLock as _RLock
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


# Held across a compile pin, and around the one read that a pin can be seen by.
# Reentrant so that a future caller reaching _resolve_rocm_arch from inside a
# pin gets today's answer rather than a deadlock.
_compile_arch_pin = _RLock()


@functools.cache
def _resolve_rocm_arch(device_index: int) -> str | None:
    """Return the gfx name to compile for, or None if it cannot be determined.

    FLYDSL_GPU_ARCH wins, then HSA_OVERRIDE_GFX_VERSION, then the device's
    cached gcnArchName. The normalized result is cached per device, so this
    reads the environment once and holds that answer for the process; taking
    the pin lock keeps that one read from landing inside another device's
    _pinned_compile_arch window and caching its arch forever.
    """
    with _compile_arch_pin:
        env = _environ.get("FLYDSL_GPU_ARCH")
        if env:
            return env.split(":", 1)[0]

        hsa = _environ.get("HSA_OVERRIDE_GFX_VERSION")
        if hsa:
            if hsa.startswith("gfx"):
                return hsa.split(":", 1)[0]
            if hsa.count(".") == 2:
                major, minor, stepping = hsa.split(".")
                try:
                    return f"gfx{major}{minor}{int(stepping):x}"
                except ValueError:
                    log.debug("Ignoring invalid HSA_OVERRIDE_GFX_VERSION=%s", hsa)

    return _get_flydsl_device_arch(device_index)


@_contextlib.contextmanager
def _pinned_compile_arch(arch: str | None) -> _Iterator[None]:
    """Compile for the arch the dispatcher gated on, not the one FlyDSL guesses.

    FlyDSL picks its own compile target in flydsl/runtime/device.py:
    FLYDSL_GPU_ARCH, then HSA_OVERRIDE_GFX_VERSION, then `rocm_agent_enumerator`
    -- and a hard-coded "gfx942" whenever that subprocess raises or times out,
    which it does wherever the tool is not on PATH. That silent fallback
    disagrees with _resolve_rocm_arch, and the HIP loader then rejects the code
    object with hipErrorNoBinaryForGpu.

    One knob still outranks this one: flydsl's own `env.compile.arch`, read
    from a bare `ARCH` (flydsl/utils/env.py declares it with an explicit
    env_var, so it is not prefixed) and consulted first by
    `RocmBackend.detect_target`. A process that already exports `ARCH` defeats
    the pin, and every ROCm host flydsl works on today leaves it unset.

    The handoff is one process-global variable, and flydsl_jit_cache locks per
    specialization so different specializations do compile at once. The lock is
    therefore load-bearing: without it whichever pin finishes first restores the
    variable out from under a compile that is still running, which is the same
    wrong-ISA artifact this is here to prevent. Each specialization compiles
    once, so serializing them costs less than caching a bad code object.
    """
    if not arch:
        yield
        return

    with _compile_arch_pin:
        previous = _environ.get("FLYDSL_GPU_ARCH")
        _environ["FLYDSL_GPU_ARCH"] = arch
        try:
            yield
        finally:
            if previous is None:
                _environ.pop("FLYDSL_GPU_ARCH", None)
            else:
                _environ["FLYDSL_GPU_ARCH"] = previous


@functools.cache
def _is_supported_arch(device_index: int, supported_arches: tuple[str, ...]) -> bool:
    """Whether a device architecture is supported by a FlyDSL operator."""
    arch = _resolve_rocm_arch(device_index)
    return arch in supported_arches


def _fits_int32_buffer_span(rows_m: int, n: int, itemsize: int) -> bool:
    """Whether a contiguous (rows_m, n) tensor fits FlyDSL buffer indexing.

    The byte span gets the wider bound because it lands in the descriptor's
    num_records, which is unsigned; kernels index elements with signed int32.
    """
    int32_max = (1 << 31) - 1
    uint32_max = (1 << 32) - 1
    return (
        0 < rows_m <= int32_max
        and 0 < n <= int32_max
        and rows_m * n * itemsize <= uint32_max
    )


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
