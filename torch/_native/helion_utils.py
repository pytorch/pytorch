import functools
import logging
import sys
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


_HELION_DSL_NAME = "helion"
# Minimum supported Helion: the latest release verified with the native DSL
# kernels (older releases carry the same APIs but are untested).
_HELION_MINIMUM_VERSION = Version("1.2.0")
# HELION_BACKEND name -> (pip package, import module); extend to add more,
# e.g. "cute": ("nvidia-cutlass-dsl", "cutlass").
_HELION_BACKENDS: dict[str, tuple[str, str]] = {"triton": ("triton", "triton")}
_DEFAULT_HELION_BACKEND = "triton"


def _chosen_backend() -> str:
    """Selected Helion lowering backend (HELION_BACKEND env, default triton)."""
    import os

    return os.getenv("HELION_BACKEND", _DEFAULT_HELION_BACKEND)


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    if not _cuda.is_built():
        return (False, None)

    import torch

    if torch.version.hip is not None:
        return (False, None)

    backend = _chosen_backend()
    dep = _HELION_BACKENDS.get(backend)
    if dep is None:
        log.info(
            "Helion native DSL ops support backends %s; HELION_BACKEND=%s",
            tuple(_HELION_BACKENDS),
            backend,
        )
        return (False, None)

    reason = _unavailable_reason([("helion", "helion"), dep])
    if reason is not None:
        log.info(
            "Helion native DSL ops require optional packages `helion` and `%s`; %s",
            dep[0],
            reason,
        )
        return (False, None)
    return (True, _available_version("helion"))


def runtime_available() -> bool:
    # Package presence only, like triton/cutedsl; version gated in register_op_override.
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> Version | None:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_sufficient() -> bool:
    _, version = _check_runtime_available()
    if version is not None and (
        version >= _HELION_MINIMUM_VERSION or check_native_version_skip()
    ):
        return True

    log.info(
        "helion version %s is not sufficient (>= %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _HELION_MINIMUM_VERSION,
    )
    return False


def deregister_op_overrides() -> None:
    _deregister_op_overrides_impl(disable_dsl_names=_HELION_DSL_NAME)


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
    available, _ = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _HELION_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


dsl_registry.register_dsl(
    _HELION_DSL_NAME, cast(DSLModuleProtocol, sys.modules[__name__])
)
