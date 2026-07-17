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
_HELION_MINIMUM_VERSION = Version("1.2.1.dev75")


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    if not _cuda.is_built():
        return (False, None)

    import torch

    if torch.version.hip is not None:
        return (False, None)

    reason = _unavailable_reason([("helion", "helion")])
    if reason is not None:
        log.info("Helion native DSL ops require optional package `helion`; %s", reason)
        return (False, None)
    return (True, _available_version("helion"))


def runtime_available() -> bool:
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


def _sync_auxiliary_overrides() -> None:
    from .ops.cross_entropy.helion_impl import (
        _install_autograd_fallthrough,
        _uninstall_autograd_fallthrough,
    )
    from .registry import _graphs

    nodes = _graphs.get(("cross_entropy_loss", "CUDA"), ())
    if any(node.dsl_name == _HELION_DSL_NAME and node.active for node in nodes):
        _install_autograd_fallthrough()
    else:
        _uninstall_autograd_fallthrough()


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
    if not available or check_native_jit_disabled() or not _version_is_sufficient():
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
