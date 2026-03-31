import functools
import logging

from packaging.version import Version

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .registry import (
    _OpFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)


_CUTEDSL_DSL_NAME = "cutedsl"
_CUTEDSL_REQUIRED_VERSIONS: set[Version] = {
    # Current version - Note Version.from_part(release=(4.4.1)) is better
    #                   but > v26 of packaging.
    Version(f"{4}.{4}.{1}"),
    Version(f"{4}.{4}.{2}"),
}


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        log.warning(
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

    log.warning(
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
    impl: _OpFn,
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
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )
