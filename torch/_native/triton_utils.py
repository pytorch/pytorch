import functools
import logging

from packaging.version import Version

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .registry import _OpFn, _register_op_override


log = logging.getLogger(__name__)


_TRITON_REQUIRED_VERSION_MAJOR = 3
_TRITON_MINIMUM_VERSION_MINOR = 6


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if triton is available

    NOTE: must not import at this point
    """

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("triton")
    else:
        log.info("triton native DSL ops require: `triton` %s", reason)
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
def _version_is_sufficient() -> bool:
    _, version = _check_runtime_available()

    if version is None:
        return False

    # Either exact version, or same major
    major_ok = version.major == _TRITON_REQUIRED_VERSION_MAJOR
    minor_ok = version.minor >= _TRITON_MINIMUM_VERSION_MINOR

    if (major_ok and minor_ok) or check_native_version_skip():
        return True

    log.info(
        "triton version %s is not sufficient (>= (%s.%s.*)); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _TRITON_REQUIRED_VERSION_MAJOR,
        _TRITON_MINIMUM_VERSION_MINOR,
    )
    return False


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
    _register_op_override
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override(
        lib_symbol,
        op_symbol,
        dispatch_key,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )
