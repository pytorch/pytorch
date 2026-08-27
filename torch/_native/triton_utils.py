import functools
import logging
import sys
from importlib.metadata import (
    distribution as _distribution,
    packages_distributions as _packages_distributions,
)
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
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


_TRITON_DSL_NAME = "triton"
_TRITON_REQUIRED_VERSION_MAJOR = 3
_TRITON_MINIMUM_VERSION_MINOR = 6

# Distribution names known to publish the `triton` module, tried before the
# sys.path scan so that no install pays for it. TRITON_DISTRIBUTIONS in
# tools/torchtlx/dev.py lists the ones that collide; `triton-xpu` is built by
# .ci/pytorch/binary_populate_env.sh. A name missing here costs a scan, not a
# wrong answer, so the list falling behind the wheels is not a correctness bug.
_TRITON_DISTRIBUTIONS = (
    "triton",
    "triton-rocm",
    "triton-xpu",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "fbtriton",
)


def _module_origin(module_name: str) -> str | None:
    """
    File the module resolves to, or None if that cannot be decided

    NOTE: must not import at this point
    """
    try:
        spec = _find_spec(module_name)
    except Exception:
        # A broken parent package raises rather than reporting the module
        # missing, and an origin is a nicety here: the caller reads None as
        # "cannot decide" and keeps whatever the distribution names report.
        return None
    return None if spec is None else spec.origin


def _distribution_owns(name: str, origin: str | None) -> bool:
    """
    Whether the distribution `name` installed the file at `origin`

    Undecidable in three cases -- no origin, metadata that cannot be read, and
    a distribution with no RECORD -- and all three answer True: nothing was
    learned, so the distribution keeps the benefit of the doubt it had before
    the question was asked.
    """
    if origin is None:
        return True

    try:
        files = _distribution(name).files
    except Exception:
        return True

    if not files:
        return True

    located = [file.locate() for file in files]
    if any(str(path) == origin for path in located):
        return True

    # Only a relocated or symlinked install needs the paths resolved, which
    # stats every file the distribution installed.
    origin_path = _Path(origin).resolve()
    return any(_Path(path).resolve() == origin_path for path in located)


def _available_triton_version() -> Version | None:
    """
    Version of the distribution that provides the importable `triton`

    The same module ships under several distribution names, and a name this
    lookup misses is indistinguishable from Triton not being installed, which
    disables the ops on a working install with nothing to point at. The names
    in _TRITON_DISTRIBUTIONS are tried first to keep the sys.path scan off the
    import path, and the scan then covers any name not listed there, so a wheel
    published under a new name still resolves.

    Neither answer is taken on the name alone. Uninstalling one provider after
    another has overwritten its files leaves a dist-info that still reports a
    version for a module it no longer owns (TRITON_DISTRIBUTIONS in
    tools/torchtlx/dev.py), and the scan lists providers in sys.path order,
    which does not rank the owner first. Each candidate is therefore checked
    against the file the module resolves to, and a candidate that cannot be
    checked is accepted as before.

    NOTE: must not import at this point
    """
    origin = _module_origin("triton")

    for name in _TRITON_DISTRIBUTIONS:
        version = _available_version(name)
        if version is not None and _distribution_owns(name, origin):
            return version

    try:
        for provider in _packages_distributions().get("triton", ()):
            if provider in _TRITON_DISTRIBUTIONS:
                # Already tried above, with the same answer.
                continue
            version = _available_version(provider)
            if version is not None and _distribution_owns(provider, origin):
                return version
    except Exception:
        # Reading the metadata is best-effort, but declining leaves the ops
        # unregistered on an install where Triton itself works, so say so.
        log.warning(
            "Could not resolve the distribution providing triton; "
            "triton native DSL ops will not register",
            exc_info=True,
        )
        return None

    log.info(
        "no installed distribution reports a parseable version for the `triton` "
        "module; triton native DSL ops will not register"
    )
    return None


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    """
    Check if triton is available

    NOTE: must not import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_triton_version()
    else:
        # info, not warning: see cutedsl_utils._check_runtime_available for
        # rationale (missing optional deps is the common case; surface via
        # TORCH_LOGS=+native_dsl when needed).
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
        # _available_triton_version already logged why. Falling through would
        # report the absent version as the problem and point at
        # TORCH_NATIVE_SKIP_VERSION_CHECK, which cannot rescue this case.
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


def deregister_op_overrides() -> None:
    """
    Deregister all ops through triton
    """
    _deregister_op_overrides_impl(disable_dsl_names=_TRITON_DSL_NAME)


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
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _TRITON_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry
# Note: Import-time registration ensures DSL is available when module is loaded
dsl_registry.register_dsl("triton", cast(DSLModuleProtocol, sys.modules[__name__]))
