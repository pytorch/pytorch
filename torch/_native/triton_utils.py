import functools
import logging
import sys
from importlib.metadata import (
    distribution as _distribution,
    packages_distributions as _packages_distributions,
)
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
from re import sub as _re_sub
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

# Names to try before the sys.path scan, so that a recognized install does not
# pay for it. No file in the tree enumerates all of them and this is not meant to
# become that file: .ci/pytorch/binary_populate_env.sh publishes `triton`,
# `triton-rocm`, `fbtriton` and `triton-xpu`, while TRITON_DISTRIBUTIONS in
# tools/torchtlx/dev.py and tools/torchtlx/_probe.py list the five that collide
# in a dev environment, omitting `triton-xpu` and adding the `pytorch-triton`
# names. This tuple is their union, and only a fast path: the scan below resolves
# a name missing here, so drift costs a scan rather than a wrong answer.
_TRITON_DISTRIBUTIONS = (
    "triton",
    "triton-rocm",
    "triton-xpu",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "fbtriton",
)


def _normalized_name(name: str) -> str:
    """
    Distribution name in the form metadata lookups compare (PEP 503)
    """
    return _re_sub(r"[-_.]+", "-", name).lower()


_TRITON_DISTRIBUTION_KEYS = frozenset(
    _normalized_name(name) for name in _TRITON_DISTRIBUTIONS
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


def _records_only_import_shims(paths: list[str]) -> bool:
    """
    Whether a distribution recorded nothing but the shims that import the module

    An editable install (PEP 660) records a `.pth` and a finder that put the
    module on sys.path, and its own metadata, but never the module: setuptools
    writes only those, and the module itself stays in the source tree. Such a
    file list cannot say what the distribution owns.
    """
    for path in paths:
        parts = _Path(path).parts
        if any(part.endswith((".dist-info", ".egg-info")) for part in parts):
            continue
        name = _Path(path).name
        if name.endswith(".pth") or name.startswith("__editable__"):
            continue
        return False
    return True


def _distribution_owns(name: str, origin: str | None) -> bool:
    """
    Whether the distribution `name` installed the file at `origin`

    Undecidable in four cases -- no origin, metadata that cannot be read, a
    distribution with no RECORD, and one that recorded only import shims -- and
    all four answer True: nothing was learned, so the distribution keeps the
    benefit of the doubt it had before the question was asked.
    """
    if origin is None:
        return True

    try:
        files = _distribution(name).files
    except Exception:
        return True

    if not files:
        return True

    # Stops at the match, so an ordinary install does not walk the whole RECORD.
    if any(str(file.locate()) == origin for file in files):
        return True

    located = [str(file.locate()) for file in files]
    if _records_only_import_shims(located):
        return True

    # Reached whenever the distribution does not own the module -- the stale
    # dist-info this lookup exists to see through -- so compare only the entries
    # that could match before resolving, which stats the file.
    origin_path = _Path(origin).resolve()
    return any(
        _Path(path).resolve() == origin_path
        for path in located
        if _Path(path).name == origin_path.name
    )


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
        providers = _packages_distributions().get("triton", ())
    except Exception:
        # Reading the metadata is best-effort, but declining leaves the ops
        # unregistered on an install where Triton itself works, so say so.
        log.warning(
            "Could not resolve the distribution providing triton; "
            "triton native DSL ops will not register",
            exc_info=True,
        )
        return None

    for provider in providers:
        try:
            if _normalized_name(provider) in _TRITON_DISTRIBUTION_KEYS:
                # Already tried above, with the same answer.
                continue
            version = _available_version(provider)
            if version is not None and _distribution_owns(provider, origin):
                return version
        except Exception:
            # A dist-info whose METADATA has no `Name` arrives here as a `None`
            # provider, which the version lookup rejects. Skip it rather than
            # abandoning the providers listed after it.
            log.warning(
                "Ignoring a distribution that reports providing triton but "
                "cannot be read",
                exc_info=True,
            )

    # Left at info: reaching here means no metadata reports a version at all,
    # which is what a source checkout on PYTHONPATH looks like, and is expected
    # rather than notable. An editable install is not this case -- it reports a
    # version, and _distribution_owns accepts it.
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
