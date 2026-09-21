import base64 as _base64
import functools
import hashlib as _hashlib
import logging
import sys
from collections.abc import Iterable as _Iterable
from importlib.metadata import (
    distribution as _distribution,
    packages_distributions as _packages_distributions,
)
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
from re import sub as _re_sub
from typing import Any as _Any, cast

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

# Fast-path union of names used by binary_populate_env.sh and torchtlx.
# Missing names are discovered by _packages_distributions().
_TRITON_DISTRIBUTIONS = (
    "triton",
    "triton-rocm",
    "triton-xpu",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "fbtriton",
)


def _normalized_name(name: str) -> str:
    """Distribution name in the form metadata lookups compare (PEP 503)."""
    return _re_sub(r"[-_.]+", "-", name).lower()


def _module_origin(module_name: str) -> str | None:
    """Resolve a module's file without importing it."""
    try:
        spec = _find_spec(module_name)
    except Exception:
        return None
    return None if spec is None else spec.origin


def _records_only_import_shims(paths: list[str]) -> bool:
    """Whether a RECORD contains only editable-install shims and metadata."""
    for path in paths:
        parts = _Path(path).parts
        if any(part.endswith((".dist-info", ".egg-info")) for part in parts):
            continue
        name = _Path(path).name
        if name.endswith(".pth") or name.startswith("__editable__"):
            continue
        return False
    return True


def _record_hash_matches(path: _Path, file_hash: _Any) -> bool | None:
    if file_hash is None:
        return None

    try:
        digest = _hashlib.new(file_hash.mode)
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        recorded = file_hash.value
        if len(recorded) == digest.digest_size * 2:
            expected = bytes.fromhex(recorded)
        else:
            expected = _base64.urlsafe_b64decode(recorded + "=" * (-len(recorded) % 4))
        return expected == digest.digest()
    except (OSError, TypeError, ValueError):
        return None


def _distribution_matches(name: str, origin: str | None) -> bool | None:
    """Whether a distribution's RECORD matches the module, or is undecidable."""
    if origin is None:
        return None

    try:
        files = _distribution(name).files
    except Exception:
        return None

    if files is None:
        return None
    if not files:
        return False

    try:
        origin_path = _Path(origin).resolve()
        located: list[str] = []
        for file in files:
            path = _Path(file.locate())
            if str(path) == origin or (
                path.name == origin_path.name
                and path.parent.name == origin_path.parent.name
                and path.resolve() == origin_path
            ):
                return _record_hash_matches(origin_path, file.hash)
            located.append(str(path))
    except Exception:
        return None

    if _records_only_import_shims(located):
        return None
    return False


def _candidate_versions(
    names: _Iterable[str | None], seen: set[str]
) -> list[tuple[str, Version]]:
    candidates: list[tuple[str, Version]] = []
    for name in names:
        if not isinstance(name, str):
            log.warning("Ignoring an unnamed distribution that provides triton")
            continue

        key = _normalized_name(name)
        if key in seen:
            continue
        seen.add(key)

        try:
            version = _available_version(name)
        except Exception:
            log.warning(
                "Ignoring unreadable triton distribution %s", name, exc_info=True
            )
            continue
        if version is not None:
            candidates.append((name, version))
    return candidates


def _resolve_ambiguous_version(
    candidates: list[tuple[str, Version]],
    origin: str | None,
    verdicts: dict[str, bool | None],
    *,
    fallback: bool,
) -> Version | None:
    matched: list[tuple[str, Version]] = []
    undecidable: list[tuple[str, Version]] = []
    for candidate in candidates:
        name = candidate[0]
        if name not in verdicts:
            verdicts[name] = _distribution_matches(name, origin)
        result = verdicts[name]
        if result is True:
            matched.append(candidate)
        elif result is None:
            undecidable.append(candidate)

    if not fallback and not matched:
        return None

    choices = matched or undecidable or candidates
    if len(choices) > 1 and len({version for _, version in choices}) > 1:
        log.warning(
            "Could not uniquely identify the triton distribution; using %s %s",
            *choices[0],
        )
    return choices[0][1]


def _available_triton_version() -> Version | None:
    """
    Best-supported installed version for the importable `triton`.

    Known names avoid a sys.path scan. A lone candidate is returned directly,
    so a stale known name can hide an unlisted provider as it did on main.
    Collisions are checked against RECORD; ties preserve the prior preference
    for `triton`. This function must not import triton.
    """
    seen: set[str] = set()
    origin: str | None = None
    verdicts: dict[str, bool | None] = {}
    candidates = _candidate_versions(_TRITON_DISTRIBUTIONS, seen)
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        origin = _module_origin("triton")
        version = _resolve_ambiguous_version(
            candidates, origin, verdicts, fallback=False
        )
        if version is not None:
            return version

    try:
        providers = _packages_distributions().get("triton", ())
    except Exception:
        if not candidates:
            log.warning(
                "Could not resolve the distribution providing triton; "
                "triton native DSL ops will not register",
                exc_info=True,
            )
            return None
        log.warning("Could not scan for additional triton distributions", exc_info=True)
    else:
        candidates.extend(_candidate_versions(providers, seen))

    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        # A nonempty cache means origin was computed, including a None result.
        if not verdicts:
            origin = _module_origin("triton")
        return _resolve_ambiguous_version(candidates, origin, verdicts, fallback=True)

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
