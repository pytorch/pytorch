import importlib
import importlib.metadata
import os
from functools import cache

import packaging.version


@cache
def check_native_jit_disabled() -> bool:
    """
    Single point to check if native DSL ops are disabled globally,
    checked via:
    TORCH_DISABLE_NATIVE_JIT=1
    """
    return int(os.getenv("TORCH_DISABLE_NATIVE_JIT", 0)) == 1


def _unavailable_reason(deps: list[tuple[str, str]]) -> None | str:
    """
    Check availability of required packages - cuteDSL & deps,
    informing user what (if anything) is missing

    NOTE: Doesn't actually import anything.
    """
    for package_name, module_name in deps:
        # Note this doesn't actually import the packages
        if importlib.util.find_spec(module_name) is None:
            return (
                f"missing optional dependency `{package_name}` "
                f"(importlib.util.find_spec({package_name}) failed)"
            )
    return None


def _available_version(package: str) -> packaging.version.Version | None:
    """
    Get the installed version of a package as (major, minor, patch).

    Handles pre-release suffixes like "0.7.0rc1" or "3.1.0.post1" by
    stripping non-numeric tails from each component. Returns None on
    parse failure.
    """
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        v = packaging.version.parse(version)
    except packaging.version.InvalidVersion:
        return None

    return v


@cache
def check_native_version_skip() -> bool:
    """
    Single point to check if native DSL version gating should be skipped,
    checked via:
    TORCH_NATIVE_SKIP_VERSION_CHECK=1
    """
    return int(os.getenv("TORCH_NATIVE_SKIP_VERSION_CHECK", 0)) == 1
