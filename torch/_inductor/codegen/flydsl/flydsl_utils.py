import functools
import logging
from importlib.machinery import PathFinder
from importlib.util import find_spec

from torch._native.common_utils import _available_version
from torch.backends import cuda as _cuda


log = logging.getLogger(__name__)
_pathfinder_find_spec = PathFinder.find_spec
# FlyDSL 0.3.x ``@fx.struct`` values must expose ``__cache_signature__()``.
_FLYDSL_SUPPORTED_RELEASE = (0, 3)


def _flydsl_runtime_unavailable_reason() -> str | None:
    try:
        flydsl_spec = find_spec("flydsl")
    except (ImportError, ValueError):
        flydsl_spec = None
    if flydsl_spec is None or flydsl_spec.submodule_search_locations is None:
        return "missing optional dependency `flydsl`"

    # Query the package paths directly so this availability check does not
    # import flydsl as a side effect during regular torch imports.
    try:
        mlir_spec = _pathfinder_find_spec(
            "_mlir",
            list(flydsl_spec.submodule_search_locations),
        )
    except (ImportError, ValueError):
        mlir_spec = None
    if mlir_spec is None:
        return "missing optional dependency `flydsl._mlir` (runtime is not built)"

    flydsl_version = _available_version("flydsl")
    if flydsl_version is None:
        return "missing or invalid FlyDSL version metadata"
    if flydsl_version.release[:2] != _FLYDSL_SUPPORTED_RELEASE:
        supported = ".".join(map(str, _FLYDSL_SUPPORTED_RELEASE))
        return (
            f"unsupported FlyDSL version `{flydsl_version}` (expected `{supported}.x`)"
        )

    return None


@functools.cache
def _check_runtime_available() -> bool:
    import torch

    if not _cuda.is_built():
        return False

    if torch.version.hip is None:
        return False

    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True


def runtime_available() -> bool:
    return _check_runtime_available()
