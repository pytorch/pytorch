import functools
import logging
from importlib.machinery import PathFinder
from importlib.util import find_spec
from typing import TYPE_CHECKING

from torch._native.common_utils import _available_version


if TYPE_CHECKING:
    import torch


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
    # Package usability only; backend and arch checks belong in template admission.
    # Compare torch._native.flydsl_utils.runtime_available(), which stays ROCm-only
    # because it gates in-tree eager ATen operator overrides.
    reason = _flydsl_runtime_unavailable_reason()
    if reason is not None:
        log.debug("FlyDSL Inductor templates are unavailable: %s", reason)
        return False

    return True


def runtime_available() -> bool:
    """Return whether the optional ``flydsl`` package and MLIR runtime are usable."""
    return _check_runtime_available()


def is_gfx950_device(device: "torch.device | None") -> bool:
    """True when ``device`` reports the in-tree gfx950 FlyDSL arch.

    The vendored FlyDSL kernels assume gfx950 (MI350) LDS capacity and MFMA
    layout, so gate strictly on gfx950.
    """
    if device is None or device.type != "cuda":
        return False
    from .flydsl_scheduling import _get_flydsl_device_arch

    device_index = device.index if device.index is not None else 0
    return _get_flydsl_device_arch(device_index) == "gfx950"
