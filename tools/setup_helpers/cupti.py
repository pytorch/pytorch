"""Locate a sufficiently-new CUPTI header (``cupti_activity.h``) for the CUPTI
field-id codegen (``tools/gen_cupti_stubs.py``), which parses the header to emit
``torch/profiler/_cupti/_cupti_stubs.py``. Kept as a small standalone helper the
CMake build and the CI build scripts both import at configure time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


# CUPTI 13.3.0 (CUPTI_API_VERSION == 130300). The field-id codegen needs the v2
# user-defined-record field-id enums, which the CUPTI ABI header only gained at
# 13.3; an older header would emit an incomplete catalog. Mirrors the monitor's
# runtime floor.
_MIN_CUPTI_API_VERSION = 130300


def _cupti_api_version(include_dir: Path) -> int | None:
    """Parse ``CUPTI_API_VERSION`` from ``cupti_version.h`` in ``include_dir``
    (None if the version header is absent or the macro can't be parsed)."""
    version_header = include_dir / "cupti_version.h"
    if not version_header.is_file():
        return None
    match = re.search(
        r"^\s*#\s*define\s+CUPTI_API_VERSION\s+(\d+)",
        version_header.read_text(),
        re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def find_cupti_header() -> Path | None:
    """Locate ``cupti_activity.h`` whose ``CUPTI_API_VERSION`` is at least
    ``_MIN_CUPTI_API_VERSION``. Candidate include dirs, in priority order:

    1. ``CUPTI_INCLUDE_DIR`` -- explicit override for out-of-tree setups.
    2. ``/usr/local/cupti-headers-<major.minor>`` -- the CUPTI redist headers
       staged into the CI Docker image by ``.ci/docker/common/install_cuda.sh``
       (``install_cupti_headers``); the highest version present wins.
    3. The ``nvidia-cuda-cupti`` wheel (namespace package ``nvidia.cu13``) -- a
       convenience fallback for local builds where the wheel is already installed.

    Returns the path only when a candidate both exists and is new enough; None
    otherwise, so callers skip the codegen and the libclang build-dep. The CUDA
    toolkit is deliberately not a source: its ``cupti_activity.h`` can predate the
    v2 field-id enums, and this version gate would reject it anyway."""
    candidate_dirs: list[Path] = []

    if env := os.environ.get("CUPTI_INCLUDE_DIR"):
        candidate_dirs.append(Path(env))

    # CUPTI redist headers staged into the CI Docker image by install_cuda.sh.
    # Several cupti-headers-<major.minor> dirs may coexist; prefer the highest.
    def _version_key(p: Path) -> tuple[int, ...]:
        suffix = p.name.removeprefix("cupti-headers-")
        return tuple(int(x) for x in suffix.split(".") if x.isdigit())

    candidate_dirs += sorted(
        Path("/usr/local").glob("cupti-headers-*"), key=_version_key, reverse=True
    )

    try:
        import nvidia.cu13  # pyrefly: ignore[missing-import]  # from nvidia-cuda-cupti

        candidate_dirs += [Path(loc) / "include" for loc in nvidia.cu13.__path__]
    except ImportError:
        pass

    for include_dir in candidate_dirs:
        header = include_dir / "cupti_activity.h"
        if not header.is_file():
            continue
        version = _cupti_api_version(include_dir)
        if version is not None and version >= _MIN_CUPTI_API_VERSION:
            return header
    return None
