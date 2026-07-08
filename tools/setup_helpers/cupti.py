"""Locate the CUPTI header (``cupti_activity.h``) for the CUPTI monitor build.

Both the native worker (compiled against the header) and the field-id codegen
(which parses it) need the same header, so the resolver lives here as a small
standalone helper the CMake build imports at configure time.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_cupti_header() -> Path | None:
    """Locate ``cupti_activity.h`` for the CUPTI monitor build. Resolution order:

    1. ``CUPTI_INCLUDE_DIR`` -- explicit override for out-of-tree setups.
    2. ``/usr/local/cupti-headers-<major.minor>`` -- the CUPTI redist headers
       staged into the CI Docker image by ``.ci/docker/common/install_cuda.sh``
       (``install_cupti_headers``); the highest version present wins. This is the
       CI path, so a CUDA build needs no ``nvidia-cuda-cupti`` wheel at build time.
    3. The ``nvidia-cuda-cupti`` wheel (namespace package ``nvidia.cu13``) -- a
       convenience fallback for local builds where the wheel is already installed.

    The CUDA toolkit is deliberately not a source: its ``cupti_activity.h`` can
    predate the v2 user-defined-record structs (the monitor's floor is 13.3), so
    building against it would fail or mismatch the runtime libcupti."""
    if env := os.environ.get("CUPTI_INCLUDE_DIR"):
        if (h := Path(env) / "cupti_activity.h").is_file():
            return h

    # CUPTI redist headers staged into the CI Docker image by install_cuda.sh.
    # Several cupti-headers-<major.minor> dirs may coexist; prefer the highest.
    def _version_key(p: Path) -> tuple[int, ...]:
        suffix = p.name.removeprefix("cupti-headers-")
        return tuple(int(x) for x in suffix.split(".") if x.isdigit())

    staged = sorted(
        Path("/usr/local").glob("cupti-headers-*"), key=_version_key, reverse=True
    )
    for d in staged:
        if (h := d / "cupti_activity.h").is_file():
            return h

    try:
        import nvidia.cu13  # pyrefly: ignore[missing-import]  # from nvidia-cuda-cupti
    except ImportError:
        return None
    for loc in nvidia.cu13.__path__:
        if (h := Path(loc) / "include" / "cupti_activity.h").is_file():
            return h
    return None
