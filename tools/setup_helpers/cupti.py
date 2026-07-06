"""Locate the CUPTI header (``cupti_activity.h``) for the CUPTI monitor build.

Both the native worker (compiled against the header) and the field-id codegen
(which parses it) need the same header, so the resolver lives here as a small
standalone helper the CMake build imports at configure time.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_cupti_header() -> Path | None:
    """Locate ``cupti_activity.h`` from the ``nvidia-cuda-cupti`` wheel (namespace
    package ``nvidia.cu13``) -- the same package that ships the runtime
    ``libcupti.so.13`` the monitor loads. The CUDA toolkit is deliberately not a
    fallback: its ``cupti_activity.h`` can predate the v2 user-defined-record
    structs (the monitor's floor is 13.3), so building against it would fail or
    mismatch the runtime. ``CUPTI_INCLUDE_DIR`` overrides for out-of-tree setups."""
    if env := os.environ.get("CUPTI_INCLUDE_DIR"):
        if (h := Path(env) / "cupti_activity.h").is_file():
            return h
    try:
        import nvidia.cu13  # pyrefly: ignore[missing-import]  # from nvidia-cuda-cupti
    except ImportError:
        return None
    for loc in nvidia.cu13.__path__:
        if (h := Path(loc) / "include" / "cupti_activity.h").is_file():
            return h
    return None
