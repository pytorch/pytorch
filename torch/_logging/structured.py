"""
Utilities for converting data types into structured JSON for dumping.
"""

import traceback
from collections.abc import Sequence
from typing import Any

import torch._logging._internal


INTERN_TABLE: dict[str, int] = {}


DUMPED_FILES: set[str] = set()


def intern_string(s: str) -> int:
    r = INTERN_TABLE.get(s, None)
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r


def dump_file(filename: str) -> None:
    if "eval_with_key" not in filename:
        return
    if filename in DUMPED_FILES:
        return
    DUMPED_FILES.add(filename)
    from torch.fx.graph_module import _loader

    torch._logging._internal.trace_structured(
        "dump_file",
        metadata_fn=lambda: {
            "name": filename,
        },
        payload_fn=lambda: _loader.get_source(filename),
    )


def from_traceback(tb: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]:
    # dict naming convention here coincides with
    # python/combined_traceback.cpp
    r = [
        {
            "line": frame.lineno,
            "name": frame.name,
            "filename": intern_string(frame.filename),
        }
        for frame in tb
    ]
    return r
