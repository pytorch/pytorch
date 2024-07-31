"""
Utilities for converting data types into structured JSON for dumping.
"""

import traceback
from typing import Any, Dict, List, Sequence

import torch._logging._internal


INTERN_TABLE: Dict[str, int] = {}


def intern_string(s: str) -> int:
    r = INTERN_TABLE.get(s, None)
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r


def from_traceback(tb: Sequence[traceback.FrameSummary]) -> List[Dict[str, Any]]:
    r = []
    for frame in tb:
        # dict naming convention here coincides with
        # python/combined_traceback.cpp
        r.append(
            {
                "line": frame.lineno,
                "name": frame.name,
                "filename": intern_string(frame.filename),
            }
        )
    return r
