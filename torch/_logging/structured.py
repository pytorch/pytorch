"""
Utilities for converting data types into structured JSON for dumping.
"""

import traceback
from typing import Sequence

def from_traceback(tb: Sequence[traceback.FrameSummary]) -> object:
    r = []
    for frame in tb:
        # dict naming convention here coincides with
        # python/combined_traceback.cpp
        r.append({
            "line": frame.lineno,
            "name": frame.name,
            "filename": frame.filename,
        })
    return r
