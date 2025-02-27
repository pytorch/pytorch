"""
Utilities for converting data types into structured JSON for dumping.
"""

import inspect
import os
import traceback
from collections.abc import Sequence
from typing import Any, Optional

import torch._logging._internal


INTERN_TABLE: dict[str, int] = {}


DUMPED_FILES: set[str] = set()


def intern_string(s: Optional[str]) -> int:
    if s is None:
        return -1

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
            "loc": frame.line,
        }
        for frame in tb
    ]
    return r


def get_user_stack(num_frames: int) -> list[dict[str, Any]]:
    from torch._guards import TracingContext
    from torch.utils._traceback import CapturedTraceback

    user_tb = TracingContext.extract_stack()
    if user_tb:
        return from_traceback(user_tb[-1 * num_frames :])

    tb = CapturedTraceback.extract().summary()

    # Filter out frames that are within the torch/ codebase
    torch_filepath = os.path.dirname(inspect.getfile(torch)) + os.path.sep
    for i, frame in enumerate(reversed(tb)):
        if torch_filepath not in frame.filename:
            # Only display `num_frames` frames in the traceback
            filtered_tb = tb[len(tb) - i - num_frames : len(tb) - i]
            return from_traceback(filtered_tb)

    return from_traceback(tb[-1 * num_frames :])


def get_framework_stack(
    num_frames: int = 25, cpp: bool = False
) -> list[dict[str, Any]]:
    """
    Returns the traceback for the user stack and the framework stack
    """
    from torch.fx.experimental.symbolic_shapes import uninteresting_files
    from torch.utils._traceback import CapturedTraceback

    tb = CapturedTraceback.extract(cpp=cpp).summary()
    tb = [
        frame
        for frame in tb
        if (
            (
                frame.filename.endswith(".py")
                and frame.filename not in uninteresting_files()
            )
            or ("at::" in frame.name or "torch::" in frame.name)
        )
    ]

    return from_traceback(tb[-1 * num_frames :])
