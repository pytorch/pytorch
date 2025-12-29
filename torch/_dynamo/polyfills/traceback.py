"""
Python polyfills for traceback
"""

import traceback
from traceback import FrameSummary
from types import TracebackType

from ..decorators import substitute_in_graph


__all__ = ["extract_tb", "clear_frames"]


@substitute_in_graph(traceback.extract_tb, can_constant_fold_through=True)
def extract_tb(
    tb: TracebackType | None, limit: int | None = None
) -> list[FrameSummary]:
    if tb is None:
        return traceback.StackSummary.from_list([])
    frame_summary = []
    while tb is not None:
        if limit:
            if len(frame_summary) < limit:
                frame_summary.append(
                    # pyrefly: ignore[missing-attribute]
                    tb.frame_summary
                )
            else:
                break
        else:
            frame_summary.append(tb.frame_summary)  # pyrefly: ignore[missing-attribute]
        tb = tb.tb_next
    return traceback.StackSummary.from_list(frame_summary)


@substitute_in_graph(traceback.clear_frames, can_constant_fold_through=True)
def clear_frames(tb: traceback.FrameSummary | None) -> None:
    # no-op
    return None
