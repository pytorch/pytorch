"""
Python polyfills for traceback
"""

from __future__ import annotations

import traceback

from ..decorators import substitute_in_graph


__all__ = ["extract_tb", "clear_frames"]


# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(traceback.extract_tb, can_constant_fold_through=True)
def extract_tb(tb, limit=None):
    if tb is None:
        return []
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
    return frame_summary


# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(traceback.clear_frames, can_constant_fold_through=True)
def clear_frames(tb):
    # no-op
    return None
