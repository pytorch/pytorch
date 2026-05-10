"""Stream utilities for Inductor codegen."""

from __future__ import annotations

import functools

from torch._inductor.stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)


__all__ = [
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "STREAM_NAME_TEMPLATE",
    "get_stream_name",
]


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)
