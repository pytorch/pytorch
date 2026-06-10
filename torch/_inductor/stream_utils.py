"""Stream utilities for Inductor codegen."""

from __future__ import annotations

import functools

from torch._inductor.stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)


__all__ = [
    "AOTI_SUPPORTED_STREAM_OP_NAMES",
    "AOTI_UNSUPPORTED_STREAM_OP_REASONS",
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "STREAM_NAME_TEMPLATE",
    "get_raw_stream_name",
    "get_stream_name",
]


AOTI_SUPPORTED_STREAM_OP_NAMES: dict[str, str] = {
    "torch.ops.streams.record_event.default": "record_event",
    "torch.ops.streams.wait_event.default": "wait_event",
    "torch.ops.streams.synchronize_event.default": "synchronize_event",
}


AOTI_UNSUPPORTED_STREAM_OP_REASONS: dict[str, str] = {
    "torch.ops.streams.synchronize_stream.default": (
        "Host-blocking sync ops should not appear inside an AOTI Run() - "
        "the caller is responsible for synchronizing before reading outputs. "
        "Replace with explicit record_event + wait_event for cross-stream "
        "ordering, or remove if not needed."
    ),
    "torch.ops.streams.synchronize_device.default": (
        "Host-blocking sync ops should not appear inside an AOTI Run() - "
        "synchronizing the entire device blocks every CUDA stream and the "
        "calling thread. The caller is responsible for any host-side "
        "synchronization."
    ),
    "torch.ops.streams.wait_stream.default": (
        "wait_stream is not supported in AOTI cpp_wrapper. Replace with "
        "explicit record_event on the waited-on stream + wait_event on the "
        "waiting stream."
    ),
}


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


@functools.lru_cache
def get_raw_stream_name(device_idx: int) -> str:
    """Generate variable name for a raw stream handle on the given device."""
    return f"raw_stream{device_idx}"
