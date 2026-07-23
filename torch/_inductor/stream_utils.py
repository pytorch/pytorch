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
    # A targeted single-event host wait (cudaEventSynchronize); required by e.g.
    # pinned non-blocking copies. Unlike synchronize_stream/device, it does not
    # block the whole stream/device, so it is safe to emit inside an AOTI Run().
    "torch.ops.streams.synchronize_event.default": "synchronize_event",
}


AOTI_UNSUPPORTED_STREAM_OP_REASONS: dict[str, str] = {
    "torch.ops.streams.synchronize_stream.default": (
        "Host-blocking sync ops are not supported inside an AOTI Run(). "
        "Use record_event + wait_event for device-side ordering instead."
    ),
    "torch.ops.streams.synchronize_device.default": (
        "Host-blocking device synchronization is not supported inside an AOTI Run()."
    ),
    "torch.ops.streams.wait_stream.default": (
        "wait_stream is not supported in AOTI cpp_wrapper. Use explicit "
        "record_event on the waited-on stream + wait_event on the waiting stream."
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
