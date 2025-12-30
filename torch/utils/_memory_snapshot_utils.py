import os
import pickle
import re
from typing import cast, TypedDict
from typing_extensions import NotRequired


class _Frame(TypedDict):
    """Frame information from memory profiler snapshots."""

    filename: str
    line: int
    name: str
    # Fields added by FX augmentation (optional)
    fx_node_op: NotRequired[str]
    fx_node_name: NotRequired[str]
    fx_node_target: NotRequired[str]
    fx_original_trace: NotRequired[str]


class _Block(TypedDict):
    """Memory block information."""

    size: int
    requested_size: int
    address: int
    state: str
    frames: list[_Frame]


class _Segment(TypedDict):
    """Memory segment information."""

    address: int
    total_size: int
    stream: int
    segment_type: str
    allocated_size: int
    active_size: int
    blocks: list[_Block]


class _TraceEntry(TypedDict):
    """Memory trace entry information."""

    action: str
    addr: NotRequired[int]
    frames: list[_Frame]
    size: int
    stream: int
    device_free: NotRequired[int]


class _Snapshot(TypedDict):
    """Memory snapshot structure."""

    segments: list[_Segment]
    device_traces: NotRequired[list[list[_TraceEntry]]]


def _augment_frames(frames: list[_Frame]) -> int:
    """
    Augment a list of frames with FX debug information. For each frame corresponding
    to an FX-generated Python file, this function attaches additional FX node
    metadata (op, name, target, and original trace).

    Args:
        frames (list[_Frame]): List of frame dictionaries to augment

    Returns:
        int: The count of frames that were augmented.
    """
    from torch.fx.graph_module import FX_GRAPH_MODULE_FILE_PREFIX
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    # Regex pattern to match FX generated files
    _FX_GENERATED_PATTERN = re.compile(
        rf"{re.escape(FX_GRAPH_MODULE_FILE_PREFIX)}.*\.py$"
    )

    count = 0

    for frame in frames:
        filename = frame.get("filename")
        lineno = frame.get("line")
        if not filename or not lineno:
            continue

        # Check if this looks like an FX generated file
        if not _FX_GENERATED_PATTERN.search(os.path.basename(filename)):
            continue

        metadata = _FX_METADATA_REGISTRY.get(filename)
        if metadata is None:
            continue

        lineno_map = metadata.get("lineno_map", {})
        node_metadata = metadata.get("node_metadata", {})
        prologue_start = metadata.get("prologue_start", 0)

        # Get the node index for this line
        node_idx = lineno_map.get(lineno - prologue_start)
        if node_idx is None:
            continue

        node_info = node_metadata.get(node_idx)
        if node_info is None:
            continue

        # Populate FX metadata fields
        frame["fx_node_op"] = node_info.get("op")
        frame["fx_node_name"] = node_info.get("name")
        frame["fx_node_target"] = str(node_info.get("target"))

        # Attach original stack trace if available
        original_trace = node_info.get("stack_trace")
        if original_trace:
            frame["fx_original_trace"] = original_trace

        count += 1

    return count


def _augment_memory_snapshot_stack_traces(
    snapshot: str | _Snapshot,
) -> _Snapshot:
    """
    Augment a memory snapshot with original source stack traces from FX metadata.

    IMPORTANT: This function reads from a global in-memory registry (_FX_METADATA_REGISTRY)
    that is populated during graph module compilation. It must be called in the same
    Python process where the FX graphs were compiled. It cannot be used to augment
    snapshots loaded from disk in a different process.

    Args:
        snapshot (str or _Snapshot): Either a memory snapshot dict or path to a snapshot pickle file

    Returns:
        _Snapshot: The augmented snapshot dictionary with fx_node_op, fx_node_name,
            fx_original_trace, and fx_node_info fields added to frames
    """

    snapshot_dict: _Snapshot
    if isinstance(snapshot, str):
        # Load the memory snapshot
        with open(snapshot, "rb") as f:
            snapshot_dict = cast(_Snapshot, pickle.load(f))
    else:
        snapshot_dict = snapshot

    # Process blocks in segments (for regular allocations)
    for segment in snapshot_dict.get("segments", []):
        for block in segment.get("blocks", []):
            if "frames" in block:
                _augment_frames(block["frames"])

    # Process device traces (for memory history)
    for trace_list in snapshot_dict.get("device_traces", []):
        for trace_entry in trace_list:
            if isinstance(trace_entry, dict) and "frames" in trace_entry:
                _augment_frames(trace_entry["frames"])

    return snapshot_dict
