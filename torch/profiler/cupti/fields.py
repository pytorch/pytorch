# mypy: allow-untyped-defs
"""CUPTI user-defined-record (v2) field-id schema.

In the v2 / user-defined-record path, observers select specific *fields* per
activity kind (rather than whole records). This module defines, per kind, the
field-id ``IntEnum`` (CUpti_Activity*FieldIds) the v2 monitor supports; the
registry used to validate selections and resolve ``"all"``; which fields are
``const char*`` strings that must be dereferenced during the buffer-completed
callback; and the field byte sizes used to compute record layouts and uniform
padding. Activity kinds come from cupti_python (no duplicate enum); the field ids
and sizes are CUPTI ABI constants spelled out here (cupti_activity.h), with the
kernel layout cross-checked byte-exact against real captured buffers.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from torch.profiler.cupti.cupti_python import ActivityKind


if TYPE_CHECKING:
    from collections.abc import Iterable


class KernelField(IntEnum):
    """CUpti_ActivityKernelFieldIds (KERNEL_FIELD_*)."""

    KIND = 0
    START = 7
    END = 8
    DEVICE_ID = 10
    CONTEXT_ID = 11
    STREAM_ID = 12
    CORRELATION_ID = 22
    NAME = 24  # const char* (mangled symbol); deref'd to str during parse
    GRAPH_NODE_ID = 31
    GRAPH_ID = 33


class MemcpyField(IntEnum):
    """CUpti_ActivityMemcpyFieldIds (MEMCPY_FIELD_*)."""

    KIND = 0
    COPY_KIND = 1
    SRC_KIND = 2
    DST_KIND = 3
    FLAGS = 4
    BYTES = 5
    START = 6
    END = 7
    DEVICE_ID = 8
    CONTEXT_ID = 9
    STREAM_ID = 10
    CORRELATION_ID = 11
    GRAPH_NODE_ID = 12
    GRAPH_ID = 13


class MemsetField(IntEnum):
    """CUpti_ActivityMemsetFieldIds (MEMSET_FIELD_*)."""

    KIND = 0
    VALUE = 1
    BYTES = 2
    START = 3
    END = 4
    DEVICE_ID = 5
    CONTEXT_ID = 6
    STREAM_ID = 7
    CORRELATION_ID = 8
    FLAGS = 9
    MEMORY_KIND = 10
    GRAPH_NODE_ID = 11
    GRAPH_ID = 12


class ApiField(IntEnum):
    """CUpti_ActivityApiFieldIds (API_FIELD_*); shared by RUNTIME and DRIVER
    (both are CUpti_ActivityAPI)."""

    KIND = 0
    CBID = 1
    START = 2
    END = 3
    PROCESS_ID = 4
    THREAD_ID = 5
    CORRELATION_ID = 6
    RETURN_VALUE = 7


class ExternalCorrelationField(IntEnum):
    """CUpti_ActivityExternalCorrelationFieldIds (EXTERNAL_CORRELATION_FIELD_*)."""

    KIND = 0
    EXTERNAL_KIND = 1
    EXTERNAL_ID = 2
    CORRELATION_ID = 3


class OverheadField(IntEnum):
    """CUpti_ActivityOverheadFieldIds (OVERHEAD_FIELD_*)."""

    KIND = 0
    OVERHEAD_KIND = 1
    PROCESS_ID = 2
    THREAD_ID = 3
    START = 4
    END = 5
    CORRELATION_ID = 6


class CudaEventField(IntEnum):
    """CUpti_ActivityCudaEventFieldIds (CUDA_EVENT_FIELD_*)."""

    KIND = 0
    CORRELATION_ID = 1
    CONTEXT_ID = 2
    STREAM_ID = 3
    EVENT_ID = 4
    DEVICE_ID = 5
    DEVICE_TIMESTAMP = 6
    CUDA_EVENT_SYNC_ID = 7


class SyncField(IntEnum):
    """CUpti_ActivitySynchronizationFieldIds (SYNCHRONIZATION_FIELD_*)."""

    KIND = 0
    TYPE = 1
    START = 2
    END = 3
    CORRELATION_ID = 4
    CONTEXT_ID = 5
    STREAM_ID = 6
    CUDA_EVENT_ID = 7
    CUDA_EVENT_SYNC_ID = 8
    RETURN_VALUE = 9


# kind -> its field-id enum (members = the field ids the v2 monitor supports).
FIELD_ENUMS: dict[int, type[IntEnum]] = {
    ActivityKind.CONCURRENT_KERNEL: KernelField,
    ActivityKind.MEMCPY: MemcpyField,
    ActivityKind.MEMSET: MemsetField,
    ActivityKind.RUNTIME: ApiField,
    ActivityKind.DRIVER: ApiField,
    ActivityKind.EXTERNAL_CORRELATION: ExternalCorrelationField,
    ActivityKind.OVERHEAD: OverheadField,
    ActivityKind.CUDA_EVENT: CudaEventField,
    ActivityKind.SYNCHRONIZATION: SyncField,
}

# kind -> frozenset of supported field ids; source of truth for validating
# observer requests and resolving "all". CUPTI requires *_FIELD_KIND (id 0) to
# be the first selected field, which the v2 monitor enforces at enable.
FIELD_REGISTRY: dict[int, frozenset[IntEnum]] = {
    kind: frozenset(enum) for kind, enum in FIELD_ENUMS.items()
}

# kind -> field ids whose record value is a const char* that must be
# dereferenced to a Python string DURING the buffer-completed callback (CUPTI
# frees the pointed-to string afterwards), rather than gathered as a numeric
# column.
STRING_FIELDS: dict[int, frozenset[int]] = {
    ActivityKind.CONCURRENT_KERNEL: frozenset({KernelField.NAME}),
}

# CUPTI user-defined records pack the selected fields in field-id order, each at
# its natural alignment, with the record rounded up to an 8-byte boundary. Field
# byte sizes are part of the CUPTI ABI (the record-struct member types), so a
# selection's record layout is fully determined -- no runtime discovery, no
# captured layout from CUPTI. (Validated live: the kernel layout below reproduces
# CUPTI's real captured offsets and decodes real buffers exactly.)
_RECORD_ALIGN = 8


def _align(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


# kind -> {field id: byte size}, from the CUpti_Activity* record member types
# (uint8->1, uint16->2, uint32/enum->4, uint64/pointer->8). CONCURRENT_KERNEL is
# validated against real captured buffers; the others are transcribed from
# cupti_activity.h and should be cross-checked against a captured layout under a
# v2 libcupti that populates it (>= 13.3).
FIELD_SIZES: dict[int, dict[int, int]] = {
    ActivityKind.CONCURRENT_KERNEL: {
        KernelField.KIND: 4,
        KernelField.START: 8,
        KernelField.END: 8,
        KernelField.DEVICE_ID: 4,
        KernelField.CONTEXT_ID: 4,
        KernelField.STREAM_ID: 4,
        KernelField.CORRELATION_ID: 4,
        KernelField.NAME: 8,
        KernelField.GRAPH_NODE_ID: 8,
        KernelField.GRAPH_ID: 4,
    },
    ActivityKind.MEMCPY: {
        MemcpyField.KIND: 4,
        MemcpyField.COPY_KIND: 1,
        MemcpyField.SRC_KIND: 1,
        MemcpyField.DST_KIND: 1,
        MemcpyField.FLAGS: 1,
        MemcpyField.BYTES: 8,
        MemcpyField.START: 8,
        MemcpyField.END: 8,
        MemcpyField.DEVICE_ID: 4,
        MemcpyField.CONTEXT_ID: 4,
        MemcpyField.STREAM_ID: 4,
        MemcpyField.CORRELATION_ID: 4,
        MemcpyField.GRAPH_NODE_ID: 8,
        MemcpyField.GRAPH_ID: 4,
    },
    ActivityKind.MEMSET: {
        MemsetField.KIND: 4,
        MemsetField.VALUE: 4,
        MemsetField.BYTES: 8,
        MemsetField.START: 8,
        MemsetField.END: 8,
        MemsetField.DEVICE_ID: 4,
        MemsetField.CONTEXT_ID: 4,
        MemsetField.STREAM_ID: 4,
        MemsetField.CORRELATION_ID: 4,
        MemsetField.FLAGS: 2,
        MemsetField.MEMORY_KIND: 2,
        MemsetField.GRAPH_NODE_ID: 8,
        MemsetField.GRAPH_ID: 4,
    },
    ActivityKind.RUNTIME: {
        ApiField.KIND: 4,
        ApiField.CBID: 4,
        ApiField.START: 8,
        ApiField.END: 8,
        ApiField.PROCESS_ID: 4,
        ApiField.THREAD_ID: 4,
        ApiField.CORRELATION_ID: 4,
        ApiField.RETURN_VALUE: 4,
    },
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelationField.KIND: 4,
        ExternalCorrelationField.EXTERNAL_KIND: 4,
        ExternalCorrelationField.EXTERNAL_ID: 8,
        ExternalCorrelationField.CORRELATION_ID: 4,
    },
    ActivityKind.OVERHEAD: {
        OverheadField.KIND: 4,
        OverheadField.OVERHEAD_KIND: 4,
        OverheadField.PROCESS_ID: 4,
        OverheadField.THREAD_ID: 4,
        OverheadField.START: 8,
        OverheadField.END: 8,
        OverheadField.CORRELATION_ID: 4,
    },
    ActivityKind.CUDA_EVENT: {
        CudaEventField.KIND: 4,
        CudaEventField.CORRELATION_ID: 4,
        CudaEventField.CONTEXT_ID: 4,
        CudaEventField.STREAM_ID: 4,
        CudaEventField.EVENT_ID: 4,
        CudaEventField.DEVICE_ID: 4,
        CudaEventField.DEVICE_TIMESTAMP: 8,
        CudaEventField.CUDA_EVENT_SYNC_ID: 8,
    },
    ActivityKind.SYNCHRONIZATION: {
        SyncField.KIND: 4,
        SyncField.TYPE: 4,
        SyncField.START: 8,
        SyncField.END: 8,
        SyncField.CORRELATION_ID: 4,
        SyncField.CONTEXT_ID: 4,
        SyncField.STREAM_ID: 4,
        SyncField.CUDA_EVENT_ID: 4,
        SyncField.CUDA_EVENT_SYNC_ID: 8,
        SyncField.RETURN_VALUE: 4,
    },
}
# RUNTIME and DRIVER share the CUpti_ActivityAPI record.
FIELD_SIZES[ActivityKind.DRIVER] = FIELD_SIZES[ActivityKind.RUNTIME]


def record_layout(
    kind: int, field_ids: Iterable[int]
) -> tuple[int, list[tuple[int, int, int]]]:
    """The packed layout of a user-defined record for ``kind`` given a field
    selection: ``(record_size, [(field_id, offset, size), ...])``. Fields are packed
    in field-id order, each at its natural alignment (= its size), the record padded
    to ``_RECORD_ALIGN``. Determined entirely by the spec -- no buffer, no captured
    layout. Fields without a known size are skipped."""
    sizes = FIELD_SIZES.get(kind, {})
    offset = 0
    entries: list[tuple[int, int, int]] = []
    for fid in sorted(int(f) for f in field_ids):
        size = sizes.get(fid)
        if size is None:
            continue
        offset = _align(offset, size)
        entries.append((fid, offset, size))
        offset += size
    return _align(offset, _RECORD_ALIGN), entries


def _subset_sum_indices(weights: list[int], target: int) -> list[int] | None:
    """Indices of a subset of ``weights`` summing exactly to ``target`` (or None).
    Inputs are a handful of discovered filler-field widths, so a small DP over
    reachable sums is plenty."""
    if target == 0:
        return []
    reach: dict[int, list[int]] = {0: []}
    for i, w in enumerate(weights):
        for s in list(reach.keys()):
            ns = s + w
            if 0 < ns <= target and ns not in reach:
                reach[ns] = reach[s] + [i]
    return reach.get(target)


def plan_padding(union: dict[int, frozenset[int]]) -> dict[int, frozenset[int]]:
    """Pad each kind's field selection with filler fields (other real fields of the
    kind) so every kind in ``union`` reaches the same record size -- letting a
    multi-kind buffer take the vectorized stride+dispatch demux instead of the
    per-record walk. Fully deterministic from the field-size spec: each kind's
    record size is ``record_layout``, fillers are sized by a subset-sum over the
    kind's other fields' sizes, and the padded layout is verified to hit the target
    (alignment can make a raw byte-sum not land exactly). Returns ``union`` unchanged
    when already uniform or when a kind can't be padded to the target."""
    if len(union) <= 1:
        return union
    base = {kind: record_layout(kind, fids)[0] for kind, fids in union.items()}
    target = max(base.values())
    if all(size == target for size in base.values()):
        return union

    padded: dict[int, frozenset[int]] = {}
    for kind, fids in union.items():
        gap = target - base[kind]
        if gap == 0:
            padded[kind] = frozenset(fids)
            continue
        # Filler candidates: the kind's other real fields (known sizes).
        fillers = [
            (f, s) for f, s in FIELD_SIZES.get(kind, {}).items() if f not in fids
        ]
        idx = _subset_sum_indices([s for _, s in fillers], gap)
        if idx is None:
            return union  # can't equalize this kind
        widened = frozenset(set(fids) | {fillers[i][0] for i in idx})
        # Alignment may make the raw byte-sum not land on target; verify.
        if record_layout(kind, widened)[0] != target:
            return union
        padded[kind] = widened
    return padded
