# mypy: allow-untyped-defs
"""CUPTI user-defined-record (v2) field schema.

In the v2 / user-defined-record path, observers select specific *fields* per activity
kind (rather than whole records). The available fields per kind are the generated
:mod:`_cupti_field_ids` catalogs (``Kernel``, ``Memcpy``, ...), one
:class:`~_records_base.Field` per ``CUpti_Activity*FieldIds`` id, each carrying its
:class:`~_records_base.Ctype` for decode. Those catalogs are generated from the CUPTI
ABI (cupti_activity.h) at build time -- cupti-python does not expose the enums.

This module *curates* which of those fields the monitor selects per kind (:data:`FIELDS`)
and derives the lookups the monitor/observers need. The selection is the editorial part
that can't be generated: it bounds record size and per-buffer decode cost.

The monitor does NOT compute record byte layouts: it requires libcupti >= 13.3, which
reports each kind's packed record layout (field offsets/sizes, record size) via
``pBufferCompleteInfo->ppRecordLayouts``. The native layer parses that and attaches it to
each completed buffer; the monitor decodes a buffer against that captured layout, using
:data:`FIELD_CTYPE` only to interpret each field's bytes (signed/unsigned/float/str).
"""

from __future__ import annotations

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

from torch.profiler._cupti._cupti_field_ids import (
    Api,
    CudaEvent,
    ExternalCorrelation,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
    Overhead,
    Synchronization,
)
from torch.profiler._cupti._records_base import Ctype, Field


# Short alias kept for the SYNCHRONIZATION catalog (the generated name is Synchronization).
Sync = Synchronization


__all__ = [
    "Ctype",
    "Field",
    "Api",
    "CudaEvent",
    "ExternalCorrelation",
    "Kernel",
    "Memcpy",
    "Memcpy2",
    "Memset",
    "Overhead",
    "Sync",
    "Synchronization",
    "FIELDS",
    "FIELD_REGISTRY",
    "FIELD_CTYPE",
    "STRING_FIELDS",
    "CORRELATION_FIELD",
    "GRAPH_NODE_FIELD",
    "RecordLayouts",
]


# kind -> the fields the v2 monitor selects for that kind, in declaration order. A curated
# subset of each generated catalog; KIND (id 0) must lead every record (CUPTI
# requires *_FIELD_KIND first at enable). RUNTIME and DRIVER share the Api catalog.
FIELDS: dict[int, tuple[Field, ...]] = {
    ActivityKind.CONCURRENT_KERNEL: (
        Kernel.KIND,
        Kernel.REGISTERS_PER_THREAD,
        Kernel.START,
        Kernel.END,
        Kernel.DEVICE_ID,
        Kernel.CONTEXT_ID,
        Kernel.STREAM_ID,
        Kernel.GRID_X,
        Kernel.GRID_Y,
        Kernel.GRID_Z,
        Kernel.BLOCK_X,
        Kernel.BLOCK_Y,
        Kernel.BLOCK_Z,
        Kernel.STATIC_SHARED_MEMORY,
        Kernel.DYNAMIC_SHARED_MEMORY,
        Kernel.CORRELATION_ID,
        Kernel.NAME,
        Kernel.GRAPH_NODE_ID,
        Kernel.GRAPH_ID,
        Kernel.LAUNCH_PRIORITY,
        Kernel.QUEUED,
        Kernel.CHANNEL_ID,
        Kernel.CHANNEL_TYPE,
    ),
    ActivityKind.MEMCPY: (
        Memcpy.KIND,
        Memcpy.COPY_KIND,
        Memcpy.SRC_KIND,
        Memcpy.DST_KIND,
        Memcpy.FLAGS,
        Memcpy.BYTES,
        Memcpy.START,
        Memcpy.END,
        Memcpy.DEVICE_ID,
        Memcpy.CONTEXT_ID,
        Memcpy.STREAM_ID,
        Memcpy.CORRELATION_ID,
        Memcpy.GRAPH_NODE_ID,
        Memcpy.GRAPH_ID,
    ),
    ActivityKind.MEMCPY2: (
        Memcpy2.KIND,
        Memcpy2.COPY_KIND,
        Memcpy2.SRC_KIND,
        Memcpy2.DST_KIND,
        Memcpy2.FLAGS,
        Memcpy2.BYTES,
        Memcpy2.START,
        Memcpy2.END,
        Memcpy2.DEVICE_ID,
        Memcpy2.CONTEXT_ID,
        Memcpy2.STREAM_ID,
        Memcpy2.SRC_DEVICE_ID,
        Memcpy2.DST_DEVICE_ID,
        Memcpy2.CORRELATION_ID,
        Memcpy2.GRAPH_NODE_ID,
        Memcpy2.GRAPH_ID,
    ),
    ActivityKind.MEMSET: (
        Memset.KIND,
        Memset.VALUE,
        Memset.BYTES,
        Memset.START,
        Memset.END,
        Memset.DEVICE_ID,
        Memset.CONTEXT_ID,
        Memset.STREAM_ID,
        Memset.CORRELATION_ID,
        Memset.FLAGS,
        Memset.MEMORY_KIND,
        Memset.GRAPH_NODE_ID,
        Memset.GRAPH_ID,
    ),
    ActivityKind.RUNTIME: (
        Api.KIND,
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
        Api.RETURN_VALUE,
    ),
    ActivityKind.DRIVER: (
        Api.KIND,
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
        Api.RETURN_VALUE,
    ),
    ActivityKind.EXTERNAL_CORRELATION: (
        ExternalCorrelation.KIND,
        ExternalCorrelation.EXTERNAL_KIND,
        ExternalCorrelation.EXTERNAL_ID,
        ExternalCorrelation.CORRELATION_ID,
    ),
    ActivityKind.OVERHEAD: (
        Overhead.KIND,
        Overhead.OVERHEAD_KIND,
        Overhead.PROCESS_ID,
        Overhead.THREAD_ID,
        Overhead.START,
        Overhead.END,
        Overhead.CORRELATION_ID,
    ),
    ActivityKind.CUDA_EVENT: (
        CudaEvent.KIND,
        CudaEvent.CORRELATION_ID,
        CudaEvent.CONTEXT_ID,
        CudaEvent.STREAM_ID,
        CudaEvent.EVENT_ID,
        CudaEvent.DEVICE_ID,
        CudaEvent.DEVICE_TIMESTAMP,
        CudaEvent.CUDA_EVENT_SYNC_ID,
    ),
    ActivityKind.SYNCHRONIZATION: (
        Synchronization.KIND,
        Synchronization.TYPE,
        Synchronization.START,
        Synchronization.END,
        Synchronization.CORRELATION_ID,
        Synchronization.CONTEXT_ID,
        Synchronization.STREAM_ID,
        Synchronization.CUDA_EVENT_ID,
        Synchronization.CUDA_EVENT_SYNC_ID,
        Synchronization.RETURN_VALUE,
    ),
}

# kind -> frozenset of supported field ids; source of truth for validating observer
# requests and resolving "all". (Field is an int, so a Field is its id here.)
FIELD_REGISTRY: dict[int, frozenset[int]] = {
    kind: frozenset(fields) for kind, fields in FIELDS.items()
}

# kind -> {field id: Ctype}; how the decoder interprets each selected field's bytes
# (width comes from CUPTI's captured layout, not from here).
FIELD_CTYPE: dict[int, dict[int, Ctype]] = {
    kind: {f.id: f.ctype for f in fields} for kind, fields in FIELDS.items()
}

# kind -> frozenset of field ids that are const char* strings (dereferenced during decode).
STRING_FIELDS: dict[int, frozenset[int]] = {
    kind: frozenset(f.id for f in fields if f.string) for kind, fields in FIELDS.items()
}

# kind -> its CORRELATION_ID field id. The launch correlation id a kernel shares with its
# runtime call; used to join activity to external-correlation (eager annotation).
CORRELATION_FIELD: dict[int, int] = {
    ActivityKind.CONCURRENT_KERNEL: Kernel.CORRELATION_ID.id,
    ActivityKind.MEMCPY: Memcpy.CORRELATION_ID.id,
    ActivityKind.MEMCPY2: Memcpy2.CORRELATION_ID.id,
    ActivityKind.MEMSET: Memset.CORRELATION_ID.id,
    ActivityKind.RUNTIME: Api.CORRELATION_ID.id,
    ActivityKind.DRIVER: Api.CORRELATION_ID.id,
    ActivityKind.EXTERNAL_CORRELATION: ExternalCorrelation.CORRELATION_ID.id,
    ActivityKind.OVERHEAD: Overhead.CORRELATION_ID.id,
    ActivityKind.CUDA_EVENT: CudaEvent.CORRELATION_ID.id,
    ActivityKind.SYNCHRONIZATION: Synchronization.CORRELATION_ID.id,
}


# Per-kind graph-node-id field, for the graph annotation resolver: only the GPU-op kinds
# carry a graph_node_id (the field the resolver maps to a region name).
GRAPH_NODE_FIELD: dict[int, int] = {
    ActivityKind.CONCURRENT_KERNEL: Kernel.GRAPH_NODE_ID.id,
    ActivityKind.MEMCPY: Memcpy.GRAPH_NODE_ID.id,
    ActivityKind.MEMCPY2: Memcpy2.GRAPH_NODE_ID.id,
    ActivityKind.MEMSET: Memset.GRAPH_NODE_ID.id,
}


# A record layout as captured by CUPTI (pBufferCompleteInfo->ppRecordLayouts) and attached
# to a completed buffer by the native layer: a list of
# (kind, record_size, [(field_id, offset, size), ...]). This is what the monitor decodes
# against -- no spec/computed layout.
RecordLayouts = list[tuple[int, int, list[tuple[int, int, int]]]]
