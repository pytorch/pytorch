# mypy: allow-untyped-defs
"""CUPTI user-defined-record (v2) field schema.

In the v2 / user-defined-record path, observers select specific *fields* per activity
kind (rather than whole records). The available fields per kind are the generated
:mod:`_cupti_stubs` catalogs (``Kernel``, ``Memcpy``, ...), one
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

from torch.profiler._cupti._cupti_stubs import (
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


def _all_fields(catalog: type) -> tuple[Field, ...]:
    """Every :class:`Field` on a generated catalog class, in id order. ``sorted`` puts KIND
    (id 0) first, which CUPTI requires (``*_FIELD_KIND`` must lead at enable). Selecting all
    of them and letting the decoder drop fields whose captured size is not 1/2/4/8 yields
    all *decodable* fields, with no hand-maintained per-field list to drift from the header."""
    return tuple(sorted(v for v in vars(catalog).values() if isinstance(v, Field)))


# The only curation: which activity kinds the monitor supports and their generated catalog.
# RUNTIME and DRIVER share the Api catalog. FIELDS / CORRELATION_FIELD / GRAPH_NODE_FIELD all
# derive from this, so adding a kind here is the single edit needed.
_CATALOGS: dict[int, type] = {
    ActivityKind.CONCURRENT_KERNEL: Kernel,
    ActivityKind.MEMCPY: Memcpy,
    ActivityKind.MEMCPY2: Memcpy2,
    ActivityKind.MEMSET: Memset,
    ActivityKind.RUNTIME: Api,
    ActivityKind.DRIVER: Api,
    ActivityKind.EXTERNAL_CORRELATION: ExternalCorrelation,
    ActivityKind.OVERHEAD: Overhead,
    ActivityKind.CUDA_EVENT: CudaEvent,
    ActivityKind.SYNCHRONIZATION: Synchronization,
}

# kind -> the fields the v2 monitor selects: all decodable fields of the generated catalog
# (struct/opaque fields are dropped at decode by their size).
FIELDS: dict[int, tuple[Field, ...]] = {
    kind: _all_fields(cat) for kind, cat in _CATALOGS.items()
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
# runtime call; used to join activity to external-correlation (eager annotation). Only kinds
# whose catalog carries the field are included.
CORRELATION_FIELD: dict[int, int] = {
    kind: cat.CORRELATION_ID.id
    for kind, cat in _CATALOGS.items()
    if hasattr(cat, "CORRELATION_ID")
}


# Per-kind graph-node-id field, for the graph annotation resolver: only the GPU-op kinds
# carry a graph_node_id (the field the resolver maps to a region name).
GRAPH_NODE_FIELD: dict[int, int] = {
    kind: cat.GRAPH_NODE_ID.id
    for kind, cat in _CATALOGS.items()
    if hasattr(cat, "GRAPH_NODE_ID")
}


# A record layout as captured by CUPTI (pBufferCompleteInfo->ppRecordLayouts) and attached
# to a completed buffer by the native layer: a list of
# (kind, record_size, [(field_id, offset, size), ...]). This is what the monitor decodes
# against -- no spec/computed layout.
RecordLayouts = list[tuple[int, int, list[tuple[int, int, int]]]]
