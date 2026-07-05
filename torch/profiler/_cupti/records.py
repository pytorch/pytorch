# mypy: allow-untyped-defs
"""CUPTI user-defined-record (v2) field schema.

In the v2 / user-defined-record path, observers select specific *fields* per
activity kind (rather than whole records). Each field is a :class:`Field` -- its
CUpti_Activity*FieldId ``id`` and whether it is a ``string`` (const char*) to
dereference during decode. The fields a kind supports are grouped in a small
per-kind class (``Kernel``, ``Sync``, ...); cupti-python does not expose these
field-id enums, so they are defined here from the CUPTI ABI (cupti_activity.h).

The monitor does NOT compute record byte layouts: it requires libcupti >= 13.3,
which reports each kind's packed record layout (field offsets/sizes, record size)
via ``pBufferCompleteInfo->ppRecordLayouts``. The native layer parses that and
attaches it to each completed buffer; :func:`decode` (below) demuxes a buffer against
that captured layout. So this module only needs to know which field ids exist and
which are strings.
"""

from __future__ import annotations

from dataclasses import dataclass

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]


@dataclass(frozen=True)
class Field:
    """One field of a CUPTI user-defined record: its CUpti_Activity*FieldId ``id``
    and whether it is a ``string`` (const char*) to dereference to a Python str
    during decode. ``int(field)`` is the id, so a field is usable directly as a
    selection element or a column key. Byte offset/size come from CUPTI's captured
    record layout at decode time, not from here."""

    id: int
    string: bool = False

    def __int__(self) -> int:
        return self.id


# Per-kind field catalogs. Each class lists the fields the v2 monitor supports for a
# kind; the *_FIELD_KIND field (id 0) leads every record. Ids are CUPTI ABI
# constants (cupti_activity.h).
class Kernel:
    """CUpti_ActivityKernel (CONCURRENT_KERNEL)."""

    KIND = Field(0)
    REGISTERS_PER_THREAD = Field(4)
    START = Field(7)
    END = Field(8)
    DEVICE_ID = Field(10)
    CONTEXT_ID = Field(11)
    STREAM_ID = Field(12)
    GRID_X = Field(13)
    GRID_Y = Field(14)
    GRID_Z = Field(15)
    BLOCK_X = Field(16)
    BLOCK_Y = Field(17)
    BLOCK_Z = Field(18)
    STATIC_SHARED_MEMORY = Field(19)
    DYNAMIC_SHARED_MEMORY = Field(20)
    CORRELATION_ID = Field(22)
    NAME = Field(24, string=True)  # const char* (mangled symbol)
    GRAPH_NODE_ID = Field(31)
    GRAPH_ID = Field(33)
    LAUNCH_PRIORITY = Field(45)
    QUEUED = Field(25)
    CHANNEL_ID = Field(35)
    CHANNEL_TYPE = Field(36)


class Memcpy:
    """CUpti_ActivityMemcpy (MEMCPY)."""

    KIND = Field(0)
    COPY_KIND = Field(1)
    SRC_KIND = Field(2)
    DST_KIND = Field(3)
    FLAGS = Field(4)
    BYTES = Field(5)
    START = Field(6)
    END = Field(7)
    DEVICE_ID = Field(8)
    CONTEXT_ID = Field(9)
    STREAM_ID = Field(10)
    CORRELATION_ID = Field(11)
    GRAPH_NODE_ID = Field(12)
    GRAPH_ID = Field(13)


class Memset:
    """CUpti_ActivityMemset (MEMSET)."""

    KIND = Field(0)
    VALUE = Field(1)
    BYTES = Field(2)
    START = Field(3)
    END = Field(4)
    DEVICE_ID = Field(5)
    CONTEXT_ID = Field(6)
    STREAM_ID = Field(7)
    CORRELATION_ID = Field(8)
    FLAGS = Field(9)
    MEMORY_KIND = Field(10)
    GRAPH_NODE_ID = Field(11)
    GRAPH_ID = Field(12)


class Memcpy2:
    """CUpti_ActivityMemcpyPtoP4 (MEMCPY2) -- peer-to-peer / cross-device copy. Like Memcpy but
    with src/dst device+context fields inserted, which shifts correlation/graph ids."""

    KIND = Field(0)
    COPY_KIND = Field(1)
    SRC_KIND = Field(2)
    DST_KIND = Field(3)
    FLAGS = Field(4)
    BYTES = Field(5)
    START = Field(6)
    END = Field(7)
    DEVICE_ID = Field(8)
    CONTEXT_ID = Field(9)
    STREAM_ID = Field(10)
    SRC_DEVICE_ID = Field(11)
    DST_DEVICE_ID = Field(13)
    CORRELATION_ID = Field(15)
    GRAPH_NODE_ID = Field(16)
    GRAPH_ID = Field(17)


class Api:
    """CUpti_ActivityAPI -- shared by RUNTIME and DRIVER."""

    KIND = Field(0)
    CBID = Field(1)
    START = Field(2)
    END = Field(3)
    PROCESS_ID = Field(4)
    THREAD_ID = Field(5)
    CORRELATION_ID = Field(6)
    RETURN_VALUE = Field(7)


class ExternalCorrelation:
    """CUpti_ActivityExternalCorrelation (EXTERNAL_CORRELATION)."""

    KIND = Field(0)
    EXTERNAL_KIND = Field(1)
    EXTERNAL_ID = Field(2)
    CORRELATION_ID = Field(3)


class Overhead:
    """CUpti_ActivityOverhead (OVERHEAD)."""

    KIND = Field(0)
    OVERHEAD_KIND = Field(1)
    PROCESS_ID = Field(2)
    THREAD_ID = Field(3)
    START = Field(4)
    END = Field(5)
    CORRELATION_ID = Field(6)


class CudaEvent:
    """CUpti_ActivityCudaEvent (CUDA_EVENT)."""

    KIND = Field(0)
    CORRELATION_ID = Field(1)
    CONTEXT_ID = Field(2)
    STREAM_ID = Field(3)
    EVENT_ID = Field(4)
    DEVICE_ID = Field(5)
    DEVICE_TIMESTAMP = Field(6)
    CUDA_EVENT_SYNC_ID = Field(7)


class Sync:
    """CUpti_ActivitySynchronization (SYNCHRONIZATION)."""

    KIND = Field(0)
    TYPE = Field(1)
    START = Field(2)
    END = Field(3)
    CORRELATION_ID = Field(4)
    CONTEXT_ID = Field(5)
    STREAM_ID = Field(6)
    CUDA_EVENT_ID = Field(7)
    CUDA_EVENT_SYNC_ID = Field(8)
    RETURN_VALUE = Field(9)


def _catalog(cls: type) -> tuple[Field, ...]:
    """The Fields declared on a per-kind catalog class, in declaration order."""
    return tuple(v for v in vars(cls).values() if isinstance(v, Field))


# kind -> its fields (the fields the v2 monitor supports for that kind).
FIELDS: dict[int, tuple[Field, ...]] = {
    ActivityKind.CONCURRENT_KERNEL: _catalog(Kernel),
    ActivityKind.MEMCPY: _catalog(Memcpy),
    ActivityKind.MEMCPY2: _catalog(Memcpy2),
    ActivityKind.MEMSET: _catalog(Memset),
    ActivityKind.RUNTIME: _catalog(Api),
    ActivityKind.DRIVER: _catalog(Api),
    ActivityKind.EXTERNAL_CORRELATION: _catalog(ExternalCorrelation),
    ActivityKind.OVERHEAD: _catalog(Overhead),
    ActivityKind.CUDA_EVENT: _catalog(CudaEvent),
    ActivityKind.SYNCHRONIZATION: _catalog(Sync),
}

# kind -> frozenset of supported field ids; source of truth for validating observer
# requests and resolving "all". CUPTI requires *_FIELD_KIND (id 0) first at enable.
FIELD_REGISTRY: dict[int, frozenset[int]] = {
    kind: frozenset(f.id for f in fields) for kind, fields in FIELDS.items()
}

# kind -> frozenset of field ids that are const char* strings (dereferenced during
# decode rather than gathered as a numeric column).
STRING_FIELDS: dict[int, frozenset[int]] = {
    kind: frozenset(f.id for f in fields if f.string) for kind, fields in FIELDS.items()
}

# kind -> its CORRELATION_ID field id. The launch correlation id a kernel shares with
# its runtime call; used to join activity to external-correlation (eager annotation).
CORRELATION_FIELD: dict[int, int] = {
    ActivityKind.CONCURRENT_KERNEL: int(Kernel.CORRELATION_ID),
    ActivityKind.MEMCPY: int(Memcpy.CORRELATION_ID),
    ActivityKind.MEMCPY2: int(Memcpy2.CORRELATION_ID),
    ActivityKind.MEMSET: int(Memset.CORRELATION_ID),
    ActivityKind.RUNTIME: int(Api.CORRELATION_ID),
    ActivityKind.DRIVER: int(Api.CORRELATION_ID),
    ActivityKind.EXTERNAL_CORRELATION: int(ExternalCorrelation.CORRELATION_ID),
    ActivityKind.OVERHEAD: int(Overhead.CORRELATION_ID),
    ActivityKind.CUDA_EVENT: int(CudaEvent.CORRELATION_ID),
    ActivityKind.SYNCHRONIZATION: int(Sync.CORRELATION_ID),
}


# Per-kind graph-node-id field, for the graph annotation resolver: only the GPU-op kinds
# carry a graph_node_id (the field the resolver maps to a region name).
GRAPH_NODE_FIELD: dict[int, int] = {
    ActivityKind.CONCURRENT_KERNEL: int(Kernel.GRAPH_NODE_ID),
    ActivityKind.MEMCPY: int(Memcpy.GRAPH_NODE_ID),
    ActivityKind.MEMCPY2: int(Memcpy2.GRAPH_NODE_ID),
    ActivityKind.MEMSET: int(Memset.GRAPH_NODE_ID),
}


# A record layout as captured by CUPTI (pBufferCompleteInfo->ppRecordLayouts) and
# attached to a completed buffer by the native layer: a list of
# (kind, record_size, [(field_id, offset, size), ...]). This is what the monitor
# decodes against -- no spec/computed layout.
RecordLayouts = list[tuple[int, int, list[tuple[int, int, int]]]]
