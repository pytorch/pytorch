# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CUPTI activity kinds and user-defined field ids as IntEnums.

Accessed as ``torch.profiler.cupti.types.KernelField.START`` etc. Members
are ints (IntEnum), so they pass straight to the CUPTI ctypes calls and
work as dict keys alongside the raw ids CUPTI reports.
"""

from __future__ import annotations

from enum import IntEnum


class ActivityKind(IntEnum):
    """CUpti_ActivityKind ids (the subset the mux supports)."""

    MEMCPY = 1
    MEMSET = 2
    CONCURRENT_KERNEL = 10


class KernelField(IntEnum):
    """CUpti_ActivityKernelFieldIds (KERNEL_FIELD_*); from cupti_activity.h,
    cross-checked against probe (START/END/NAME/GRAPH_NODE_ID)."""

    KIND = 0
    START = 7
    END = 8
    DEVICE_ID = 10
    CONTEXT_ID = 11
    STREAM_ID = 12
    CORRELATION_ID = 22
    NAME = 24  # const char* (mangled symbol); deref'd to a str during parse
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


# kind -> its field-id enum (members = the field ids the mux supports for it).
FIELD_ENUMS: dict[int, type[IntEnum]] = {
    ActivityKind.CONCURRENT_KERNEL: KernelField,
    ActivityKind.MEMCPY: MemcpyField,
    ActivityKind.MEMSET: MemsetField,
}

# kind -> frozenset of supported field ids; source of truth for validating
# observer requests and resolving ``"all"``. CUPTI requires *_FIELD_KIND
# (id 0) to be the first selected field, which the mux enforces at enable.
FIELD_REGISTRY: dict[int, frozenset[IntEnum]] = {
    kind: frozenset(enum) for kind, enum in FIELD_ENUMS.items()
}

# (kind -> field ids) whose record value is a ``const char*`` that must be
# dereferenced to a Python string DURING the buffer-completed callback (CUPTI
# frees the pointed-to string afterwards), rather than gathered as a numeric
# column. The mux delivers these as an object array of (mangled) strings.
# Verified via experiments/probe_name_field.py.
STRING_FIELDS: dict[int, frozenset[int]] = {
    ActivityKind.CONCURRENT_KERNEL: frozenset({KernelField.NAME}),
}

# Padding/filler candidates (see mux._plan_padding / enable_uniform_padding) are
# simply "every field id a kind has that the observers aren't already requesting"
# -- the delta between the requested set and the kind's full field space. We
# don't enumerate that space (it's not in FIELD_REGISTRY, which only gates what
# observers may request); instead the mux scans field ids up to this ceiling and
# DISCOVERS validity + real byte width at runtime from CUPTI's record layout (an
# id that isn't a real field reports size 0 and is pruned). So nothing here is a
# hand-picked or version-fragile id -- just an upper bound on the id space to
# probe. (Observed valid kernel field ids top out in the low 50s; see
# experiments/probe_kernel_fields.py.)
MAX_FIELD_ID: int = 64
