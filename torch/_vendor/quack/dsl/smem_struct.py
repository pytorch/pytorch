# Copyright (c) 2026, Tri Dao.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-field smem-partition annotations for SharedStorage declarations.

A `cute.struct` lowers to ONE smem_alloca op carrying ONE `smem.partition_id`
attribute, so a single struct cannot mix RESERVED and USER fields at the IR
level. `Reserved[...]` + `@partitioned_struct` provide the single-declaration
sugar instead: at trace time the decorator splits the annotated class into two
plain cute.structs — fields wrapped in `Reserved[...]` go to a struct
allocated with `partition=SmemPartition.RESERVED` (low addresses, packing with
the pipeline mbarriers and the TMEM holding buf under the 1KB that is
otherwise alignment pad ahead of the 1024-aligned USER buffers), everything
else stays USER — and `.allocate(smem)` returns one namespace exposing all
fields uniformly.

    @partitioned_struct
    class SharedStorage:
        sdt: Reserved[spec.smem_struct(128)]   # RESERVED partition
        sX: X.smem_struct(1024)                # USER partition
        ...

    storage = SharedStorage.allocate(smem)
    storage.sdt, storage.sX                    # fields, regardless of partition

This is trace-time-only machinery (no monkey-patching of the DSL).
"""

from types import SimpleNamespace

import cutlass.cute as cute
from cutlass.utils import SmemPartition


class Reserved:
    """Annotation marker: allocate this field in the RESERVED smem partition."""

    def __init__(self, inner):
        self.inner = inner

    def __class_getitem__(cls, inner):
        return cls(inner)


class PartitionedStruct:
    """A SharedStorage declaration split by partition. Not a cute.struct itself:
    holds one cute.struct per partition and allocates/merges them."""

    def __init__(self, cls):
        annotations = dict(cls.__annotations__)
        reserved_ann = {k: v.inner for k, v in annotations.items() if isinstance(v, Reserved)}
        user_ann = {k: v for k, v in annotations.items() if not isinstance(v, Reserved)}
        self._user_struct = (
            cute.struct(type(cls.__name__, (), {"__annotations__": user_ann})) if user_ann else None
        )
        self._reserved_struct = (
            cute.struct(type(cls.__name__ + "Reserved", (), {"__annotations__": reserved_ann}))
            if reserved_ann
            else None
        )
        self._user_fields = list(user_ann)
        self._reserved_fields = list(reserved_ann)

    def size_in_bytes(self) -> int:
        """USER-partition footprint (what counts against smem_capacity - 1KB)."""
        return self._user_struct.size_in_bytes() if self._user_struct is not None else 0

    def reserved_size_in_bytes(self) -> int:
        """RESERVED-partition footprint of the declared fields (the pipeline
        mbarriers / TMEM holding buf allocate there separately)."""
        return self._reserved_struct.size_in_bytes() if self._reserved_struct is not None else 0

    def allocate(self, smem) -> SimpleNamespace:
        """Allocate both partitions (RESERVED first, at the partition base) and
        return a namespace exposing every declared field. A partition whose
        struct is empty for this config (every field zero-sized) is skipped —
        smem_alloca rejects 0-byte layouts — and its fields come back as None;
        callers only touch such fields under the same has_* guards that made
        them zero-sized."""
        fields = {}
        if self._reserved_struct is not None:
            if self._reserved_struct.size_in_bytes() > 0:
                inst = smem.allocate(self._reserved_struct, partition=SmemPartition.RESERVED)
                for name in self._reserved_fields:
                    fields[name] = getattr(inst, name)
            else:
                fields.update(dict.fromkeys(self._reserved_fields))
        if self._user_struct is not None:
            if self._user_struct.size_in_bytes() > 0:
                inst = smem.allocate(self._user_struct)
                for name in self._user_fields:
                    fields[name] = getattr(inst, name)
            else:
                fields.update(dict.fromkeys(self._user_fields))
        return SimpleNamespace(**fields)


def partitioned_struct(cls) -> PartitionedStruct:
    return PartitionedStruct(cls)
