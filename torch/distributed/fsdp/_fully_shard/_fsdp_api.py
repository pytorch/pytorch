# mypy: allow-untyped-defs
from dataclasses import dataclass

from torch.distributed.fsdp._flex_shard.all_gather_layout import AllGatherInput
from torch.distributed.fsdp._flex_shard.api import (
    _ReduceOp,  # noqa: F401
    AllGather,
    AllReduce,
    Comm,
    CPUOffloadPolicy,
    GradientReductionHandle,
    MixedPrecisionPolicy,
    OffloadPolicy,
    ReduceScatter,
    UnshardHandle,
)


__all__ = [
    "AllGather",
    "AllGatherInput",
    "AllReduce",
    "Comm",
    "CPUOffloadPolicy",
    "DataParallelMeshDims",
    "GradientReductionHandle",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "ReduceScatter",
    "UnshardHandle",
]


@dataclass
class DataParallelMeshDims:
    """
    Specifies which dimensions of a full SPMD :class:`DeviceMesh` correspond to
    data parallelism when using :func:`fully_shard` whose parameters are already
    DTensors on that mesh.

    Attributes:
        shard (Optional[Union[str, tuple[str, ...]]]): Mesh dimension name(s)
            that FSDP shards parameters on. If a tuple of names, those dims
            are flattened into a single shard dimension. At least one of
            ``shard`` and ``replicate`` must be set.
        replicate (Optional[Union[str, tuple[str, ...]]]): Mesh dimension
            name(s) for HSDP or DDP replication. If a tuple of names, those
            dims are flattened into a single replicate dimension.
    """

    shard: str | tuple[str, ...] | None = None
    replicate: str | tuple[str, ...] | None = None

    def __post_init__(self):
        if self.shard is None and self.replicate is None:
            raise ValueError(
                "At least one of shard or replicate must be set in DataParallelMeshDims"
            )

    @property
    def shard_names(self) -> tuple[str, ...]:
        if self.shard is None:
            return ()
        if isinstance(self.shard, str):
            return (self.shard,)
        return tuple(self.shard)

    @property
    def replicate_names(self) -> tuple[str, ...]:
        if self.replicate is None:
            return ()
        if isinstance(self.replicate, str):
            return (self.replicate,)
        return tuple(self.replicate)
