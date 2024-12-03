from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fully_shard import (
    FSDPModule,
    fully_shard,
    register_fsdp_forward_method,
    UnshardHandle,
)


__all__ = [
    "CPUOffloadPolicy",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
]

# Set namespace for exposed private names
CPUOffloadPolicy.__module__ = "torch.distributed.fsdp._fully_shard"
MixedPrecisionPolicy.__module__ = "torch.distributed.fsdp._fully_shard"
OffloadPolicy.__module__ = "torch.distributed.fsdp._fully_shard"
