from ._flat_param import FlatParameter as FlatParameter
from ._fully_shard import (
    CPUOffloadPolicy,
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
    register_fsdp_forward_method,
    UnshardHandle,
)
from .fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    OptimStateKeyType,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)


__all__ = [
    # FSDP1
    "BackwardPrefetch",
    "CPUOffload",
    "FullOptimStateDictConfig",
    "FullStateDictConfig",
    "FullyShardedDataParallel",
    "LocalOptimStateDictConfig",
    "LocalStateDictConfig",
    "MixedPrecision",
    "OptimStateDictConfig",
    "OptimStateKeyType",
    "ShardedOptimStateDictConfig",
    "ShardedStateDictConfig",
    "ShardingStrategy",
    "StateDictConfig",
    "StateDictSettings",
    "StateDictType",
    # FSDP2
    "CPUOffloadPolicy",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
]

# Set namespace for exposed private names
CPUOffloadPolicy.__module__ = "torch.distributed.fsdp"
FSDPModule.__module__ = "torch.distributed.fsdp"
fully_shard.__module__ = "torch.distributed.fsdp"
MixedPrecisionPolicy.__module__ = "torch.distributed.fsdp"
OffloadPolicy.__module__ = "torch.distributed.fsdp"
register_fsdp_forward_method.__module__ = "torch.distributed.fsdp"
UnshardHandle.__module__ = "torch.distributed.fsdp"
