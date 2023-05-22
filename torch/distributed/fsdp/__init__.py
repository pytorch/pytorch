import torch

from .flat_param import FlatParameter
from .fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateKeyType,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from .sharded_grad_scaler import _ShardedGradScaler

torch.cuda.amp.ShardedGradScaler = _ShardedGradScaler  # type: ignore[attr-defined]
del torch
del _ShardedGradScaler
