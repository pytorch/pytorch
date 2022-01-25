# Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed.shard` package.
import sys
import torch.distributed.shard.sharded_tensor
import warnings
warnings.warn(
    "torch.distributed._sharded_tensor will be deprecated, use torch.distributed.shard.sharded_tensor instead",
    DeprecationWarning
)
sys.modules['torch.distributed._sharded_tensor'] = torch.distributed.shard.sharded_tensor
