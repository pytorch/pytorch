# Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed.shard` package.
import sys
import torch
import warnings

from torch.distributed.shard.sharding_spec import *  # noqa: F403
warnings.warn(
    "torch.distributed._sharding_spec will be deprecated, use torch.distributed.shard.sharding_spec instead",
    DeprecationWarning
)
sys.modules['torch.distributed._sharding_spec'] = torch.distributed.shard.sharding_spec
