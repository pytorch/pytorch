import torch.distributed._shard.sharded_tensor._ops.misc_ops
import torch.distributed._shard.sharded_tensor._ops.tensor_ops

from .binary_cmp import equal, allclose
from .init import kaiming_uniform_, normal_, uniform_, constant_

# Import all ChunkShardingSpec ops
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import sharded_embedding
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import sharded_embedding_bag
