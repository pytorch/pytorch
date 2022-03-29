import torch.distributed._shard.sharded_tensor._ops.elementwise_ops

from .binary_cmp import equal, allclose
from .embedding import sharded_embedding
from .embedding_bag import sharded_embedding_bag
from .init import kaiming_uniform_, normal_, uniform_, constant_
from .linear import sharded_linear
