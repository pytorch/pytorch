import torch.distributed._shard.sharded_tensor._ops.elementwise_ops
import torch.distributed._shard.sharded_tensor._ops.math_ops

from .binary_cmp import equal, allclose
from .chunk import sharded_chunk
from .embedding import sharded_embedding
from .embedding_bag import sharded_embedding_bag
from .init import kaiming_uniform_, normal_, uniform_, constant_
from .linear import sharded_linear
from .matrix_ops import sharded_bmm, sharded_softmax, sharded_dropout
