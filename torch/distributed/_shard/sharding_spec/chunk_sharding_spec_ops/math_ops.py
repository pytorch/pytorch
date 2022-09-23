import torch
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed._shard.sharded_tensor._ops.math_ops import binary_math_op_impl

from ._common import (
    _chunk_sharding_spec_check,
)

def register_math_op(op):
    @custom_sharding_spec_op(ChunkShardingSpec, op)
    def binary_math_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the binary math ops
        such as `torch.add`, `torch.mul`, `torch.div`, etc.
        This method computes on ShardedTensor
        """
        if len(args) != 2:
            raise ValueError("Only support binary math op on ShardedTensor for now!")
        lhs = args[0]
        rhs = args[1]
        # Validate types
        if isinstance(lhs, ShardedTensor) and isinstance(rhs, ShardedTensor):
            lhs_spec = lhs.sharding_spec()
            rhs_spec = rhs.sharding_spec()
            _chunk_sharding_spec_check(lhs_spec, op)
            _chunk_sharding_spec_check(rhs_spec, op)

            if lhs.size() == rhs.size() and lhs_spec.dim == rhs_spec.dim:  # type: ignore[attr-defined]
                # perform local element-wise math op
                res = op(lhs.local_tensor(), rhs.local_tensor())
                return ShardedTensor._init_from_local_tensor(
                    res,
                    lhs_spec,
                    lhs.size(),  # type: ignore[arg-type]
                    process_group=pg)
            else:
                raise RuntimeError("Implicit broadcasting not supported yet!")
        else:
            # Try dispatch to ShardingSpec agnostic ops.
            return binary_math_op_impl(op, types, args, kwargs, pg)

binary_ops = [
    # add
    torch.add,
    Tensor.add,
    Tensor.__add__,
    Tensor.__radd__,
    # sub
    torch.sub,
    Tensor.sub,
    Tensor.__sub__,
    Tensor.__rsub__,
    # mul
    torch.mul,
    Tensor.mul,
    Tensor.__mul__,
    Tensor.__rmul__,
    # div
    torch.div,
    Tensor.div,
    Tensor.__div__,
    Tensor.__rdiv__,
]

for op in binary_ops:
    register_math_op(op)
