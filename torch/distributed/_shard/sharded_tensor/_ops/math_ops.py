import torch
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    sharded_op_impl
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.replicated_tensor import ReplicatedTensor

from torch.distributed._shard._utils import narrow_tensor

def register_math_op(op):
    @sharded_op_impl(op)
    def binary_math_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the binary math ops
        such as `torch.add`, `torch.mul`, `torch.div`, etc.
        This method computes on ShardedTensor, or ShardedTensor op ReplicatedTensor
        """
        if len(args) != 2:
            raise ValueError("Only support binary math op on ShardedTensor for now!")
        lhs = args[0]
        rhs = args[1]
        # Validate types
        if isinstance(lhs, ShardedTensor) and isinstance(rhs, ShardedTensor):
            lhs_spec = lhs.sharding_spec()
            rhs_spec = rhs.sharding_spec()
            if not isinstance(lhs_spec, ChunkShardingSpec) or not isinstance(rhs_spec, ChunkShardingSpec):
                raise TypeError("Only ShardedTensor with ChunkShardingSpec supports"
                                " two ShardedTensor together")

            if lhs.size() == rhs.size() and lhs_spec.dim == rhs_spec.dim:
                # perform local element-wise math op
                res = op(lhs.local_tensor(), rhs.local_tensor())
                return ShardedTensor._init_from_local_tensor(
                    res,
                    lhs_spec,
                    lhs.size(),  # type: ignore[arg-type]
                    process_group=pg)
            else:
                raise RuntimeError("Implicit broadcasting not supported yet!")

        elif isinstance(lhs, ReplicatedTensor):
            assert isinstance(rhs, ShardedTensor)
            st_size = rhs.size()
            st_meta = rhs.local_shards()[0].metadata
            if st_size != lhs.size():
                # try to broadcast replicated tensor
                lhs = lhs.expand(st_size)

            replica_part = narrow_tensor(lhs, st_meta)
            res = op(replica_part, rhs.local_tensor())

            return ShardedTensor._init_from_local_tensor(
                res,
                rhs.sharding_spec(),
                rhs.size(),  # type: ignore[arg-type]
                process_group=pg)

        elif isinstance(rhs, ReplicatedTensor):
            assert isinstance(lhs, ShardedTensor)
            st_size = lhs.size()
            st_meta = lhs.local_shards()[0].metadata
            if st_size != rhs.size():
                # try to broadcast replicated tensor
                rhs = rhs.expand(st_size)

            replica_part = narrow_tensor(rhs, st_meta)
            res = op(lhs.local_tensor(), replica_part)
            return ShardedTensor._init_from_local_tensor(
                res,
                lhs.sharding_spec(),
                lhs.size(),  # type: ignore[arg-type]
                process_group=pg)

        elif isinstance(lhs, (int, float)):
            assert isinstance(rhs, ShardedTensor)
            res = op(lhs, rhs.local_tensor())
            return ShardedTensor._init_from_local_tensor(
                res,
                rhs.sharding_spec(),
                rhs.size(),  # type: ignore[arg-type]
                process_group=pg)

        elif isinstance(rhs, (int, float)):
            assert isinstance(lhs, ShardedTensor)
            res = op(lhs.local_tensor(), rhs)
            return ShardedTensor._init_from_local_tensor(
                res,
                lhs.sharding_spec(),
                lhs.size(),  # type: ignore[arg-type]
                process_group=pg)
        else:
            raise RuntimeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} not supported yet for ShardedTensor!")

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
