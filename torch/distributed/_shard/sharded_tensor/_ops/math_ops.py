import torch
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    sharded_op_impl
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.replicated_tensor import ReplicatedTensor

def register_math_op(op):
    @sharded_op_impl(op)
    def binary_math_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the elementwise op such
        as ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.
        This method computes on either a normal tensor or a sharded tensor.
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

            if lhs.size() == rhs.size() and lhs_spec.sharding_dim == rhs_spec.sharding_dim:
                # perform local element-wise math op
                res = op(lhs.local_tensor(), rhs.local_tensor)
                return ShardedTensor._init_from_local_tensor(res, lhs_spec, lhs.size(), process_group=pg)
            else:
                raise RuntimeError("Implicit broadcasting not supported yet!")

        elif isinstance(lhs, ReplicatedTensor):
            assert isinstance(rhs, ShardedTensor)
            if lhs.size() == rhs.local_tensor().size():
                res = op(lhs, rhs.local_tensor())
                return ShardedTensor._init_from_local_tensor(res, rhs.sharding_spec(), rhs.size(), process_group=pg)
            else:
                raise RuntimeError("Implicit broadcasting not supported yet!")

        elif isinstance(rhs, ReplicatedTensor):
            assert isinstance(lhs, ShardedTensor)
            if rhs.size() == lhs.local_tensor().size():
                res = op(lhs.local_tensor(), rhs)
                return ShardedTensor._init_from_local_tensor(res, lhs.sharding_spec(), lhs.size(), process_group=pg)
            else:
                raise RuntimeError("Implicit broadcasting not supported yet!")
        else:
            raise RuntimeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} not supported yet for ShardedTensor!")


register_math_op(torch.add)
register_math_op(Tensor.add)
register_math_op(torch.sub)
register_math_op(Tensor.sub)
register_math_op(Tensor.__rsub__)
register_math_op(torch.mul)
register_math_op(Tensor.mul)
register_math_op(torch.div)
register_math_op(Tensor.div)
register_math_op(Tensor.__rdiv__)
