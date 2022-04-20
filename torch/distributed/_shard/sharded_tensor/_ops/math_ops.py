import torch
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    sharded_op_impl
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.replicated_tensor import ReplicatedTensor

from ._common import (
    _chunk_sharding_spec_check,
    _register_sharded_op_on_local_tensor,
)


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


def sharded_bmm_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_bmm op, for example, mat2 needs to
    be a sharded tensor and both tensors need to sharded by dim 0, etc.

    Args: same as ``torch.bmm``.

    Return: None
    """
    if len(args) < 2:
        raise TypeError("Needs two tensors to perform torch.bmm.")
    st = args[0]
    mat2 = args[1]
    # Validate types
    if not isinstance(mat2, ShardedTensor):
        raise TypeError("mat2 needs to be a ShardedTensor for torch.bmm.")
    _chunk_sharding_spec_check(mat2.sharding_spec(), torch.bmm)
    if st.dim() != 3 or mat2.dim() != 3:
        raise TypeError("both st and mat2 need to be a 3D ShardedTensor")
    if (
        st.sharding_spec().dim != mat2.sharding_spec().dim
        or st.sharding_spec().dim != 0
    ):
        raise NotImplementedError(
            "Only support performing bmm on tensors sharded on dim 0 now."
        )
    if st.sharding_spec().placements != mat2.sharding_spec().placements:
        raise NotImplementedError(
            "Both st and mat2 need to have same placements for bmm."
        )


def sharded_bmm(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the sharded_bmm op.

    Warning: For now we only supports the case when both tensors are sharded
             by dim 0 so that no local communication.

    Args: same as ``torch.bmm``.

    Return:
        local_tensor (Tensor): New local tensor to build the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            sharding spec of the new sharded tensor.
        new_st_size (torch.Size): Size of the new sharded tensor.
    """
    st = args[0]
    mat2 = args[1]
    local_tensor = torch.bmm(st.local_tensor(), mat2.local_tensor())
    new_st_size = (*st.size()[:-1], mat2.size(-1))
    return local_tensor, st.sharding_spec(), new_st_size


_register_sharded_op_on_local_tensor(
    torch.Tensor.bmm,
    extra_check=sharded_bmm_check,
    customized_func=sharded_bmm,
)

_register_sharded_op_on_local_tensor(
    torch.bmm,
    extra_check=sharded_bmm_check,
    customized_func=sharded_bmm,
)
