import torch
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec


def register_chunk_op(op):
    @_sharded_op_impl(op)
    def sharded_chunk(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the chunk op.
        If we chunk by the non-sharding dim, we just directly chunk the
        local tensor and create a list of sharded tensor based on them.

        Warnings: Chunk by the sharding dim is not supported.

        Args: same as ``torch.chunk``.

        Return:
            List[ShardedTensor]: Chunk results as a list of ShardedTensor.
        """
        st = args[0]
        chunk_num = args[1]
        dim = kwargs.get("dim")
        dim = dim if dim else 0

        # Validate types
        if not isinstance(st, ShardedTensor):
            raise TypeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} are called for non ShardedTensor!"
            )
        spec = st.sharding_spec()
        if not isinstance(spec, ChunkShardingSpec):
            raise NotImplementedError("Only ChunkShardingSpec is supported for chunk.")
        if spec.dim == dim or st.dim() + spec.dim == dim or st.dim() + dim == spec.dim:  # type: ignore[operator]
            raise NotImplementedError("Chunk by sharding dim is not supported.")

        local_tensor = st.local_tensor()
        st_size = st.size()
        dim = dim if dim > 0 else st.dim() + dim
        results = []
        for chunk_tensor in local_tensor.chunk(chunk_num, dim=dim):
            new_st_size = (*st_size[:dim], chunk_tensor.size(dim), *st_size[dim + 1 :])  # type: ignore[index]
            results.append(
                ShardedTensor._init_from_local_tensor(
                    chunk_tensor.contiguous(),
                    st.sharding_spec(),
                    new_st_size,
                    process_group=pg,
                )
            )
        return results


chunk_ops = [
    torch.chunk,
    torch.Tensor.chunk,
]
for op in chunk_ops:
    register_chunk_op(op)
