"""
The following example demonstrates how to represent torchrec's embedding
sharding with the DTensor API.
"""
import argparse
import os
from typing import List, Tuple

import torch

from torch.distributed._tensor import DTensor, init_device_mesh, Shard
from torch.distributed._tensor.debug.visualize_sharding import visualize_sharding


class TorchrecLocalShards(torch.Tensor):
    _local_shards: List[torch.Tensor]
    _shard_offsets: List[Tuple[int, ...]]
    _shard_sizes: List[torch.Size]
    __slots__ = ["_local_shards", "_shard_offsets", "_shard_sizes"]

    @staticmethod
    def __new__(
        cls,
        local_tensors: List[torch.Tensor],
        shard_offsets: List[Tuple[int, ...]],
        shard_sizes: List[torch.Size],
    ) -> "TorchrecLocalShards":
        # requirement 1: do not support zero shard
        assert len(local_tensors) > 0

        # requirement 2: all shards are placed on the same device
        device = local_tensors[0].device
        assert all(t.device == device for t in local_tensors)

        # requirement 3: all shards should have the same dtype
        dtype = local_tensors[0].dtype
        assert all(t.dtype == dtype for t in local_tensors)

        # requirement 4: all shards should have the same grad setting
        requires_grad = local_tensors[0].requires_grad
        assert all(t.requires_grad == requires_grad for t in local_tensors)

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            torch.Size([0]),  # this attribute is required but shouldn't be used
            device=device,
            dtype=dtype,
            layout=local_tensors[0].layout,
            requires_grad=requires_grad,
        )

        r._local_shards = local_tensors
        r._shard_offsets = shard_offsets
        r._shard_sizes = shard_sizes
        return r

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # so far, only view_as() can be performed
        # TODO: allow _c10d_functional::all_gather_into_tensor()
        local_shards = args[0]._local_shards
        res_list = [func(t, t.shape, **kwargs) for t in local_shards]
        return TorchrecLocalShards(
            res_list,
            args[0]._shard_offsets,
            args[0]._shard_sizes,
        )


def get_device_type():
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


def run_torchrec_row_wise_sharding_example(rank, world_size):
    # row-wise example:
    #   one table is sharded by rows within the global ProcessGroup
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, so each rank will have one 2 by 16 local
    #   shard.

    # device mesh is a representation of the worker ranks
    # create a 1-D device mesh that includes every rank
    device_type = get_device_type()
    device = torch.device(device_type)
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])
    local_shard_shape = torch.Size(
        [num_embeddings // world_size, embedding_dim]  # (local_rows, local_cols)
    )
    # in our case, the embedding table will be sharded row-wisely into 4 local shards
    # and each rank will have 1 of them.
    local_shards = [
        torch.randn(local_shard_shape, device=device) for _ in range(world_size)
    ]
    # row-wise sharding: one shard per rank
    local_shard = local_shards[rank]

    ###########################################################################
    # example 1: transform local_shards into DTensor
    # usage in TorchRec:
    #   ShardedEmbeddingCollection stores model parallel params in
    #   _model_parallel_name_to_sharded_tensor which is initialized in
    #   _initialize_torch_state() and torch.Tensor params are transformed
    #   into ShardedTensor by ShardedTensor._init_from_local_shards().
    #
    #   This allows state_dict() to always return ShardedTensor objects.

    # this is the sharding placement we use in DTensor to represent row-wise sharding
    # row_wise_sharding_placements means that the global tensor is sharded by first dim
    # over the 1-d mesh.
    row_wise_sharding_placements = [Shard(0)]
    # create a DTensor from the local shard
    dtensor = DTensor.from_local(
        local_shard, device_mesh, row_wise_sharding_placements, run_check=False
    )

    # display the DTensor's sharding
    visualize_sharding(dtensor, header="Row-wise sharding example in DTensor")

    # get the global tensor from the DTensor
    dtensor_full = dtensor.full_tensor()  # torch.Tensor
    # manually compose the global tensor from the local shards
    global_tensor = torch.cat(local_shards, dim=0)
    # we demonstrate that the DTensor constructed has the same
    # global view as the actual global tensor
    assert torch.equal(dtensor_full, global_tensor)

    ###########################################################################
    # example 2: transform DTensor into local_shards
    # usage in TorchRec:
    #   In ShardedEmbeddingCollection's load_state_dict pre hook
    #   _pre_load_state_dict_hook, if the source param is a ShardedTensor
    #   then we need to transform it into its local_shards.

    # transform DTensor into local_shards
    local_shard = dtensor.to_local()

    # another case is that the source param is a torch.Tensor. In this case,
    # the source param is the global tensor rather than shards so we need to
    # splice the global tensor into local shards. This will be identical to
    # existing code in TorchRec.
    local_shard_shape_list = list(local_shard_shape)
    local_shard_spliced = global_tensor[
        local_shard_shape_list[0] * rank : local_shard_shape_list[0] * (rank + 1),
        :,
    ]
    # the local shard obtained from both approaches should be identical
    assert torch.equal(local_shard, local_shard_spliced)

    ###########################################################################
    # example 3: load state dict
    # usage in TorchRec:
    #   In case where the source param and the destination param are both
    #   DTensors, we can directly call DTensor.copy_() to load the state.
    src_dtensor = torch.distributed._tensor.ones(
        emb_table_shape,
        device_mesh=device_mesh,
        placements=row_wise_sharding_placements,
    )
    dtensor.copy_(src_dtensor)
    # these two DTensors should have the same global view after loading
    assert torch.equal(dtensor.full_tensor(), src_dtensor.full_tensor())


def run_torchrec_col_wise_even_sharding_example(rank, world_size):
    # col-wise even sharding example:
    #   One table is evenly sharded by columns within the global ProcessGroup.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, so each rank will have two 8 by 2 local
    #   shards.

    # device mesh is a representation of the worker ranks
    # create a 1-D device mesh that includes every rank
    device_type = get_device_type()
    device = torch.device(device_type)
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])
    num_shards = 2 * world_size
    local_shard_shape = torch.Size(
        [num_embeddings, embedding_dim // num_shards]  # (local_rows, local_cols)
    )
    # in our case, the embedding table will be sharded column-wisely into 8 local shards
    # and each rank will have 2 of them by the following placement:
    #   [0, 1, 2, 3, 0, 1, 2, 3] where the number is the rank to place each shard
    local_shards = [
        torch.randn(local_shard_shape, device=device) for _ in range(num_shards)
    ]
    local_shards_on_rank = {
        rank: [local_shards[rank], local_shards[rank + world_size]]
        for rank in range(world_size)
    }
    local_shard_offsets = [
        [
            (0, local_shard_shape[1] * rank),
            (0, local_shard_shape[1] * (rank + world_size)),
        ]
        for rank in range(world_size)
    ]
    ###########################################################################
    # example 1: transform local_shards into DTensor
    # col_wise_sharding_placements means that the global tensor is sharded by second dim
    # note: to shard by last dim, the placement needs be Shard(-1)
    col_wise_sharding_placements = [Shard(1)]
    # we need to first construct a TorchrecLocalShards object
    torchrec_local_shards = TorchrecLocalShards(
        local_shards_on_rank[rank],
        local_shard_offsets[rank],  # shard offset
        [local_shard_shape] * 2,  # shard sizes
    )
    # create a DTensor from the local shard
    dtensor = DTensor.from_local(
        torchrec_local_shards,
        device_mesh,
        col_wise_sharding_placements,
        run_check=False,
        shape=emb_table_shape,  # shape and stride are required for col-wise sharding
        stride=(1, num_embeddings),
    )
    for x, y in zip(dtensor._local_tensor._local_shards, local_shards_on_rank[rank]):
        assert torch.equal(x, y)

    # display the DTensor's sharding
    # TODO: need fix
    visualize_sharding(dtensor, header="Col-wise even sharding example in DTensor")

    exit()
    # TODO: need to support to_local()
    for x, y in zip(dtensor.to_local(), local_shards_on_rank[rank]):
        assert torch.equal(x, y)

    # TODO: need to support all-gather
    # get the global tensor from the DTensor
    dtensor_full = dtensor.full_tensor()  # torch.Tensor
    # manually compose the global tensor from the local shards
    global_tensor = torch.cat(local_shards, dim=1)
    # we demonstrate that the DTensor constructed has the same
    # global view as the actual global tensor
    assert torch.equal(dtensor_full, global_tensor)

    ###########################################################################
    # example 2: transform DTensor into local_shards

def run_example(rank, world_size, example_name):
    # the dict that stores example code
    name_to_example_code = {
        "row-wise": run_torchrec_row_wise_sharding_example,
        "col-wise-even": run_torchrec_col_wise_even_sharding_example,
    }
    if example_name not in name_to_example_code:
        print(f"example for {example_name} does not exist!")
        return

    # the example to run
    example_func = name_to_example_code[example_name]

    # set manual seed
    torch.manual_seed(0)

    # run the example
    example_func(rank, world_size)


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4  # our example uses 4 worker ranks
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="torchrec sharding examples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    example_prompt = (
        "choose one sharding example from below:\n"
        "\t1. row-wise;\n"
        "\t4. col-wise-even\n"
        "e.g. you want to try the row-wise sharding example, please input 'row-wise'\n"
    )
    parser.add_argument("-e", "--example", help=example_prompt, required=True)
    args = parser.parse_args()
    run_example(rank, world_size, args.example)
