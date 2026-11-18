# mypy: allow-untyped-defs
"""
The following example demonstrates how to represent torchrec's embedding
sharding with the DTensor API.
"""

import argparse
import os
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import visualize_sharding


if TYPE_CHECKING:
    from torch.distributed.tensor.placement_types import Placement


def get_device_type():
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


aten = torch.ops.aten
supported_ops = [aten.view.default, aten._to_copy.default]


# this torch.Tensor subclass is a wrapper around all local shards associated
# with a single sharded embedding table.
class LocalShardsWrapper(torch.Tensor):
    local_shards: list[torch.Tensor]
    storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], offsets: list[torch.Size]
    ) -> "LocalShardsWrapper":
        if len(local_shards) <= 0:
            raise AssertionError
        if len(local_shards) != len(offsets):
            raise AssertionError
        if local_shards[0].ndim != 2:
            raise AssertionError
        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].shape)
        if len(local_shards) > 1:  # column-wise sharding
            for shard_size in [s.shape for s in local_shards[1:]]:
                cat_tensor_shape[1] += shard_size[1]

        # according to DCP, each chunk is expected to have the same properties of the
        # TensorStorageMetadata that includes it. Vice versa, the wrapper's properties
        # should also be the same with that of its first chunk.
        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(o, s.shape) for s, o in zip(local_shards, offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            wrapper_shape,
        )
        r.shards = local_shards
        r.storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}

        # TODO: we shall continually extend this function to support more ops if needed
        if func in supported_ops:
            res_shards_list = [
                func(shard, *args[1:], **kwargs)
                # pyrefly: ignore [bad-index]
                for shard in args[0].shards
            ]
            # pyrefly: ignore [bad-index]
            return LocalShardsWrapper(res_shards_list, args[0].shard_offsets)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @property
    def shards(self) -> list[torch.Tensor]:
        return self.local_shards

    @shards.setter
    def shards(self, local_shards: list[torch.Tensor]):
        self.local_shards = local_shards

    @cached_property
    def shard_sizes(self) -> list[torch.Size]:
        return [chunk.sizes for chunk in self.storage_meta.chunks]

    @cached_property
    def shard_offsets(self) -> list[torch.Size]:
        return [chunk.offsets for chunk in self.storage_meta.chunks]


def run_torchrec_row_wise_even_sharding_example(rank, world_size):
    # row-wise even sharding example:
    #   One table is evenly sharded by rows within the global ProcessGroup.
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
    # tensor shape
    local_shard_shape = torch.Size(
        [num_embeddings // world_size, embedding_dim]  # (local_rows, local_cols)
    )
    # tensor offset
    local_shard_offset = torch.Size((rank * 2, embedding_dim))
    # tensor
    local_tensor = torch.randn(local_shard_shape, device=device)
    # row-wise sharding: one shard per rank
    # create the local shards wrapper
    # pyrefly: ignore [no-matching-overload]
    local_shards_wrapper = LocalShardsWrapper(
        local_shards=[local_tensor],
        offsets=[local_shard_offset],
    )

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
    row_wise_sharding_placements: list[Placement] = [Shard(0)]

    # create a DTensor from the local shard
    dtensor = DTensor.from_local(
        local_shards_wrapper, device_mesh, row_wise_sharding_placements, run_check=False
    )

    # display the DTensor's sharding
    visualize_sharding(dtensor, header="Row-wise even sharding example in DTensor")

    ###########################################################################
    # example 2: transform DTensor into local_shards
    # usage in TorchRec:
    #   In ShardedEmbeddingCollection's load_state_dict pre hook
    #   _pre_load_state_dict_hook, if the source param is a ShardedTensor
    #   then we need to transform it into its local_shards.

    # transform DTensor into LocalShardsWrapper
    dtensor_local_shards = dtensor.to_local()
    if not isinstance(dtensor_local_shards, LocalShardsWrapper):
        raise AssertionError
    shard_tensor = dtensor_local_shards.shards[0]
    if not torch.equal(shard_tensor, local_tensor):
        raise AssertionError
    if dtensor_local_shards.shard_sizes[0] != local_shard_shape:  # unwrap shape
        raise AssertionError
    if dtensor_local_shards.shard_offsets[0] != local_shard_offset:  # unwrap offset
        raise AssertionError


def run_torchrec_row_wise_uneven_sharding_example(rank, world_size):
    # row-wise uneven sharding example:
    #   One table is unevenly sharded by rows within the global ProcessGroup.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, and each rank will have the local shard
    #   of shape:
    #       rank 0: [1, 16]
    #       rank 1: [3, 16]
    #       rank 2: [1, 16]
    #       rank 3: [3, 16]

    # device mesh is a representation of the worker ranks
    # create a 1-D device mesh that includes every rank
    device_type = get_device_type()
    device = torch.device(device_type)
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])
    # tensor shape
    local_shard_shape = (
        torch.Size([1, embedding_dim])
        if rank % 2 == 0
        else torch.Size([3, embedding_dim])
    )
    # tensor offset
    local_shard_offset = torch.Size((rank // 2 * 4 + rank % 2 * 1, embedding_dim))
    # tensor
    local_tensor = torch.randn(local_shard_shape, device=device)
    # local shards
    # row-wise sharding: one shard per rank
    # create the local shards wrapper
    # pyrefly: ignore [no-matching-overload]
    local_shards_wrapper = LocalShardsWrapper(
        local_shards=[local_tensor],
        offsets=[local_shard_offset],
    )

    ###########################################################################
    # example 1: transform local_shards into DTensor
    # create the DTensorMetadata which torchrec should provide
    row_wise_sharding_placements: list[Placement] = [Shard(0)]

    # note: for uneven sharding, we need to specify the shape and stride because
    # DTensor would assume even sharding and compute shape/stride based on the
    # assumption. Torchrec needs to pass in this information explicitly.
    # shape/stride are global tensor's shape and stride
    dtensor = DTensor.from_local(
        local_shards_wrapper,  # a torch.Tensor subclass
        device_mesh,  # DeviceMesh
        row_wise_sharding_placements,  # List[Placement]
        run_check=False,
        shape=emb_table_shape,  # this is required for uneven sharding
        stride=(embedding_dim, 1),
    )
    # so far visualize_sharding() cannot print correctly for unevenly sharded DTensor
    # because it relies on offset computation which assumes even sharding.
    visualize_sharding(dtensor, header="Row-wise uneven sharding example in DTensor")
    # check the dtensor has the correct shape and stride on all ranks
    if dtensor.shape != emb_table_shape:
        raise AssertionError
    if dtensor.stride() != (embedding_dim, 1):
        raise AssertionError

    ###########################################################################
    # example 2: transform DTensor into local_shards
    # note: DTensor.to_local() always returns a LocalShardsWrapper
    dtensor_local_shards = dtensor.to_local()
    if not isinstance(dtensor_local_shards, LocalShardsWrapper):
        raise AssertionError
    shard_tensor = dtensor_local_shards.shards[0]
    if not torch.equal(shard_tensor, local_tensor):
        raise AssertionError
    if dtensor_local_shards.shard_sizes[0] != local_shard_shape:  # unwrap shape
        raise AssertionError
    if dtensor_local_shards.shard_offsets[0] != local_shard_offset:  # unwrap offset
        raise AssertionError


def run_torchrec_table_wise_sharding_example(rank, world_size):
    # table-wise example:
    #   each rank in the global ProcessGroup holds one different table.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, so each rank will have one 8 by 16 complete
    #   table as its local shard.

    device_type = get_device_type()
    device = torch.device(device_type)
    # note: without initializing this mesh, the following local_tensor will be put on
    # device cuda:0.
    init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])

    # for table i, if the current rank holds the table, then the local shard is
    # a LocalShardsWrapper containing the tensor; otherwise the local shard is
    # an empty torch.Tensor
    table_to_shards = {}  # map {table_id: local shard of table_id}
    table_to_local_tensor = {}  # map {table_id: local tensor of table_id}
    # create 4 embedding tables and place them on different ranks
    # each rank will hold one complete table, and the dict will store
    # the corresponding local shard.
    for i in range(world_size):
        # tensor
        local_tensor = (
            torch.randn(*emb_table_shape, device=device)
            if rank == i
            else torch.empty(0, device=device)
        )
        table_to_local_tensor[i] = local_tensor
        # tensor offset
        local_shard_offset = torch.Size((0, 0))
        # wrap local shards into a wrapper
        local_shards_wrapper = (
            # pyrefly: ignore [no-matching-overload]
            LocalShardsWrapper(
                local_shards=[local_tensor],
                offsets=[local_shard_offset],
            )
            if rank == i
            else local_tensor
        )
        table_to_shards[i] = local_shards_wrapper

    ###########################################################################
    # example 1: transform local_shards into DTensor
    table_to_dtensor = {}  # same purpose as _model_parallel_name_to_sharded_tensor
    table_wise_sharding_placements = [Replicate()]  # table-wise sharding

    for table_id, local_shards in table_to_shards.items():
        # create a submesh that only contains the rank we place the table
        # note that we cannot use ``init_device_mesh'' to create a submesh
        # so we choose to use the `DeviceMesh` api to directly create a DeviceMesh
        device_submesh = DeviceMesh(
            device_type=device_type,
            mesh=torch.tensor(
                [table_id], dtype=torch.int64
            ),  # table ``table_id`` is placed on rank ``table_id``
        )
        # create a DTensor from the local shard for the current table
        # note: for uneven sharding, we need to specify the shape and stride because
        # DTensor would assume even sharding and compute shape/stride based on the
        # assumption. Torchrec needs to pass in this information explicitly.
        dtensor = DTensor.from_local(
            local_shards,
            device_submesh,
            table_wise_sharding_placements,
            run_check=False,
            shape=emb_table_shape,  # this is required for uneven sharding
            stride=(embedding_dim, 1),
        )
        table_to_dtensor[table_id] = dtensor

    # print each table's sharding
    for table_id, dtensor in table_to_dtensor.items():
        visualize_sharding(
            dtensor,
            header=f"Table-wise sharding example in DTensor for Table {table_id}",
        )
        # check the dtensor has the correct shape and stride on all ranks
        if dtensor.shape != emb_table_shape:
            raise AssertionError
        if dtensor.stride() != (embedding_dim, 1):
            raise AssertionError

    ###########################################################################
    # example 2: transform DTensor into torch.Tensor
    for table_id, local_tensor in table_to_local_tensor.items():
        # important: note that DTensor.to_local() always returns an empty torch.Tensor
        # no matter what was passed to DTensor._local_tensor.
        dtensor_local_shards = table_to_dtensor[table_id].to_local()
        if rank == table_id:
            if not isinstance(dtensor_local_shards, LocalShardsWrapper):
                raise AssertionError
            shard_tensor = dtensor_local_shards.shards[0]
            if not torch.equal(shard_tensor, local_tensor):  # unwrap tensor
                raise AssertionError
            if dtensor_local_shards.shard_sizes[0] != emb_table_shape:  # unwrap shape
                raise AssertionError
            if dtensor_local_shards.shard_offsets[0] != torch.Size(
                (0, 0)
            ):  # unwrap offset
                raise AssertionError
        else:
            if dtensor_local_shards.numel() != 0:
                raise AssertionError


def run_example(rank, world_size, example_name):
    # the dict that stores example code
    name_to_example_code = {
        "row-wise-even": run_torchrec_row_wise_even_sharding_example,
        "row-wise-uneven": run_torchrec_row_wise_uneven_sharding_example,
        "table-wise": run_torchrec_table_wise_sharding_example,
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
    if world_size != 4:  # our example uses 4 worker ranks
        raise AssertionError
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="torchrec sharding examples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    example_prompt = (
        "choose one sharding example from below:\n"
        "\t1. row-wise-even;\n"
        "\t2. row-wise-uneven\n"
        "\t3. table-wise\n"
        "e.g. you want to try the row-wise even sharding example, please input 'row-wise-even'\n"
    )
    parser.add_argument("-e", "--example", help=example_prompt, required=True)
    args = parser.parse_args()
    run_example(rank, world_size, args.example)
