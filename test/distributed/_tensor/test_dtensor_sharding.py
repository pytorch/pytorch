# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorShardingTest(DTensorTestBase):
    @with_comms
    def test_dtensor_row_wise_sharding(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        row_wise_sharding_placements = [Shard(0)]  # row-wise sharding
        local_shard_shape = [4, 4]
        global_tensor_shape = [4 * self.world_size, 4]
        global_tensor = torch.randn(*global_tensor_shape)

        # example 1: embedding sharding
        dtensor = distribute_tensor(
            global_tensor, device_mesh, row_wise_sharding_placements
        )
        local_shard = dtensor.to_local()
        self.assertEqual(local_shard.shape, torch.Size(local_shard_shape))

        # example 2: transform DTensor into local_shards
        # usage in TorchRec:
        #   In ShardedEmbeddingCollection's load_state_dict pre hook
        #   _pre_load_state_dict_hook, the source param will be spliced
        #   according to local_shards' original place in the gloabl tensor
        #   if the source param is a torch.Tensor because the source param
        #   holds the global tensor and we want to tailor the shards from it.
        #   We can let ``distribute_tensor'' do the splicing work for us.
        src_dtensor = distribute_tensor(
            global_tensor, device_mesh, row_wise_sharding_placements
        )
        splice_result = row_wise_sharding_placements[0]._split_tensor(
            global_tensor, self.world_size, with_padding=False, contiguous=True
        )[0][self.rank]
        self.assertEqual(splice_result, src_dtensor.to_local())

        # example 3: copy DTensor into DTensor as in _load_state_dict()
        dtensor.copy_(src_dtensor)

        # example 4: transform local_shards into DTensor
        # usage in TorchRec:
        #   ShardedEmbeddingCollection stores model parallel params in
        #   _model_parallel_name_to_sharded_tensor which is initialized in
        #   _initialize_torch_state() and torch.Tensor params are transformed
        #   into ShardedTensor by ShardedTensor._init_from_local_shards().
        #
        #   This allows state_dict() to always return ShardedTensor objects.
        dtensor = DTensor.from_local(
            local_shard, device_mesh, row_wise_sharding_placements, run_check=False
        )
        self.assertEqual(dtensor.full_tensor(), global_tensor)

    @with_comms
    def test_dtensor_table_wise_sharding(self):
        # initialize N tables
        table_shape = [4, 4]
        table_to_local_shard = {}  # map {table_id: local shard of table_id}
        for i in range(self.world_size):
            local_shard = torch.randn(*table_shape)
            # table i is placed on rank i
            table_to_local_shard[i] = local_shard if self.rank == i else torch.empty(0)

        # example 1: transform local_shards into DTensor
        table_to_dtensor = {}  # same purpose as _model_parallel_name_to_sharded_tensor
        for table_id, local_shard in table_to_local_shard.items():
            placements = [Replicate()]  # table-wise sharding
            device_mesh = DeviceMesh(
                self.device_type, [table_id]  # table i is placed on rank i
            )
            dtensor = DTensor.from_local(
                local_shard, device_mesh, placements, run_check=False
            )
            table_to_dtensor[table_id] = dtensor

        # example 2: transform DTensor into torch.Tensor
        for table_id, local_shard in table_to_local_shard.items():
            dtensor_local_shard = table_to_dtensor[table_id].to_local()
            self.assertEqual(dtensor_local_shard, local_shard)


if __name__ == "__main__":
    run_tests()
