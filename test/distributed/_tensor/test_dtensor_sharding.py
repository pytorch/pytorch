# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
)
from torch.distributed._tensor.placement_types import _Partial, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorShardingTest(DTensorTestBase):
    @with_comms
    def test_dtensor_row_wise_sharding(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements = [Shard(0)]  # row-wise sharding
        local_shard_shape = [4, 4]
        global_tensor_shape = [4 * self.world_size, 4]
        global_tensor = torch.randn(*global_tensor_shape)

        # example 1: embedding sharding
        dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_shard = dtensor.to_local()
        self.assertEqual(local_shard.shape, torch.Size(local_shard_shape))

        # example 2: load state dict from a global Tensor
        # pre hook: let distribute_tensor do the splicing
        src_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        # _load_state_dict: copy DTensor into DTensor
        dtensor.copy_(src_dtensor)

        # example 3: state dict
        # post hook: always return DTensor
        # torchrec ShardedEmbeddingCollection keeps a list _model_parallel_name_to_sharded_tensor
        # which is initialized in _initialize_torch_state where torch.Tensor params are transformed
        # into ShardedTensor by ShardedTensor._init_from_local_shards()
        dtensor = DTensor.from_local(local_shard, device_mesh, placements, run_check=False)
        self.assertEqual(dtensor.full_tensor(), global_tensor)

    @with_comms
    def test_dtensor_table_wise_sharding(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size))) # do we need this?
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
            device_mesh = DeviceMesh(self.device_type, [table_id])  # table i is placed on rank i
            dtensor = DTensor.from_local(local_shard, device_mesh, placements, run_check=False)
            table_to_dtensor[table_id] = dtensor

        # example 2: transform DTensor into torch.Tensor
        for table_id, local_shard in table_to_local_shard.items():
            dtensor_local_shard = table_to_dtensor[table_id].to_local()
            self.assertEqual(dtensor_local_shard, local_shard)


if __name__ == "__main__":
    run_tests()
