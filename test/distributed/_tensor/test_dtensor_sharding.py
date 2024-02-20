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
    def test_dtensor_rowwise_sharding(self):
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


if __name__ == "__main__":
    run_tests()
