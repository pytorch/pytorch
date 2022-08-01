# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._shard.checkpoint.planner import WriteItemType

from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardMetadata,
    ShardedTensor,
    ShardedTensorMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

from torch.testing._internal.common_utils import (
    TestCase,
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.distributed._shard.checkpoint.metadata import MetadataIndex
from torch.testing._internal.distributed.distributed_utils import (
    with_fake_comms,
    with_dist
)

from torch.distributed._shard.checkpoint.resharding import (
    create_default_global_save_plan,
    create_default_local_save_plan,
    create_default_local_load_plan,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

def create_sharded_tensor(rank, world_size, shards_per_rank):
    shards_metadata = []
    local_shards = []
    for idx in range(0, world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        shard_md = ShardMetadata(shard_offsets=[idx * 8], shard_sizes=[8], placement=f"rank:{shard_rank}/cpu")
        shards_metadata.append(shard_md)
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(
                torch.rand(*shard_md.shard_sizes),
                shard_offsets=shard_md.shard_offsets,
                rank=rank
            )
            local_shards.append(shard)

    sharded_tensor_md = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size([8 * len(shards_metadata)]),
        tensor_properties=TensorProperties.create_from_tensor(torch.zeros(1))
    )

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards,
        sharded_tensor_metadata=sharded_tensor_md
    )


class TestSavePlan(TestCase):
    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan(self):
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {
            "tensor": tensor,
            "value": val,
            "st": st
        }
        plan = create_default_local_save_plan(state_dict, False)
        self.assertEqual(1, len(plan.items))
        wi = plan.items[0]
        self.assertEqual(wi.index, MetadataIndex("st", [8]))
        self.assertEqual(wi.type, WriteItemType.SHARD)
        self.assertEqual(wi.tensor_data.size, st.size())
        self.assertEqual(wi.tensor_data.properties, TensorProperties.create_from_tensor(torch.zeros(1)))
        self.assertEqual(wi.tensor_data.chunk.offsets, torch.Size([8]))
        self.assertEqual(wi.tensor_data.chunk.sizes, torch.Size([8]))
        self.assertEqual(wi.tensor_data.chunk.size_in_bytes, -1)

        # Coordinator rank, should include replicated items as well
        plan = create_default_local_save_plan(state_dict, True)
        self.assertEqual(3, len(plan.items))

        tensor_wi = next(wi for wi in plan.items if wi.type == WriteItemType.TENSOR)
        self.assertEqual(tensor_wi.index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_wi.tensor_data.size, tensor.size())
        self.assertEqual(tensor_wi.tensor_data.properties, TensorProperties.create_from_tensor(tensor))
        self.assertEqual(tensor_wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(tensor_wi.tensor_data.chunk.sizes, torch.Size([10]))
        self.assertEqual(tensor_wi.tensor_data.chunk.size_in_bytes, -1)

        bytes_wi = next(wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO)
        self.assertEqual(bytes_wi.index, MetadataIndex("value"))
        self.assertIsNone(bytes_wi.tensor_data)

    def test_global_plan(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {
                    "tensor": tensor,
                    "value": val,
                    "st": st
                }
                return create_default_local_save_plan(state_dict, rank == 0)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        final_plans, metadata = create_default_global_save_plan(all_plans=all_plans)


"""
I want tests for the following functionality:

create_default_global_save_plan
create_default_local_save_plan
create_default_local_load_plan

No need to test the default planners TBH.

The main complication I want in testing is resharding
"""



if __name__ == "__main__":
    run_tests()
