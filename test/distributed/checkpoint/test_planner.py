# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as TensorProperties_Shard,
)
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import (
    _create_default_local_metadata,
    _validate_global_plan,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_local_save_plan,
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    SavePlan,
    SavePlanner,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _compare_save_plans,
    _merge_delta_local_plans,
    create_read_items_for_chunk_list,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.distributed_utils import (
    with_dist,
    with_fake_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def create_sharded_tensor(rank, world_size, shards_per_rank, shard_size=8):
    shards_metadata = []
    local_shards = []
    for idx in range(world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        shard_md = ShardMetadata(
            shard_offsets=[idx * shard_size],
            shard_sizes=[shard_size],
            placement=f"rank:{shard_rank}/cpu",
        )
        shards_metadata.append(shard_md)
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(
                torch.rand(*shard_md.shard_sizes),
                shard_offsets=shard_md.shard_offsets,
                rank=rank,
            )
            local_shards.append(shard)

    sharded_tensor_md = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size([shard_size * len(shards_metadata)]),
        tensor_properties=TensorProperties_Shard.create_from_tensor(torch.zeros(1)),
    )

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


class TestSavePlan(TestCase):
    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan(self):
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {"tensor": tensor, "value": val, "st": st}
        plan = create_default_local_save_plan(state_dict, False)
        self.assertEqual(3, len(plan.items))
        wi = plan.items[0]
        self.assertEqual(wi.index, MetadataIndex("tensor", [0]))
        self.assertEqual(wi.type, WriteItemType.TENSOR)
        self.assertEqual(wi.tensor_data.size, tensor.size())
        self.assertEqual(
            wi.tensor_data.properties,
            TensorProperties.create_from_tensor(torch.zeros(1)),
        )
        self.assertEqual(wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(wi.tensor_data.chunk.sizes, torch.Size([10]))

        st_wi = plan.items[2]
        self.assertEqual(st_wi.index, MetadataIndex("st", [8]))
        self.assertEqual(st_wi.type, WriteItemType.SHARD)
        self.assertEqual(st_wi.tensor_data.size, st.size())
        self.assertEqual(
            st_wi.tensor_data.properties,
            TensorProperties.create_from_tensor(torch.zeros(1)),
        )
        self.assertEqual(st_wi.tensor_data.chunk.offsets, torch.Size([8]))
        self.assertEqual(st_wi.tensor_data.chunk.sizes, torch.Size([8]))

        # Coordinator rank, should include replicated items as well
        plan = create_default_local_save_plan(state_dict, True)
        self.assertEqual(3, len(plan.items))

        tensor_wi = next(wi for wi in plan.items if wi.type == WriteItemType.TENSOR)
        self.assertEqual(tensor_wi.index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_wi.tensor_data.size, tensor.size())
        self.assertEqual(
            tensor_wi.tensor_data.properties,
            TensorProperties.create_from_tensor(tensor),
        )
        self.assertEqual(tensor_wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(tensor_wi.tensor_data.chunk.sizes, torch.Size([10]))

        bytes_wi = next(wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO)
        self.assertEqual(bytes_wi.index, MetadataIndex("value"))
        self.assertIsNone(bytes_wi.tensor_data)

    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan_with_caching(self):
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {"tensor": tensor, "value": val, "st": st}
        planner = DefaultSavePlanner(enable_plan_caching=True)
        planner.set_up_planner(state_dict, is_coordinator=False)
        # First iteration, should create a new plan
        first_plan = planner.create_local_plan()

        # Validate that the plan has been cached
        cached_plan = SavePlanner._cached_save_plan[planner._cached_plans_key]
        self.assertEqual(first_plan, cached_plan)

        # second iteration, should create an empty unusable plan
        second_plan = planner.create_local_plan()
        self.assertFalse(second_plan.usable)
        self.assertEqual(0, len(second_plan.items))
        self.assertIsNone(second_plan.planner_data)
        self.assertIsNone(second_plan.storage_data)

    def test_global_plan(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        all_plans = dedup_save_plans(all_plans)
        final_plans, metadata = create_default_global_save_plan(all_plans=all_plans)

        # The default global plan updates all indexes to include hints
        for new_plan, old_plan in zip(final_plans, all_plans):
            for new_item, old_item in zip(new_plan.items, old_plan.items):
                self.assertEqual(new_item.index, old_item.index)
                self.assertEqual(new_item.type, old_item.type)
                self.assertEqual(new_item.tensor_data, old_item.tensor_data)
                self.assertIn(new_item.index.fqn, metadata.state_dict_metadata)

                item_md = metadata.state_dict_metadata[new_item.index.fqn]
                if new_item.type == WriteItemType.BYTE_IO:
                    self.assertTrue(isinstance(item_md, BytesStorageMetadata))
                else:
                    self.assertTrue(isinstance(item_md, TensorStorageMetadata))
                    self.assertEqual(item_md.size, old_item.tensor_data.size)
                    self.assertEqual(
                        item_md.properties, old_item.tensor_data.properties
                    )

                    self.assertIsNotNone(new_item.index.index)
                    # Make sure the hint is correct
                    self.assertEqual(
                        item_md.chunks[new_item.index.index], old_item.tensor_data.chunk
                    )

    def test_dedup_plans(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        deduped_plans = dedup_save_plans(all_plans)

        # Number of plans should remain unchanged
        self.assertEqual(len(all_plans), len(deduped_plans))

        # Number of items in the deduped plans should be less than the original plans
        for new_plan, old_plan in zip(deduped_plans, all_plans):
            self.assertFalse(_compare_save_plans(new_plan, old_plan))
            self.assertTrue(len(new_plan.items) < len(old_plan.items))

    def test_global_plan_with_caching(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                planner = DefaultSavePlanner(enable_plan_caching=True)
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
                return planner.create_local_plan()

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        planner = DefaultSavePlanner(enable_plan_caching=True)
        # First iteration, should create a new plan
        first_global_plan, first_metadata = planner.create_global_plan(all_plans)

        # Validate that the plan has been cached
        cached_global_plan = SavePlanner._cached_global_plan[planner._cached_plans_key]
        self.assertEqual(cached_global_plan, first_global_plan)

        # Validate that all_plans are cached
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        # Second iteration, should return empty plans
        # Recreate the plans as the previous ones are deduped.
        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        second_global_plan, second_metadata = planner.create_global_plan(all_plans)
        # All the plans should be empty and usable
        for plan in second_global_plan:
            self.assertFalse(plan.usable)
            self.assertEqual(0, len(plan.items))
            self.assertIsNone(plan.planner_data)
            self.assertIsNone(plan.storage_data)

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(
            second_metadata, planner._cached_metadata[planner._cached_plans_key]
        )

        # Validate that all_plans are cached and remain unchanged.
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        # Third iteration with changed plans
        def create_data_v2(rank):
            with with_dist(rank=rank, world_size=4):
                planner = DefaultSavePlanner(enable_plan_caching=True)
                tensor = torch.rand(20)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
                return planner.create_local_plan()

        all_plans = [
            create_data_v2(0),
            create_data_v2(1),
            create_data_v2(2),
            create_data_v2(3),
        ]
        third_global_plan, third_metadata = planner.create_global_plan(all_plans)
        # Only the rank 0 plan should be non-empty. The rest should be empty
        tensor_plan = third_global_plan[0]
        self.assertNotEqual(0, len(tensor_plan.items))
        self.assertTrue(tensor_plan.usable)

        # Validate that all_plans are updated and cached
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        for plan in third_global_plan[1:]:
            self.assertFalse(plan.usable)
            self.assertEqual(0, len(plan.items))
            self.assertIsNone(plan.planner_data)
            self.assertIsNone(plan.storage_data)

        # Global metadata should be different as one plan has changed
        self.assertNotEqual(second_metadata, third_metadata)
        # Validate that the metadata is cached
        self.assertEqual(
            third_metadata, planner._cached_metadata[planner._cached_plans_key]
        )

        # Validate that the new plan has been cached
        cached_global_plan = SavePlanner._cached_global_plan[planner._cached_plans_key][
            0
        ]
        self.assertEqual(cached_global_plan, tensor_plan)

    def test_finish_plan_with_caching(self):
        planner = DefaultSavePlanner(enable_plan_caching=True)
        tensor = torch.rand(10)
        val = [1, 2, 3]
        state_dict = {"tensor": tensor, "value": val}
        planner.set_up_planner(state_dict, is_coordinator=True)
        plan = planner.create_local_plan()

        # First iteration, should create a new plan
        first_finished_plan = planner.finish_plan(plan)

        # Validate that the plan has been cached
        cached_finished_plan = SavePlanner._cached_final_save_plan[
            planner._cached_plans_key
        ]
        self.assertEqual(first_finished_plan, cached_finished_plan)

        # second iteration, should return the cached plan
        second_finished_plan = planner.finish_plan(SavePlan([], usable=False))
        self.assertEqual(second_finished_plan, first_finished_plan)

    def test_local_load_plan(self):
        def create_state_dict(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                return {"tensor": tensor, "value": val, "st": st}

        state_dict = create_state_dict(1)
        metadata = _create_default_local_metadata(state_dict)

        load_plan = create_default_local_load_plan(state_dict, metadata)
        # This will create 3 entries
        self.assertEqual(3, len(load_plan.items))
        st_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "st")
        tensor_item = next(
            ri for ri in load_plan.items if ri.dest_index.fqn == "tensor"
        )
        bytes_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "value")

        self.assertEqual(st_item.type, LoadItemType.TENSOR)
        # This is an exact copy
        self.assertEqual(st_item.dest_index, MetadataIndex("st", [8]))
        self.assertEqual(st_item.dest_offsets, torch.Size([0]))
        self.assertEqual(st_item.storage_index, MetadataIndex("st", [8]))
        self.assertEqual(st_item.storage_offsets, torch.Size([0]))
        self.assertEqual(st_item.lengths, torch.Size([8]))

        self.assertEqual(tensor_item.type, LoadItemType.TENSOR)
        self.assertEqual(tensor_item.dest_index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_item.dest_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.storage_index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_item.storage_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.lengths, torch.Size([10]))

        self.assertEqual(bytes_item.type, LoadItemType.BYTE_IO)
        self.assertEqual(bytes_item.dest_index, MetadataIndex("value"))

    def test_load_with_resharding(self):
        def create_state_dict(rank, world_size):
            with with_dist(rank=rank, world_size=world_size):
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=128 // world_size,
                    )
                }

        # Rank 1 has a 16 bytes shard from [16, 32[
        world8_state_dict = create_state_dict(rank=1, world_size=8)
        world8_metadata = _create_default_local_metadata(world8_state_dict)

        # Rank 1 has a 32 bytes shard from [32, 64[
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # First scenario, going from world=8 to world=4, need to load 2 shards
        # Each 4-world shard has 32 elements, so it needs to load 2 shards
        load_plan = create_default_local_load_plan(world4_state_dict, world8_metadata)
        self.assertEqual(2, len(load_plan.items))
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([16])
        )

        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([16]))

        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [48]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([16]))
        self.assertEqual(high_ri.lengths, torch.Size([16]))

        # Second scenario, going from world=4 to world=8, need to load half of 1 shard
        # rank1 on 8-world needs to load the upper half of the rank0 4-world shard
        load_plan = create_default_local_load_plan(world8_state_dict, world4_metadata)
        self.assertEqual(1, len(load_plan.items))
        ri = load_plan.items[0]
        self.assertEqual(ri.storage_index, MetadataIndex("st", [0]))
        self.assertEqual(ri.storage_offsets, torch.Size([16]))
        self.assertEqual(ri.dest_index, MetadataIndex("st", [16]))
        self.assertEqual(ri.dest_offsets, torch.Size([0]))
        self.assertEqual(ri.lengths, torch.Size([16]))

    def test_load_with_world_size_diff_by_one(self):
        def create_state_dict(rank, world_size):
            with with_dist(rank=rank, world_size=world_size):
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=120 // world_size,
                    )
                }

        # rank 1 has a 30 bytes shard from [30, 60[
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # rank 1 has a 40 bytes shard from [40, 80[
        world3_state_dict = create_state_dict(rank=1, world_size=3)

        load_plan = create_default_local_load_plan(world3_state_dict, world4_metadata)
        self.assertEqual(2, len(load_plan.items))
        # this is [30, 60] to load [40, 60]
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        # this is [60, 90] to load [60, 80]
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([20])
        )

        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [30]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([10]))
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [40]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([20]))

        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [60]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [40]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([20]))
        self.assertEqual(high_ri.lengths, torch.Size([20]))


class TestPlannerHelpers(TestCase):
    def test_create_read_item_from_chunks(self):
        tensor_md = TensorStorageMetadata(
            properties=TensorProperties.create_from_tensor(torch.empty([16])),
            size=torch.Size([16]),
            chunks=[
                ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([8])),
                ChunkStorageMetadata(offsets=torch.Size([8]), sizes=torch.Size([8])),
            ],
        )

        chunk = ChunkStorageMetadata(offsets=torch.Size([4]), sizes=torch.Size([7]))
        read_items = create_read_items_for_chunk_list("foo", tensor_md, [chunk])

        self.assertEqual(2, len(read_items))
        self.assertEqual(MetadataIndex("foo", [4]), read_items[0].dest_index)
        self.assertEqual(torch.Size([0]), read_items[0].dest_offsets)

        self.assertEqual(MetadataIndex("foo", [0]), read_items[0].storage_index)
        self.assertEqual(torch.Size([4]), read_items[0].storage_offsets)

        self.assertEqual(torch.Size([4]), read_items[0].lengths)

        self.assertEqual(MetadataIndex("foo", [4]), read_items[1].dest_index)
        self.assertEqual(torch.Size([4]), read_items[1].dest_offsets)

        self.assertEqual(MetadataIndex("foo", [8]), read_items[1].storage_index)
        self.assertEqual(torch.Size([0]), read_items[1].storage_offsets)

        self.assertEqual(torch.Size([3]), read_items[1].lengths)

    def test_merge_delta_local_plans(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        def _validate_plans(plan1: SavePlan, plan2: SavePlan):
            self.assertEqual(len(plan1.items), len(plan2.items))
            for item1, item2 in zip(plan1.items, plan2.items):
                self.assertEqual(item1.index, item2.index)
                self.assertEqual(item1.type, item2.type)
                self.assertEqual(item1.tensor_data, item2.tensor_data)

        cached_plans = [create_data(0), create_data(1)]
        delta_plans = [create_data(2), create_data(3)]

        # Both the plans changed.
        # Merge plan should have both the plans from the delta plans
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        self.assertEqual(2, len(merged_plans))
        _validate_plans(delta_plans[0], merged_plans[0])
        _validate_plans(delta_plans[1], merged_plans[1])

        # Only the first plan changed.
        # Merge plan should have the first plan from the delta plans and the second plan from the cached plans
        delta_plans = [create_data(2), SavePlan([], usable=False)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(delta_plans[0], merged_plans[0])
        _validate_plans(cached_plans[1], merged_plans[1])

        # Only the second plan changed.
        # Merge plan should have the first plan from the cached plans and the second plan from the delta plans
        delta_plans = [SavePlan([], usable=False), create_data(3)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(cached_plans[0], merged_plans[0])
        _validate_plans(delta_plans[1], merged_plans[1])

        # None of the plans changed. Cached plans should be returned
        delta_plans = [SavePlan([], usable=False), SavePlan([], usable=False)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(cached_plans[0], merged_plans[0])
        _validate_plans(cached_plans[1], merged_plans[1])

    def test_compare_save_plans(self):
        def create_data(rank):
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        plan1 = create_data(0)
        plan2 = create_data(1)
        self.assertFalse(_compare_save_plans(plan1, plan2))
        self.assertTrue(_compare_save_plans(plan1, plan1))
        self.assertTrue(_compare_save_plans(plan2, plan2))


class TestValidateGlobalPlan(TestCase):
    def _make_metadata(self, chunks, size):
        storage = TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=torch.Size(size),
            chunks=chunks,
        )
        return Metadata(state_dict_metadata={"param": storage})

    def test_non_overlapping_chunks(self):
        chunks = [
            ChunkStorageMetadata(offsets=torch.Size([i]), sizes=torch.Size([1]))
            for i in range(4)
        ]
        metadata = self._make_metadata(chunks, [4])
        self.assertTrue(_validate_global_plan([SavePlan([])], metadata))

    def test_detect_overlapping_chunks(self):
        chunks = [
            ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([2])),
            ChunkStorageMetadata(offsets=torch.Size([1]), sizes=torch.Size([2])),
        ]
        metadata = self._make_metadata(chunks, [4])
        self.assertFalse(_validate_global_plan([SavePlan([])], metadata))


class TestLoadPlanner(TestCase):
    @with_temp_dir
    def test_strict(self):
        original_module = nn.Linear(2, 2)
        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(2, 2)
        new_module.extra_param = nn.Parameter(torch.randn(2, 2))
        dcp.load(
            state_dict={"module": new_module},
            checkpoint_id=self.temp_dir,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        with self.assertRaisesRegex(CheckpointException, "Missing key in checkpoint"):
            dcp.load(
                state_dict={"module": new_module},
                checkpoint_id=self.temp_dir,
                planner=DefaultLoadPlanner(allow_partial_load=False),
            )

    @with_temp_dir
    def test_load_different_sizes_throws(self):
        original_module = nn.Linear(2, 2)
        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(3, 2)
        with self.assertRaisesRegex(CheckpointException, "Size mismatch"):
            dcp.load(
                state_dict={"module": new_module},
                checkpoint_id=self.temp_dir,
                planner=DefaultLoadPlanner(),
            )

    @with_temp_dir
    def test_version_key_in_planner_data(self):
        original_module = nn.Linear(2, 2)

        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(2, 2)
        planner = DefaultLoadPlanner()
        dcp.load(
            state_dict={"module": new_module},
            checkpoint_id=self.temp_dir,
            planner=planner,
        )

        self.assertEqual(planner.metadata.version, CURRENT_DCP_VERSION)


if __name__ == "__main__":
    run_tests()
