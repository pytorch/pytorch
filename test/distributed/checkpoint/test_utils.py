# Owner(s): ["oncall: distributed"]

import io
import sys
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.c10d_logger import _c10d_logger
from torch.distributed.checkpoint.logger import _dcp_logger
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.utils import (
    _create_file_view,
    _DistWrapper,
    find_state_dict_object,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.distributed_utils import with_fake_comms


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
        shard_md = ShardMetadata(
            shard_offsets=[idx * 8], shard_sizes=[8], placement=f"rank:{shard_rank}/cpu"
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
        size=torch.Size([8 * len(shards_metadata)]),
        tensor_properties=TensorProperties.create_from_tensor(torch.zeros(1)),
    )

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


class TestMedatadaIndex(TestCase):
    def test_init_convert_offset(self):
        a = MetadataIndex("foo", [1, 2])
        b = MetadataIndex("foo", torch.Size([1, 2]))
        self.assertEqual(a, b)

    def test_index_hint_ignored_on_equals(self):
        a = MetadataIndex("foo")
        b = MetadataIndex("foo", index=99)
        self.assertEqual(a, b)

    def test_index_hint_ignored_on_hash(self):
        a = MetadataIndex("foo")
        b = MetadataIndex("foo", index=99)
        self.assertEqual(hash(a), hash(b))

    def test_flat_data(self):
        state_dict = {
            "a": torch.rand(10),
            "b": [1, 2, 3],
        }

        a = find_state_dict_object(state_dict, MetadataIndex("a"))
        self.assertEqual(a, state_dict["a"])
        a = find_state_dict_object(state_dict, MetadataIndex("a", [0]))
        self.assertEqual(a, state_dict["a"])
        a = find_state_dict_object(state_dict, MetadataIndex("a", index=99))
        self.assertEqual(a, state_dict["a"])

        b = find_state_dict_object(state_dict, MetadataIndex("b"))
        self.assertEqual(b, state_dict["b"])
        b = find_state_dict_object(state_dict, MetadataIndex("b", index=1))
        self.assertEqual(b, state_dict["b"])

        with self.assertRaisesRegex(ValueError, "FQN"):
            find_state_dict_object(state_dict, MetadataIndex("c"))
        with self.assertRaisesRegex(ValueError, "ShardedTensor"):
            find_state_dict_object(state_dict, MetadataIndex("b", [1]))

    @with_fake_comms(rank=0, world_size=2)
    def test_sharded_tensor_lookup(self):
        st = create_sharded_tensor(rank=0, world_size=2, shards_per_rank=3)
        state_dict = {"st": st}

        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8]))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # good hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=1))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # bad hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=2))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # broken hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=99))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        with self.assertRaisesRegex(ValueError, "no offset was provided"):
            find_state_dict_object(state_dict, MetadataIndex("st"))

        with self.assertRaisesRegex(ValueError, "Could not find shard"):
            find_state_dict_object(state_dict, MetadataIndex("st", [1]))

    def test_dcp_logger(self):
        self.assertTrue(_c10d_logger is not _dcp_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))


class TestReaderView(TestCase):
    def setUp(self):
        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.front_view = _create_file_view(buffer, 0, 5)

        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.middle_view = _create_file_view(buffer, 10, 5)

        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.back_view = _create_file_view(buffer, len(buffer.getbuffer()) - 5, 5)

    def testShortRead(self):
        self.assertEqual(self.front_view.read(3), b"ABC")
        self.assertEqual(self.middle_view.read(3), b"KLM")
        self.assertEqual(self.back_view.read(3), b"VWX")

    def testLongRead(self):
        self.assertEqual(self.front_view.read(10), b"ABCDE")
        self.assertEqual(self.middle_view.read(10), b"KLMNO")
        self.assertEqual(self.back_view.read(10), b"VWXYZ")

    def testAllRead(self):
        self.assertEqual(self.front_view.read(-1), b"ABCDE")
        self.assertEqual(self.middle_view.read(-1), b"KLMNO")
        self.assertEqual(self.back_view.read(-1), b"VWXYZ")

    def testShortReadinto(self):
        ba = bytearray(3)

        self.assertEqual(self.front_view.readinto(ba), 3)
        self.assertEqual(ba, b"ABC")

        self.assertEqual(self.middle_view.readinto(ba), 3)
        self.assertEqual(ba, b"KLM")

        self.assertEqual(self.back_view.readinto(ba), 3)
        self.assertEqual(ba, b"VWX")

    def testLongReadinto(self):
        ba = bytearray(8)
        self.assertEqual(self.front_view.readinto(ba), 5)
        self.assertEqual(ba, b"ABCDE\0\0\0")
        self.assertEqual(self.front_view.readinto(ba), 0)
        self.assertEqual(ba, b"ABCDE\0\0\0")

        self.assertEqual(self.middle_view.readinto(ba), 5)
        self.assertEqual(ba, b"KLMNO\0\0\0")
        self.assertEqual(self.middle_view.readinto(ba), 0)
        self.assertEqual(ba, b"KLMNO\0\0\0")

        self.assertEqual(self.back_view.readinto(ba), 5)
        self.assertEqual(ba, b"VWXYZ\0\0\0")
        self.assertEqual(self.back_view.readinto(ba), 0)
        self.assertEqual(ba, b"VWXYZ\0\0\0")


class TestDistWrapper(DTensorTestBase):
    @property
    def world_size(self):
        return min(4, torch.accelerator.device_count())

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_object(self):
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        rank = mesh_2d.get_rank()
        half_world_size = self.world_size // 2
        gathered_objects = dist_wrapper.gather_object(rank)
        expected_objects = (
            list(range(rank, rank + half_world_size))
            if rank % half_world_size == 0
            else None
        )
        assert gathered_objects == expected_objects

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_scatter_object(self):
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        rank = mesh_2d.get_rank()
        half_world_size = self.world_size // 2

        objects = (
            list(range(rank, rank + half_world_size))
            if rank % half_world_size == 0
            else None
        )
        scattered_objects = dist_wrapper.scatter_object(objects)
        expected_objects = rank
        assert scattered_objects == expected_objects

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_broadcast_object_with_nonzero_coordinator(self):
        # Everybody uses WORLD, but src is coordinator_rank=1
        dist_wrapper = _DistWrapper(
            group=dist.group.WORLD,
            use_dist=True,
            coordinator_rank=1,
        )

        rank = dist.get_rank()
        # only local rank 1 supplies the payload
        payload: Optional[int] = rank if rank == 1 else None

        result = dist_wrapper.broadcast_object(payload)
        # every rank should receive the value from global rank 1
        assert result == 1

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_broadcast_object_global_local_mismatch(self):
        # reproduces issue 152310

        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        dist_wrapper = _DistWrapper(
            group=mesh_2d.get_group(1),
            use_dist=True,
            coordinator_rank=1,  # local coordinator index within the subgroup
        )

        rank = mesh_2d.get_rank()

        # only the local coordinator in each subgroup provides payload
        payload: Optional[int] = rank if dist_wrapper.is_coordinator else None
        got = dist_wrapper.broadcast_object(payload)

        # ensure we broadcast from the *global* coordinator rank,
        # not the local index.  For rows [0,1] this is global rank 1;
        # for rows [2,3] this is global rank 3.
        expected = dist_wrapper.global_coordinator_rank
        assert got == expected

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_barrier(self):
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        # No exception should be raised.
        dist_wrapper.barrier()


if __name__ == "__main__":
    run_tests()
