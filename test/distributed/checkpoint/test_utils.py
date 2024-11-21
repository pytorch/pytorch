# Owner(s): ["oncall: distributed"]

import sys

import torch
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
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
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


if __name__ == "__main__":
    run_tests()
