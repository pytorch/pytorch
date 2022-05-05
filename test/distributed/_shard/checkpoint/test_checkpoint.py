# Owner(s): ["oncall: distributed"]

import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn
from torch.testing._internal.common_utils import TestCase

from torch.distributed._shard.checkpoint.resharding import (
    _prepare_sharded_tensor_write,
    _create_storage_key
)

from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.checkpoint.state_dict_loader import validate_metadata
from torch.distributed._shard.checkpoint.state_dict_saver import _prepare
from torch.distributed._shard.checkpoint.metadata import Metadata
from torch.distributed._shard.sharded_tensor import (
    state_dict_hook,
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)

from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)



class TestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sharded: ShardedTensor = sharded_tensor.zeros(self.spec(), 4, 4)
        self.regular = torch.nn.Parameter(torch.ones(4, 4))
        self.extra_sharded: Optional[ShardedTensor] = None
        self.extra_param: Optional[torch.nn.Parameter] = None
        self._register_state_dict_hook(state_dict_hook)

    def spec(self) -> ChunkShardingSpec:
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        return ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )


class TestDistributedCheckpointing(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_validate_metadata(self) -> None:
        module = TestModule()

        metadata, _, _ = _prepare(module.state_dict())
        self.assertTrue(
            "regular" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        module = TestModule()
        validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.extra_param = torch.nn.Parameter(torch.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, "Could not find Tensor metadata"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.regular = torch.nn.Parameter(torch.zeros(2, 4))

        with self.assertRaisesRegex(ValueError, "Incompatible tensor size"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.extra_sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        with self.assertRaisesRegex(ValueError, "Could not find ShardedTensor metadata"):
            validate_metadata(module.state_dict(), metadata)

        module = TestModule()
        module.sharded = sharded_tensor.zeros(module.spec(), 4, 2)
        with self.assertRaisesRegex(ValueError, "Incompatible ShardedTensor size"):
            validate_metadata(module.state_dict(), metadata)

    def gen_metadata(self) -> Metadata:
        module = TestModule()
        # compute the default saved metadata (must pass include_non_replicated_tensors or we'll get incomplete MD)
        metadata, _, _ = _prepare(module.state_dict())

        # _prepare only produc
        metadata = [metadata]
        dist.broadcast_object_list(metadata)

        return metadata[0]

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_shard_too_small(self) -> None:
        metadata = self.gen_metadata()

        # we make the first stored shard smaller
        self.assertTrue(
            ".sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata[".sharded"]
            .storage_metadata[0]
            .shard_metadata.shard_sizes
        )
        for i in range(len(sizes)):
            sizes[i] = 1

        module = TestModule()
        with self.assertRaisesRegex(ValueError, "only has 1 available"):
            validate_metadata(module.state_dict(), metadata)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_shard_overlap(self) -> None:
        metadata = self.gen_metadata()

        # we make the first stored shard smaller
        self.assertTrue(
            ".sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata[".sharded"]
            .storage_metadata[0]
            .shard_metadata.shard_sizes
        )
        for i in range(len(sizes)):
            sizes[i] += 1

        module = TestModule()
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_metadata(module.state_dict(), metadata)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_checkpoint_has_storage_type_mismatch(self) -> None:
        module = TestModule()

        metadata = self.gen_metadata()
        regular = metadata.state_dict_metadata["regular"]
        metadata.state_dict_metadata[".sharded"] = regular
        with self.assertRaisesRegex(ValueError, "ShardedTensorStorageMetadata but found"):
            validate_metadata(module.state_dict(), metadata)

        metadata = self.gen_metadata()
        sharded = metadata.state_dict_metadata[".sharded"]
        metadata.state_dict_metadata["regular"] = sharded
        with self.assertRaisesRegex(ValueError, "TensorStorageMetadata but found"):
            validate_metadata(module.state_dict(), metadata)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_tensor_metadata_with_missing_rank_spec(self) -> None:
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:1",
            ],
        )

        st = sharded_tensor.zeros(spec, 4, 4, dtype=torch.float64)
        mapping = dict()

        (_, md) = _prepare_sharded_tensor_write(st, "tensor", mapping)

        self.assertEqual(1, len(md.storage_metadata))
        self.assertEqual(4 * 4 * 8, md.storage_metadata[0].length)
        self.assertEqual(1, len(mapping))


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_storage_key_mapping(self) -> None:
        device = f"cuda:{dist.get_rank()}"
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        state_dict = {
            'sharded': sharded_tensor.rand(spec, (10, 10, )),
            'replicated': torch.rand(4, device=device),
            'bytes': [1, 2, 3, 4],
        }

        metadata, bytes_reqs, tensor_reqs = _prepare(state_dict)

        if self.rank == 0:
            self.assertEqual(1, len(bytes_reqs))
            self.assertEqual(2, len(tensor_reqs))

            self.assertTrue('bytes' in metadata.state_dict_metadata)
            self.assertEqual(bytes_reqs[0].storage_key, metadata.state_dict_metadata['bytes'].storage_key)

            # tensor ordering is unspecified
            if len(tensor_reqs[0].tensor.size()) == 1:
                replicated = tensor_reqs[0]
                shard = tensor_reqs[1]
            else:
                replicated = tensor_reqs[1]
                shard = tensor_reqs[0]

            self.assertTrue('replicated' in metadata.state_dict_metadata)
            self.assertEqual(replicated.storage_key, metadata.state_dict_metadata['replicated'].storage_key)
        else:
            self.assertEqual(0, len(bytes_reqs))
            self.assertEqual(1, len(tensor_reqs))
            shard = tensor_reqs[0]

            self.assertTrue('sharded' in metadata.state_dict_metadata)
            shard_keys = [sm.storage_key for sm in metadata.state_dict_metadata['sharded'].storage_metadata]
            self.assertTrue(shard.storage_key in shard_keys)

class TestStorageKeys(TestCase):
    def test_create_key_handles_collision(self):
        keys = dict()
        key0 = _create_storage_key(keys, "foo")
        key1 = _create_storage_key(keys, "foo")
        self.assertNotEqual(key0, key1)

if __name__ == "__main__":
    run_tests()
