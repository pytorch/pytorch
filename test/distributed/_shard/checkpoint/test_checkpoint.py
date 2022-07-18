# Owner(s): ["oncall: distributed"]

import random
import sys
from typing import Optional, List, Union
from torch.distributed._shard.checkpoint import (
    StorageReader,
    StorageWriter,
    CheckpointException,
    load_state_dict,
    save_state_dict,
)

import torch
import torch.distributed as dist
import torch.nn
import torch.futures
from torch.futures import Future
from torch.testing._internal.common_utils import TestCase

from torch.distributed._shard.checkpoint.resharding import (
    _prepare_sharded_tensor_write,
    _create_storage_key
)

from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.checkpoint.state_dict_loader import (
    validate_metadata,
)

from torch.distributed._shard.checkpoint.state_dict_saver import (
    _prepare,
)

from torch.distributed._shard.checkpoint.metadata import (
    Metadata,
    BytesReadRequest,
    BytesWriteRequest,
    TensorReadRequest,
    TensorWriteRequest,
)

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

        metadata, _, _ = _prepare(module.state_dict(), True)
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
        metadata, _, _ = _prepare(module.state_dict(), True)

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
            "sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata["sharded"]
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
            "sharded" in metadata.state_dict_metadata,
            f"keys: {metadata.state_dict_metadata.keys()}",
        )

        sizes = (
            metadata.state_dict_metadata["sharded"]
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
        metadata.state_dict_metadata["sharded"] = regular
        with self.assertRaisesRegex(ValueError, "ShardedTensorStorageMetadata but found"):
            validate_metadata(module.state_dict(), metadata)

        metadata = self.gen_metadata()
        sharded = metadata.state_dict_metadata["sharded"]
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

        metadata, bytes_reqs, tensor_reqs = _prepare(state_dict, write_replicated_data=self.rank == 0)

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




class TestStorageBase:
    def __init__(
        self,
        fail_conf
    ):
        self.fail_conf = fail_conf
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _get_ranks(self, name):
        return self.fail_conf[name] if name in self.fail_conf else None

    def _fail_rank(self, name):
        ranks = self._get_ranks(name)
        if ranks is not None and self.rank in ranks:
            raise ValueError(f"rank fail {self.rank} for {name}")

    def _fail_rank_async(self, name):
        ranks = self._get_ranks(name)
        fut = Future()
        if ranks is not None and self.rank in ranks:
            fut.set_exception(ValueError(f"async rank fail {self.rank} for {name}"))
        else:
            fut.set_result(None)
        return fut


class FaultyStorageWriter(TestStorageBase, StorageWriter):
    def __init__(
        self,
        fail_conf
    ):
        super(FaultyStorageWriter, self).__init__(fail_conf)

    def prepare(self) -> None:
        self._fail_rank("fail_prepare")

    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        self._fail_rank("fail_write_bytes_on_ranks")
        return self._fail_rank_async("fail_write_bytes_on_ranks_async")

    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        self._fail_rank("fail_write_tensors_on_ranks")
        return self._fail_rank_async("fail_write_tensors_on_ranks_async")

    def finish(self, metadata: Metadata) -> None:
        self._fail_rank("fail_finish")

    def prepare_storage(self, storage_writes: List[Union[TensorWriteRequest, BytesWriteRequest]]) -> None:
        self._fail_rank("fail_prepare_storage")

class FaultyStorageReader(TestStorageBase, StorageReader):
    def __init__(
        self,
        metadata,
        fail_conf
    ):
        super(FaultyStorageReader, self).__init__(fail_conf)
        self.metadata = metadata

    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        self._fail_rank("fail_read_bytes")
        bad_ranks = self._get_ranks("fail_deser_bytes")
        for r in requests:
            if bad_ranks is not None and self.rank in bad_ranks:
                # this is not "guaranteed" to fail, but hard to beat
                rand = random.Random(1237)
                r.bytes.write(rand.randbytes(32))
            else:
                torch.save([1, 2, 3], r.bytes)

        return self._fail_rank_async("fail_read_bytes_async")

    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        self._fail_rank("fail_read_tensors")
        return self._fail_rank_async("fail_read_tensors_async")

    def read_metadata(self) -> Metadata:
        self._fail_rank("fail_read_metadata")
        return self.metadata

class TestDistributedFailure(ShardedTensorTestBase):
    def get_spec(self):
        return ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{r}/cuda:{r}" for r in range(dist.get_world_size())
            ]
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_dummy_writer_works(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        save_state_dict(state_dict, FaultyStorageWriter({}))


    def _test_dist_failure(self, callback, kwargs):
        bad_ranks = list(kwargs.values())[0] if len(kwargs) > 0 else []

        # Empty bad_ranks means it must work
        if len(bad_ranks) == 0:
            callback()
        else:
            with self.assertRaises(CheckpointException) as cm:
                callback()
            e = cm.exception
            for rank, ex in e.failures.items():
                self.assertTrue(rank in bad_ranks, msg=f"{rank} did not fail")
                if not kwargs.get("ignore_exception_type", False):
                    self.assertEqual(ValueError, type(ex), str(ex))

            failed_ranks = e.failures.keys()
            for rank in bad_ranks:
                self.assertTrue(rank in failed_ranks, msg=f"{rank} was supposed to fail was fine")


    def _test_save(self, state_dict, coordinator=0, **kwargs):
        no_dist = not dist.is_initialized()

        def _save():
            save_state_dict(
                state_dict,
                storage_writer=FaultyStorageWriter(kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )
        self._test_dist_failure(_save, kwargs)

    def _test_load(self, state_dict, coordinator=0, **kwargs):
        no_dist = not dist.is_initialized()
        write_replicated = dist.is_initialized() and dist.get_rank() == coordinator

        def _load():
            metadata, _, _ = _prepare(state_dict, write_replicated)
            load_state_dict(
                state_dict,
                storage_reader=FaultyStorageReader(metadata, kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        self._test_dist_failure(_load, kwargs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_save_error_handling(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self._test_save(state_dict, fail_prepare=[0])
        self._test_save(state_dict, fail_finish=[0])

        self._test_save(state_dict, fail_prepare_storage=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks=[1])
        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[2])
        self._test_save(state_dict, fail_write_bytes_on_ranks=[3])
        self._test_save(state_dict, fail_write_bytes_on_ranks_async=[1])

        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[1, 3])

        self._test_save(state_dict, coordinator=1, fail_prepare=[1])
        self._test_save(state_dict, coordinator=1, fail_finish=[1])


    def test_save_error_handling_no_dist(self) -> None:
        state_dict = {
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self.assertFalse(dist.is_initialized())

        self._test_save(state_dict, fail_prepare=[0])
        self._test_save(state_dict, fail_finish=[0])

        self._test_save(state_dict, fail_prepare_storage=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks=[0])
        self._test_save(state_dict, fail_write_tensors_on_ranks_async=[0])
        self._test_save(state_dict, fail_write_bytes_on_ranks=[0])
        self._test_save(state_dict, fail_write_bytes_on_ranks_async=[0])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_error_handling(self) -> None:
        state_dict = {
            'sharded': sharded_tensor.rand(self.get_spec(), 20, 20),
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }

        self._test_load(state_dict)
        self._test_load(state_dict, fail_read_metadata=[0])
        self._test_load(state_dict, fail_read_bytes=[1])
        self._test_load(state_dict, fail_read_bytes_async=[2])
        self._test_load(state_dict, fail_read_tensors=[3])
        self._test_load(state_dict, fail_read_tensors_async=[1])
        # We don't want to depend on the actual exception raised by pickle
        self._test_load(state_dict, fail_deser_bytes=[2], ignore_exception_type=True)

        self._test_load(state_dict, coordinator=1, fail_read_metadata=[3])
        self._test_load(state_dict, coordinator=2, fail_read_bytes=[0])
        self._test_load(state_dict, coordinator=3, fail_read_tensors_async=[2])


    def test_load_error_handling_no_dist(self) -> None:
        state_dict = {
            'replicated': torch.rand(10, 10),
            'bytes': [1, 2, 3, 4]
        }
        self._test_load(state_dict)
        self._test_load(state_dict, fail_read_metadata=[0])
        self._test_load(state_dict, fail_read_bytes=[0])
        self._test_load(state_dict, fail_read_bytes_async=[0])
        self._test_load(state_dict, fail_read_tensors=[0])
        self._test_load(state_dict, fail_read_tensors_async=[0])
        self._test_load(state_dict, fail_deser_bytes=[0], ignore_exception_type=True)
if __name__ == "__main__":
    run_tests()
