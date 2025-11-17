# Owner(s): ["oncall: distributed"]

import os
import sys
from typing import Any, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.futures
import torch.nn
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint import (
    CheckpointException,
    load_state_dict,
    save_state_dict,
    StorageReader,
    StorageWriter,
)
from torch.distributed.checkpoint.default_planner import _create_default_local_metadata
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


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
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )


class TestDistributedCheckpointing(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_tensor_metadata_with_missing_rank_spec(self) -> None:
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:1/{device_type}:1",
            ],
        )

        st = sharded_tensor.zeros(spec, 4, 4, dtype=torch.float64)
        md = _create_default_local_metadata({"st": st})
        st_md = md.state_dict_metadata["st"]

        self.assertEqual(1, len(st_md.chunks))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_default_metadata(self) -> None:
        device = f"{device_type}:{dist.get_rank()}"
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        state_dict = {
            "sharded": sharded_tensor.rand(
                spec,
                (
                    10,
                    10,
                ),
            ),
            "replicated": torch.rand(4, device=device),
            "bytes": [1, 2, 3, 4],
        }

        metadata = _create_default_local_metadata(state_dict)
        self.assertTrue("bytes" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["bytes"], BytesStorageMetadata
        )

        self.assertTrue("replicated" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["replicated"], TensorStorageMetadata
        )
        md = metadata.state_dict_metadata["replicated"]
        self.assertEqual(md.size, state_dict["replicated"].size())
        self.assertEqual(md.properties.dtype, torch.float32)
        self.assertEqual(1, len(md.chunks))

        self.assertTrue("sharded" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["sharded"], TensorStorageMetadata
        )
        md = metadata.state_dict_metadata["sharded"]
        self.assertEqual(md.properties.dtype, torch.float32)
        self.assertEqual(md.size, state_dict["sharded"].size())
        self.assertEqual(2, len(md.chunks))


class TestStorageBase:
    def __init__(self, fail_conf):
        self.fail_conf = fail_conf
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _get_ranks(self, name):
        return self.fail_conf[name] if name in self.fail_conf else None

    def _fail_rank(self, name):
        ranks = self._get_ranks(name)
        if ranks is not None and self.rank in ranks:
            raise ValueError(f"rank fail {self.rank} for {name}")

    def _fail_rank_async(self, name, result=None):
        ranks = self._get_ranks(name)
        fut = Future()
        if ranks is not None and self.rank in ranks:
            fut.set_exception(ValueError(f"async rank fail {self.rank} for {name}"))
        else:
            fut.set_result(result)
        return fut


class FaultyStorageWriter(TestStorageBase, StorageWriter):
    def __init__(self, fail_conf):
        super().__init__(fail_conf)

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        self._fail_rank("fail_set_up_storage_writer")

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self._fail_rank("fail_prepare_local_plan")
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        self._fail_rank("fail_prepare_global_plan")
        return plans

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        self._fail_rank("fail_write_data")
        return self._fail_rank_async("fail_write_data_async", [])

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        self._fail_rank("fail_finish")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return True


class FaultyStorageReader(TestStorageBase, StorageReader):
    def __init__(self, metadata, fail_conf):
        super().__init__(fail_conf)
        self.metadata = metadata

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self._fail_rank("fail_set_up_storage_reader")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        self._fail_rank("fail_prepare_local_plan")
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        self._fail_rank("fail_prepare_global_plan")
        return plans

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        self._fail_rank("fail_read_data")
        return self._fail_rank_async("fail_read_data_async")

    def read_metadata(self) -> Metadata:
        self._fail_rank("fail_read_metadata")
        return self.metadata

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return True


class TestDistributedFailure(ShardedTensorTestBase):
    def get_spec(self):
        return ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{r}/{device_type}:{r}" for r in range(dist.get_world_size())
            ],
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_dummy_writer_works(self) -> None:
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        save_state_dict(state_dict, FaultyStorageWriter({}))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_dummy_reader_works(self) -> None:
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }
        metadata = _create_default_local_metadata(state_dict)

        load_state_dict(state_dict, FaultyStorageReader(metadata, {}))

    def _test_dist_failure(self, callback, kwargs):
        bad_ranks = next(iter(kwargs.values())) if len(kwargs) > 0 else []

        # Empty bad_ranks means it must work
        if len(bad_ranks) == 0:
            callback()
        else:
            with self.assertRaises(CheckpointException) as cm:
                callback()
            e = cast(CheckpointException, cm.exception)
            for rank, wrapped_ex in e.failures.items():
                ex = wrapped_ex[0]
                self.assertTrue(rank in bad_ranks, msg=f"{rank} did not fail")
                if not kwargs.get("ignore_exception_type", False):
                    self.assertEqual(ValueError, type(ex), str(ex))

            failed_ranks = e.failures.keys()
            for rank in bad_ranks:
                self.assertTrue(
                    rank in failed_ranks,
                    msg=f"{rank} was supposed to fail was fine",
                )

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

        def _load():
            metadata = _create_default_local_metadata(state_dict)
            load_state_dict(
                state_dict,
                storage_reader=FaultyStorageReader(metadata, kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        self._test_dist_failure(_load, kwargs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_save_error_handling(self) -> None:
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        self._test_save(state_dict, fail_finish=[0])
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        self._test_save(state_dict, fail_prepare_local_plan=[0])
        self._test_save(state_dict, fail_write_data=[2])
        self._test_save(state_dict, fail_write_data_async=[3])

        self._test_save(state_dict, coordinator=1, fail_set_up_storage_writer=[1])
        self._test_save(state_dict, coordinator=1, fail_finish=[1])

    def test_save_error_handling_no_dist(self) -> None:
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}

        self.assertFalse(dist.is_initialized())

        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        self._test_save(state_dict, fail_finish=[0])
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        self._test_save(state_dict, fail_prepare_local_plan=[0])
        self._test_save(state_dict, fail_write_data=[0])
        self._test_save(state_dict, fail_write_data_async=[0])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_load_error_handling(self) -> None:
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        self._test_load(state_dict)
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        self._test_load(state_dict, fail_read_metadata=[0], ignore_exception_type=True)
        self._test_load(state_dict, fail_prepare_local_plan=[1])
        self._test_load(state_dict, fail_read_data=[3])
        self._test_load(state_dict, fail_read_data_async=[1])

        self._test_load(state_dict, coordinator=3, fail_set_up_storage_reader=[0])
        self._test_load(
            state_dict,
            coordinator=1,
            fail_read_metadata=[3],
            ignore_exception_type=True,
        )
        self._test_load(state_dict, coordinator=2, fail_read_data=[0])
        self._test_load(state_dict, coordinator=3, fail_read_data_async=[2])
        self._test_load(state_dict, coordinator=1, fail_prepare_global_plan=[1])

    def test_load_error_handling_no_dist(self) -> None:
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}
        self._test_load(state_dict)
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        self._test_load(state_dict, fail_read_metadata=[0], ignore_exception_type=True)
        self._test_load(state_dict, fail_prepare_local_plan=[0])
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        self._test_load(state_dict, fail_read_data=[0])
        self._test_load(state_dict, fail_read_data_async=[0])


if __name__ == "__main__":
    run_tests()
