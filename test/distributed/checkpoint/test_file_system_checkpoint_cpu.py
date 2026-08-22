# Owner(s): ["oncall: distributed"]

import io
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, IO

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    load_state_dict,
    save,
    save_state_dict,
)
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms, FileSystem
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.stateful import Stateful
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    MyShardedModel1,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    get_test_extension_registry,
    Rot13Example,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


_THREAD_COUNTS = {1, 2}


def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
) -> bool:
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_2.tensor),
                    lambda msg: f"{msg}\nKey {key}'s shard does not match",
                )
        elif isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2),
                lambda msg: f"{msg}\nKey {key}'s tensor does not match",
            )
        elif isinstance(value_1, Stateful):
            self.assertEqual(value_1, value_2)

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from test/distributed/_sharded_tensor/test_sharded_tensor.py
class MyShardedModel3(torch.nn.Module):
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        super().__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )


class BlobState:
    def __init__(self, value: IO[bytes]) -> Any:
        self.state = {"blob": value}

    def state_dict(self) -> dict[str, Any]:
        return self.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state = state_dict

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlobState) and self.state == other.state

    def __repr__(self) -> str:
        return f"BlobState({self.state['blob']})"


class TestDistributedStateDictSaveLoad(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadRot13(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_tensor_and_blob(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[Rot13Example()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding.  Note there is no extension
            # specification here; it is determined dynamically from the metadata.
            fs_reader = FileSystemReader(
                path=path, _extension_registry=get_test_extension_registry()
            )
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadZStandard(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[ZStandard()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestFileSystemWriterFlushBehavior(TestCase):
    def test_no_close_writer_does_not_flush_raw_stream(self) -> None:
        """
        Verify that NoCloseWriter.close() does not eagerly call flush()
        on the underlying raw stream for each write item.
        """

        class MockRawStream(io.BytesIO):
            def __init__(self) -> None:
                super().__init__()
                self.flush_count = 0

            def flush(self) -> None:
                self.flush_count += 1
                super().flush()

        raw_stream = MockRawStream()
        transforms = _StorageWriterTransforms()
        write_item = WriteItem(
            index=MetadataIndex("test_item"),
            type=WriteItemType.BYTE_IO,
        )

        transform_to, _ = transforms.transform_save_stream(write_item, raw_stream)
        transform_to.write(b"test_payload")
        transform_to.close()

        # Closing transform_to (NoCloseWriter) should NOT flush the underlying raw stream
        self.assertEqual(raw_stream.flush_count, 0)
        self.assertEqual(raw_stream.getvalue(), b"test_payload")

    def test_transform_with_extension_does_not_flush_raw_stream(self) -> None:
        """
        Verify that closing a stream transformed by extensions completes
        transformation without flushing the underlying raw stream.
        """

        class MockRawStream(io.BytesIO):
            def __init__(self) -> None:
                super().__init__()
                self.flush_count = 0

            def flush(self) -> None:
                self.flush_count += 1
                super().flush()

        raw_stream = MockRawStream()
        transforms = _StorageWriterTransforms(extensions=[Rot13Example()])
        write_item = WriteItem(
            index=MetadataIndex("test_item"),
            type=WriteItemType.BYTE_IO,
        )

        transform_to, descriptors = transforms.transform_save_stream(
            write_item, raw_stream
        )
        transform_to.write(b"Hello World")
        transform_to.close()

        # Extension transformed data is written, but raw stream is not eagerly flushed
        self.assertEqual(raw_stream.flush_count, 0)
        self.assertNotEqual(raw_stream.getvalue(), b"Hello World")
        self.assertEqual(len(descriptors), 1)

    @parametrize("sync_files", [True, False])
    def test_file_system_writer_flush_counts(self, sync_files: bool) -> None:
        """
        Verify that FileSystemWriter does not perform per-tensor flushes on shard streams.
        With N tensors in a single shard file:
          - If sync_files=True, stream.flush() is called 2 times per file (1 explicit before
            os.fsync + 1 during stream.close()).
          - If sync_files=False, stream.flush() is called 1 time per file (0 explicit + 1 during stream.close()).
          - In both cases, there are NO per-item flushes (which previously added N flushes per shard file).
        """
        flush_counts: dict[str, int] = {}

        class FlushTrackingFileSystem(FileSystem):
            @contextmanager
            def create_stream(self, path, mode):
                with super().create_stream(path, mode) as stream:
                    path_str = str(path)
                    flush_counts.setdefault(path_str, 0)

                    orig_flush = stream.flush

                    def tracking_flush():
                        flush_counts[path_str] += 1
                        return orig_flush()

                    stream.flush = tracking_flush  # type: ignore[method-assign]
                    yield stream

        with tempfile.TemporaryDirectory() as path:
            num_tensors = 10
            state_dict_to_save = {
                f"tensor_{i}": torch.ones(5, 5) * i for i in range(num_tensors)
            }

            fs_writer = FileSystemWriter(
                path=path,
                sync_files=sync_files,
                single_file_per_rank=True,
                thread_count=1,
            )
            fs_writer.fs = FlushTrackingFileSystem()

            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            # Separate shard file and metadata file flush counts
            shard_flushes = [
                count
                for p, count in flush_counts.items()
                if not p.endswith(".tmp") and "__0_0.distcp" in p
            ]
            metadata_flushes = [
                count
                for p, count in flush_counts.items()
                if p.endswith(".tmp") or ".metadata" in p
            ]

            if sync_files:
                # Shard file: 1 explicit flush before fsync + 1 on close (NOT num_tensors + 2)
                self.assertEqual(shard_flushes, [2])
                self.assertEqual(metadata_flushes, [2])
            else:
                # Shard file: 0 explicit flush before fsync + 1 on close (NOT num_tensors + 1)
                self.assertEqual(shard_flushes, [1])
                self.assertEqual(metadata_flushes, [1])

            # Verify saved checkpoint loads back correctly
            state_dict_to_load = {
                f"tensor_{i}": torch.zeros(5, 5) for i in range(num_tensors)
            }
            fs_reader = FileSystemReader(path=path)
            load(
                state_dict=state_dict_to_load,
                storage_reader=fs_reader,
            )
            assert_state_dict_equal(self, state_dict_to_load, state_dict_to_save)


class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_shard_tensor(self, thread_count) -> None:
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)

        path = paths[0]

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


class TestDistributedReshardOnLoad(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def get_file_path(self) -> str:
        paths = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(paths)
        return paths[0]

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        res = torch.zeros(tensor.shape, device="cpu") if dist.get_rank() == 0 else None
        tensor.gather(out=res)
        return res

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_load_with_different_shard_plan(self, thread_count) -> None:
        path = self.get_file_path()

        # We hardcode the assumption of how many shards are around
        self.assertEqual(self.world_size, dist.get_world_size())

        specs = [
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                ],
            ),
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                    "rank:1",
                    "rank:0",
                ],
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0],
                        shard_sizes=[1, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[3, 0],
                        shard_sizes=[3, 20],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[6, 0],
                        shard_sizes=[3, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[9, 0],
                        shard_sizes=[1, 20],
                        placement="rank:0",
                    ),
                ]
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[8, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0",
                    ),
                ]
            ),
        ]

        for s0 in specs:
            for s1 in specs:
                if s0 == s1:
                    continue

                dist.barrier()

                model_to_save = MyShardedModel3(s0)
                model_to_save._register_state_dict_hook(state_dict_hook)
                state_dict_to_save = model_to_save.state_dict()

                fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
                save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

                dist.barrier()

                model_to_load = MyShardedModel3(s1)
                model_to_load._register_state_dict_hook(state_dict_hook)
                state_dict_to_load_to = model_to_load.state_dict()
                dist.barrier()

                fs_reader = FileSystemReader(path=path)
                load_state_dict(
                    state_dict=state_dict_to_load_to, storage_reader=fs_reader
                )

                dist.barrier()
                store_tensor = self.load_tensor(model_to_save.sharded_tensor)
                dist.barrier()
                load_tensor = self.load_tensor(model_to_load.sharded_tensor)

                if dist.get_rank() == 0:
                    self.assertTrue(
                        torch.allclose(store_tensor, load_tensor),
                        msg=lambda msg: f"{msg}\n{s0} vs {s1}",
                    )

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_load_rowwise_to_colwise(self, thread_count) -> None:
        path = self.get_file_path()
        self.assertEqual(self.world_size, dist.get_world_size())

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        model_to_save = MyShardedModel3(src_spec).cuda(dist.get_rank())
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        model_to_load = MyShardedModel3(dst_spec).cuda(dist.get_rank())
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        fs_reader = FileSystemReader(path=path)

        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # We can't use torch.allclose since each ST has a different sharding spec
        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        if dist.get_rank() == 0:
            self.assertTrue(torch.allclose(store_tensor, load_tensor))

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_save_load_bytes(self, thread_count) -> None:
        path = self.get_file_path()

        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        self.assertEqual([1], state_dict_to_load["bytes0"])
        self.assertEqual("string", state_dict_to_load["bytes1"])

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_switch_between_sharded_tensor_to_tensor(self, thread_count) -> None:
        path = self.get_file_path()
        tensor_size = 32

        specs = [
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                ],
            ),
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                    "rank:1",
                    "rank:0",
                ],
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[8],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8],
                        shard_sizes=[tensor_size - 8],
                        placement="rank:0",
                    ),
                ]
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[10],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[10],
                        shard_sizes=[tensor_size - 10],
                        placement="rank:1",
                    ),
                ]
            ),
        ]

        for save_spec in specs:
            for load_spec in specs:
                save_dict = {
                    "sharded": sharded_tensor.rand(save_spec, tensor_size),
                    "replicated": torch.rand(tensor_size, device="cpu"),
                }
                dist.broadcast(save_dict["replicated"], src=0)

                fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
                save_state_dict(state_dict=save_dict, storage_writer=fs_writer)

                # Freaky Friday the tensors
                load_dict = {
                    "sharded": torch.zeros(tensor_size, device="cpu"),
                    "replicated": sharded_tensor.zeros(load_spec, tensor_size),
                }

                fs_reader = FileSystemReader(path=path)
                load_state_dict(state_dict=load_dict, storage_reader=fs_reader)

                save_dict_sharded = self.load_tensor(save_dict["sharded"])
                load_dict_replicated = self.load_tensor(load_dict["replicated"])

                if dist.get_rank() == 0:
                    self.assertTrue(
                        torch.allclose(save_dict_sharded, load_dict["sharded"]),
                        lambda msg: f"{msg}\nsave-spec {save_spec} load-spec {load_spec}",
                    )

                    self.assertTrue(
                        torch.allclose(save_dict["replicated"], load_dict_replicated),
                        lambda msg: f"{msg}\nsave-spec {save_spec} load-spec {load_spec}",
                    )


instantiate_parametrized_tests(TestDistributedStateDictSaveLoad)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadRot13)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadWithSharedTensor)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadZStandard)
instantiate_parametrized_tests(TestDistributedReshardOnLoad)
instantiate_parametrized_tests(TestFileSystemWriterFlushBehavior)

if __name__ == "__main__":
    run_tests()
