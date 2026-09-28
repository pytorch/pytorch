# Owner(s): ["oncall: distributed"]

import io
import math
import os
import threading
from unittest.mock import patch

import fsspec
import fsspec.implementations.memory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint._fsspec_filesystem import (
    FileSystem,
    FsspecReader,
    FsspecWriter,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.utils import CheckpointException
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
BACKEND = torch.distributed.get_default_backend_for_device(device_type)


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestFSSpec(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsspec(self):
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(MyTestModule().to(device_type))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optim),
            }

            dcp.save(
                state_dict=state_dict,
                storage_writer=FsspecWriter(CHECKPOINT_DIR),
                planner=dcp.DefaultSavePlanner(),
            )

        model_2 = FSDP(MyTestModule().to(device_type))
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertNotEqual(n_p1[1], n_p2[1])

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dcp.load(
                state_dict=state_dict,
                storage_reader=FsspecReader(CHECKPOINT_DIR),
                planner=dcp.DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FsspecReader(CHECKPOINT_DIR),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model_2, optim_2, optim_state["optim"]
            )
            optim_2.load_state_dict(flattened_osd)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertEqual(n_p1[1], n_p2[1])

        def opt_at(opt, idx):
            return list(iter(opt.state.values()))[idx]

        # Adam lazily creates its state
        self.assertEqual(opt_at(optim, 0)["exp_avg"], opt_at(optim_2, 0)["exp_avg"])
        self.assertEqual(
            opt_at(optim, 0)["exp_avg_sq"], opt_at(optim_2, 0)["exp_avg_sq"]
        )

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_overwrite(self):
        t1, t2 = torch.randn(10), torch.randn(10)

        dcp.save(
            {"random": t1}, storage_writer=FsspecWriter(self.temp_dir, overwrite=False)
        )
        dcp.save(
            {"random": t2}, storage_writer=FsspecWriter(self.temp_dir, overwrite=True)
        )

        sd = {"random": torch.zeros(10)}
        dcp.load(sd, checkpoint_id=self.temp_dir)
        self.assertTrue(torch.allclose(sd["random"], t2))

        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            dcp.save(
                {"random": t2},
                storage_writer=FsspecWriter(self.temp_dir, overwrite=False),
            )


@instantiate_parametrized_tests
class TestFileSystem(TestCase):
    @with_temp_dir
    def test_remove_on_fail(self):
        fs = FileSystem()
        path = fs.init_path(self.temp_dir)

        write_file = fs.concat_path(path, "writeable")
        with self.assertRaises(OSError):
            with fs.create_stream(write_file, "w") as s:
                s.write("aaa")
                raise OSError("fail")
        self.assertFalse(fs.exists(write_file))

        read_file = fs.concat_path(path, "readable")
        with fs.create_stream(read_file, "w") as s:
            s.write("bbb")
        self.assertTrue(fs.exists(read_file))

        with self.assertRaises(OSError):
            with fs.create_stream(read_file, "r") as s:
                raise OSError("fail")
        self.assertTrue(fs.exists(read_file))

    @patch("os.sync")
    def test_fsspec_without_fileno_support(self, mock_os_sync):
        """
        fsspec's "memory://" protocol simulates cloud storage
        by not supporting .fileno() and raising io.UnsupportedOperation.
        This tests that the stream is flushed and fsync degrades gracefully.
        """
        checkpoint_dir = "memory://test_checkpoint_with_no_file_no"

        # Create a dummy state dict
        state_dict = {"tensor": torch.randn(10)}

        # Spy on the MemoryFile flush method to ensure it gets called
        with patch.object(
            fsspec.implementations.memory.MemoryFile, "flush", autospec=True
        ) as mock_flush:
            # Save using FsspecWriter
            dcp.save(
                state_dict=state_dict,
                storage_writer=FsspecWriter(checkpoint_dir),
                planner=dcp.DefaultSavePlanner(),
                no_dist=True,
            )

            # Assert that flush() was explicitly called on the stream
            self.assertTrue(
                mock_flush.called, "Expected stream.flush() to be called explicitly."
            )

        # Verify it saved properly and can be loaded
        load_dict = {"tensor": torch.zeros(10)}
        dcp.load(
            state_dict=load_dict,
            storage_reader=FsspecReader(checkpoint_dir),
            planner=dcp.DefaultLoadPlanner(),
            no_dist=True,
        )

        self.assertEqual(state_dict["tensor"], load_dict["tensor"])

        # os.sync() may be called on backends that don't support per-file fsync
        self.assertLessEqual(mock_os_sync.call_count, 2)

    @parametrize("batch_size", [1, 2, 64])
    def test_fsspec_reader_batched_cat_ranges(self, batch_size):
        checkpoint_dir = f"memory://test_fsspec_batched_load_{batch_size}"

        state_dict = {
            "t1": torch.randn(10),
            "t2": torch.randn(5, 5),
            "t3": torch.randn(8, 4, 2),
            "t4": torch.randn(20),
            "bytes_data": io.BytesIO(b"custom_serialized_bytes_data"),
        }

        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        load_dict = {
            "t1": torch.zeros(10),
            "t2": torch.zeros(5, 5),
            "t3": torch.zeros(8, 4, 2),
            "t4": torch.zeros(20),
            "bytes_data": io.BytesIO(),
        }
        reader = FsspecReader(checkpoint_dir, max_batch_size=batch_size)
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ) as mock_cat_ranges:
            dcp.load(
                state_dict=load_dict,
                storage_reader=reader,
                planner=dcp.DefaultLoadPlanner(),
                no_dist=True,
            )
            self.assertEqual(
                mock_cat_ranges.call_count, math.ceil(len(state_dict) / batch_size)
            )

        self.assertEqual(state_dict["t1"], load_dict["t1"])
        self.assertEqual(state_dict["t2"], load_dict["t2"])
        self.assertEqual(state_dict["t3"], load_dict["t3"])
        self.assertEqual(state_dict["t4"], load_dict["t4"])
        self.assertEqual(
            state_dict["bytes_data"].getvalue(),
            load_dict["bytes_data"].getvalue(),
        )

    def test_fsspec_reader_batched_max_batch_bytes(self):
        checkpoint_dir = "memory://test_fsspec_batched_max_bytes"
        state_dict = {
            "t1": torch.randn(100),
            "t2": torch.randn(100),
            "t3": torch.randn(100),
        }
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        load_dict = {
            "t1": torch.zeros(100),
            "t2": torch.zeros(100),
            "t3": torch.zeros(100),
        }
        # Set max_batch_bytes=1 so each item exceeds the byte budget and forms its own batch
        reader = FsspecReader(checkpoint_dir, max_batch_size=64, max_batch_bytes=1)
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ) as mock_cat_ranges:
            dcp.load(
                state_dict=load_dict,
                storage_reader=reader,
                planner=dcp.DefaultLoadPlanner(),
                no_dist=True,
            )
            self.assertEqual(mock_cat_ranges.call_count, 3)

        self.assertEqual(state_dict["t1"], load_dict["t1"])
        self.assertEqual(state_dict["t2"], load_dict["t2"])
        self.assertEqual(state_dict["t3"], load_dict["t3"])

    def test_fsspec_reader_sync_fallback(self):
        checkpoint_dir = "memory://test_fsspec_sync_fallback"
        state_dict = {"t1": torch.randn(10), "t2": torch.randn(5)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        load_dict = {"t1": torch.zeros(10), "t2": torch.zeros(5)}
        reader = FsspecReader(checkpoint_dir)
        # MemoryFileSystem inherits AbstractFileSystem.cat_ranges and is not an AsyncFileSystem,
        # so _supports_batched_cat_ranges() should be False and read_data should fall back to super().read_data()
        self.assertFalse(reader._supports_batched_cat_ranges())
        dcp.load(
            state_dict=load_dict,
            storage_reader=reader,
            planner=dcp.DefaultLoadPlanner(),
            no_dist=True,
        )
        self.assertEqual(state_dict["t1"], load_dict["t1"])
        self.assertEqual(state_dict["t2"], load_dict["t2"])

    @parametrize("invalid_val", [0, -1, -100])
    def test_fsspec_reader_clamp_max_batch_size(self, invalid_val):
        checkpoint_dir = "memory://test_clamp_batch_size"
        reader = FsspecReader(
            checkpoint_dir,
            max_batch_size=invalid_val,
            max_batch_bytes=invalid_val,
            cpu_workers=invalid_val,
        )
        self.assertEqual(reader.max_batch_size, 1)
        self.assertEqual(reader.max_batch_bytes, 1)
        self.assertEqual(reader.cpu_workers, 1)

    def test_fsspec_reader_empty_local_world_size_env(self):
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": ""}):
            reader = FsspecReader("memory://test_empty_local_world_size")
            self.assertGreaterEqual(reader.cpu_workers, 1)

    def test_fsspec_reader_concurrent_planner_thread_safety(self):
        checkpoint_dir = "memory://test_concurrent_planner_safety"
        state_dict = {
            "nested": {
                f"b_{i}": io.BytesIO(f"payload_{i}".encode()) for i in range(16)
            },
            "tensors": [torch.randn(8) for _ in range(16)],
        }
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        load_dict = {
            "nested": {f"b_{i}": io.BytesIO() for i in range(16)},
            "tensors": [torch.zeros(8) for _ in range(16)],
        }
        reader = FsspecReader(checkpoint_dir, max_batch_size=4, cpu_workers=8)
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ) as mock_cat_ranges:
            dcp.load(
                state_dict=load_dict,
                storage_reader=reader,
                planner=dcp.DefaultLoadPlanner(),
                no_dist=True,
            )
            self.assertEqual(mock_cat_ranges.call_count, 8)

        for i in range(16):
            self.assertEqual(
                state_dict["nested"][f"b_{i}"].getvalue(),
                load_dict["nested"][f"b_{i}"].getvalue(),
            )
            self.assertEqual(state_dict["tensors"][i], load_dict["tensors"][i])

    def test_fsspec_reader_cat_ranges_error(self):
        checkpoint_dir = "memory://test_cat_ranges_error"
        state_dict = {"t1": torch.randn(10)}

        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir)
        with patch.object(
            reader.fs.fs,
            "cat_ranges",
            side_effect=RuntimeError("cat_ranges failed"),
        ):
            load_dict = {"t1": torch.zeros(10)}
            with self.assertRaisesRegex(CheckpointException, "cat_ranges failed"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

    def test_fsspec_reader_cpu_exception_propagation(self):
        checkpoint_dir = "memory://test_cpu_exception"
        state_dict = {"t1": torch.randn(10)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir)
        load_dict = {"t1": torch.zeros(10)}

        with (
            patch.object(reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges),
            patch(
                "torch.distributed.checkpoint._fsspec_filesystem.narrow_tensor_by_index",
                side_effect=RuntimeError("simulated cpu crash"),
            ),
        ):
            with self.assertRaisesRegex(CheckpointException, "simulated cpu crash"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

    def test_fsspec_reader_concurrent_shutdown(self):
        checkpoint_dir = "memory://test_shutdown"
        state_dict = {f"t_{i}": torch.randn(4) for i in range(8)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir, max_batch_size=2, cpu_workers=4)
        load_dict = {f"t_{i}": torch.zeros(4) for i in range(8)}
        initial_threads = threading.active_count()
        with (
            patch.object(reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges),
            patch(
                "torch.distributed.checkpoint._fsspec_filesystem.narrow_tensor_by_index",
                side_effect=RuntimeError("shutdown test"),
            ),
        ):
            with self.assertRaisesRegex(CheckpointException, "shutdown test"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

        self.assertLessEqual(threading.active_count(), initial_threads + 1)
        self.assertFalse(any("ThreadPoolExecutor" in t.name for t in threading.enumerate()))

    def test_fsspec_reader_cat_ranges_inband_exception(self):
        checkpoint_dir = "memory://test_cat_ranges_inband_exception"
        state_dict = {f"t_{i}": torch.randn(4) for i in range(4)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir)
        real_cat_ranges = reader.fs.fs.cat_ranges

        def cat_ranges_returning_exception(paths, starts, ends, **kwargs):
            chunks = real_cat_ranges(paths, starts, ends, **kwargs)
            return [OSError("simulated range failure"), *chunks[1:]]

        load_dict = {f"t_{i}": torch.zeros(4) for i in range(4)}
        with patch.object(
            reader.fs.fs, "cat_ranges", side_effect=cat_ranges_returning_exception
        ):
            with self.assertRaisesRegex(CheckpointException, "Failed to read bytes"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

    def test_fsspec_reader_destination_disjoint_fallback(self):
        checkpoint_dir = "memory://test_disjoint_fallback"
        state_dict = {
            "t1": torch.randn(10, 10),
            "t2": torch.randn(20),
        }
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        # Non-contiguous destination: transposed slice
        base_t1 = torch.zeros(10, 10)
        non_contig_dst = base_t1.t()

        load_dict = {
            "t1": non_contig_dst,
            "t2": torch.zeros(20),
        }
        reader = FsspecReader(checkpoint_dir)
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ):
            dcp.load(
                state_dict=load_dict,
                storage_reader=reader,
                planner=dcp.DefaultLoadPlanner(),
                no_dist=True,
            )
        self.assertEqual(state_dict["t1"], load_dict["t1"])
        self.assertEqual(state_dict["t2"], load_dict["t2"])

        # Directly test overlapping vs disjoint slices of the same storage
        from torch.distributed.checkpoint._fsspec_filesystem import _destinations_disjoint

        base_buf = torch.zeros(30)
        overlap_a = base_buf[0:15]
        overlap_b = base_buf[10:25]
        self.assertFalse(_destinations_disjoint([overlap_a, overlap_b]))

        disjoint_a = base_buf[0:10]
        disjoint_b = base_buf[10:20]
        self.assertTrue(_destinations_disjoint([disjoint_a, disjoint_b]))

        # Tensor subclasses that raise on data_ptr() safely fall back to sequential copy
        class SubclassRaisingDataPtr(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

            def data_ptr(self):
                raise RuntimeError("data_ptr not supported for custom tensor subclass")

        subclass_tensor = SubclassRaisingDataPtr(torch.zeros(20))
        self.assertFalse(_destinations_disjoint([subclass_tensor, torch.zeros(10)]))

        # Verify end-to-end dcp.load succeeds when state_dict contains SubclassRaisingDataPtr
        load_subclass_dict = {
            "t1": SubclassRaisingDataPtr(torch.zeros(10, 10)),
            "t2": torch.zeros(20),
        }
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ):
            dcp.load(
                state_dict=load_subclass_dict,
                storage_reader=reader,
                planner=dcp.DefaultLoadPlanner(),
                no_dist=True,
            )
        self.assertEqual(state_dict["t1"], load_subclass_dict["t1"])
        self.assertEqual(state_dict["t2"], load_subclass_dict["t2"])

    def test_fsspec_reader_custom_planner_hook_pairing(self):
        checkpoint_dir = "memory://test_planner_hook_pairing"
        state_dict = {f"t_{i}": torch.randn(10) for i in range(4)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        class StrictPairingPlanner(dcp.DefaultLoadPlanner):
            def __init__(self):
                super().__init__()
                self.active_resolve = False

            def resolve_tensor(self, read_item):
                if self.active_resolve:
                    raise AssertionError("resolve_tensor called before previous commit_tensor")
                self.active_resolve = True
                return super().resolve_tensor(read_item)

            def commit_tensor(self, read_item, tensor):
                if not self.active_resolve:
                    raise AssertionError("commit_tensor called without active resolve_tensor")
                self.active_resolve = False
                super().commit_tensor(read_item, tensor)

        load_dict = {f"t_{i}": torch.zeros(10) for i in range(4)}
        reader = FsspecReader(checkpoint_dir, max_batch_size=4)
        with patch.object(
            reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
        ):
            dcp.load(
                state_dict=load_dict,
                storage_reader=reader,
                planner=StrictPairingPlanner(),
                no_dist=True,
            )
        for k in state_dict:
            self.assertEqual(state_dict[k], load_dict[k])

        bytes_dir = "memory://test_planner_bytes_only"
        bytes_state = {"step": 5, "config": [1, 2, 3]}
        dcp.save(
            state_dict=bytes_state,
            storage_writer=FsspecWriter(bytes_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )
        loaded_bytes = {"step": None, "config": None}
        bytes_reader = FsspecReader(bytes_dir, max_batch_size=2)
        with patch.object(
            bytes_reader.fs.fs, "cat_ranges", wraps=bytes_reader.fs.fs.cat_ranges
        ):
            dcp.load(
                state_dict=loaded_bytes,
                storage_reader=bytes_reader,
                planner=StrictPairingPlanner(),
                no_dist=True,
            )
        self.assertEqual(bytes_state, loaded_bytes)

    def test_fsspec_reader_inference_mode(self):
        checkpoint_dir = "memory://test_inference_mode"
        state_dict = {"t1": torch.randn(10), "t2": torch.randn(10)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        with torch.inference_mode():
            load_dict = {"t1": torch.zeros(10), "t2": torch.zeros(10)}
            reader = FsspecReader(checkpoint_dir)
            with patch.object(
                reader.fs.fs, "cat_ranges", wraps=reader.fs.fs.cat_ranges
            ):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )
            self.assertEqual(state_dict["t1"], load_dict["t1"])
            self.assertEqual(state_dict["t2"], load_dict["t2"])

    def test_supports_batched_cat_ranges_wrappers(self):
        from fsspec.implementations.cached import (
            SimpleCacheFileSystem,
            WholeFileCacheFileSystem,
        )
        from fsspec.implementations.dirfs import DirFileSystem

        mem = fsspec.implementations.memory.MemoryFileSystem()
        dfs = DirFileSystem(path="test_dir", fs=mem)
        reader_dfs = FsspecReader("memory://test_dfs")
        reader_dfs.fs.fs = dfs
        self.assertFalse(reader_dfs._supports_batched_cat_ranges())

        nested_dfs = DirFileSystem(path="inner", fs=dfs)
        reader_nested = FsspecReader("memory://test_nested")
        reader_nested.fs.fs = nested_dfs
        self.assertFalse(reader_nested._supports_batched_cat_ranges())

        scfs = SimpleCacheFileSystem(fs=mem)
        dfs_cache = DirFileSystem(path="test_dir", fs=scfs)
        reader_dfs_cache = FsspecReader("memory://test_dfs_cache")
        reader_dfs_cache.fs.fs = dfs_cache
        self.assertFalse(reader_dfs_cache._supports_batched_cat_ranges())

        reader_scfs = FsspecReader("memory://test_scfs")
        reader_scfs.fs.fs = scfs
        self.assertFalse(reader_scfs._supports_batched_cat_ranges())

        wfc = WholeFileCacheFileSystem(fs=mem)
        reader_wfc = FsspecReader("memory://test_wfc")
        reader_wfc.fs.fs = wfc
        self.assertFalse(reader_wfc._supports_batched_cat_ranges())

        class DummyAsyncFS(fsspec.asyn.AsyncFileSystem):
            async_impl = True

            def cat_ranges(self, *args, **kwargs):
                return []

        dummy = DummyAsyncFS()
        reader_async = FsspecReader("memory://test_async")
        reader_async.fs.fs = dummy
        self.assertTrue(reader_async._supports_batched_cat_ranges())

        nested_async = DirFileSystem(
            path="inner", fs=DirFileSystem(path="outer", fs=dummy)
        )
        reader_nested_async = FsspecReader("memory://test_nested_async")
        reader_nested_async.fs.fs = nested_async
        self.assertTrue(reader_nested_async._supports_batched_cat_ranges())

        class SyncAsyncFS(fsspec.asyn.AsyncFileSystem):
            async_impl = False

            def cat_ranges(self, *args, **kwargs):
                return []

        sync_dummy = SyncAsyncFS()
        reader_sync_dummy = FsspecReader("memory://test_sync_dummy")
        reader_sync_dummy.fs.fs = sync_dummy
        self.assertFalse(reader_sync_dummy._supports_batched_cat_ranges())

        nested_sync = DirFileSystem(path="dir", fs=sync_dummy)
        reader_nested_sync = FsspecReader("memory://test_nested_sync")
        reader_nested_sync.fs.fs = nested_sync
        self.assertFalse(reader_nested_sync._supports_batched_cat_ranges())

    def test_fsspec_reader_cat_ranges_length_mismatch(self):
        checkpoint_dir = "memory://test_len_mismatch"
        state_dict = {"t1": torch.randn(10)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir)
        load_dict = {"t1": torch.zeros(10)}
        with patch.object(
            reader.fs.fs, "cat_ranges", return_value=[b"too_short"]
        ):
            with self.assertRaisesRegex(CheckpointException, "Read .* bytes for"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

    def test_fsspec_reader_cat_ranges_count_mismatch(self):
        checkpoint_dir = "memory://test_count_mismatch"
        state_dict = {"t1": torch.randn(10), "t2": torch.randn(10)}
        dcp.save(
            state_dict=state_dict,
            storage_writer=FsspecWriter(checkpoint_dir),
            planner=dcp.DefaultSavePlanner(),
            no_dist=True,
        )

        reader = FsspecReader(checkpoint_dir, max_batch_size=2)
        load_dict = {"t1": torch.zeros(10), "t2": torch.zeros(10)}
        with patch.object(
            reader.fs.fs, "cat_ranges", return_value=[b"dummy"]
        ):
            with self.assertRaisesRegex(CheckpointException, "cat_ranges returned 1 chunks for 2 ranges"):
                dcp.load(
                    state_dict=load_dict,
                    storage_reader=reader,
                    planner=dcp.DefaultLoadPlanner(),
                    no_dist=True,
                )

    def test_fsspec_reader_workers_config(self):
        checkpoint_dir = "memory://test_workers_config"
        reader = FsspecReader(checkpoint_dir, cpu_workers=8)
        self.assertEqual(reader.cpu_workers, 8)


if __name__ == "__main__":
    run_tests()
