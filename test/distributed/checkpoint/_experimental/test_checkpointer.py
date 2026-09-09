# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile
from concurrent.futures import Future
from unittest.mock import Mock

import torch
from torch.distributed.checkpoint._experimental.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
)
from torch.distributed.checkpoint._experimental.checkpoint_reader import (
    CheckpointReader,
)
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
)
from torch.distributed.checkpoint._experimental.checkpointer import (
    AsyncCheckpointer,
    Checkpointer,
    SyncCheckpointer,
)
from torch.distributed.checkpoint._experimental.staging import (
    CheckpointStagerConfig,
    DefaultStager,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


def subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize the subprocess for async checkpointer tests."""
    if name != "test-async-checkpointer":
        raise AssertionError(f"Unexpected subprocess name: {name}")
    if os.getpid() == parent_pid:
        raise AssertionError("This was supposed to run in a different process")
    if os.getppid() != parent_pid:
        raise AssertionError("This was supposed to run as a child to main process")


def ckpt_writer_init_fn(**kwargs) -> CheckpointWriter:
    """Initialize a CheckpointWriter in the subprocess."""
    return CheckpointWriter(
        config=kwargs.get("config"),
        rank_info=kwargs.get("rank_info"),
    )


class TestCheckpointer(TestCase):
    """Parameterized tests that work with both sync and async checkpointers."""

    def setUp(self):
        super().setUp()
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )
        self.writer_config = CheckpointWriterConfig()

        # Create reader for testing
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
        )

        # Create test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def _create_sync_checkpointer(self) -> SyncCheckpointer:
        """Create a synchronous checkpointer."""
        writer = CheckpointWriter(
            config=self.writer_config,
            rank_info=self.rank_info,
        )
        return SyncCheckpointer(writer, self.reader)

    def _create_async_checkpointer(self) -> AsyncCheckpointer:
        """Create an asynchronous checkpointer."""
        # Create staging config for async operations
        # Use conservative settings to avoid CUDA issues in test environment
        stager_config = CheckpointStagerConfig(
            use_async_staging=True,
            use_pinned_memory=False,  # Disable to avoid CUDA memory issues
            use_shared_memory=True,
            use_non_blocking_copy=False,  # Disable to avoid CUDA issues
        )

        # Create process config
        process_config = CheckpointProcessConfig(
            subprocess_init_timeout_secs=30,
            subprocess_shutdown_timeout_secs=60,
        )

        # Create stager
        checkpoint_stager = DefaultStager(stager_config)

        # Create checkpoint process
        checkpoint_process = CheckpointProcess(
            rank_info=self.rank_info,
            config=process_config,
            subprocess_init_fn=subprocess_init_fn,
            subprocess_init_args=(
                "test-async-checkpointer",
                os.getpid(),
            ),
            checkpoint_writer_init_fn=ckpt_writer_init_fn,
            checkpoint_writer_init_args={
                "config": self.writer_config,
                "rank_info": self.rank_info,
            },
        )

        # Wait for process initialization
        checkpoint_process.process_creation_future.result()

        return AsyncCheckpointer(
            checkpoint_stager=checkpoint_stager,
            checkpoint_process=checkpoint_process,
            reader=self.reader,
        )

    def _get_checkpointers(self):
        """Get both sync and async checkpointers for parameterized testing."""
        return [
            ("sync", self._create_sync_checkpointer()),
            ("async", self._create_async_checkpointer()),
        ]

    def _save_checkpoint(self, checkpointer: Checkpointer, path, state_dict, **kwargs):
        """Save checkpoint and handle both sync/async return values."""
        result = checkpointer.save(path, state_dict, **kwargs)
        return (None, None) if result is None else result

    def _wait_for_save(self, stage_future, write_future):
        """Wait for save operation to complete."""
        if write_future is not None:
            write_future.result()
        if stage_future is not None:
            stage_future.result()

    def test_save_and_load_basic(self):
        """Test basic save and load functionality for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_{checkpointer_type}"
                    )

                    # Save the checkpoint
                    stage_future, write_future = self._save_checkpoint(
                        checkpointer, checkpoint_path, self.state_dict
                    )
                    self._wait_for_save(stage_future, write_future)

                    # Verify that the checkpoint file exists
                    checkpoint_file = os.path.join(
                        checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
                    )
                    self.assertTrue(os.path.exists(checkpoint_file))

                    # Load the checkpoint using the checkpointer
                    loaded_state_dict = checkpointer.load(checkpoint_path)

                    # Verify the loaded state dictionary
                    self.assertIn("model", loaded_state_dict)
                    self.assertIn("optimizer", loaded_state_dict)
                    self.assertEqual(loaded_state_dict["epoch"], 5)
                    self.assertEqual(loaded_state_dict["step"], 1000)

                finally:
                    checkpointer.close()

    def test_load_with_map_location(self):
        """Test loading with map_location for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_map_{checkpointer_type}"
                    )

                    # Save the checkpoint
                    stage_future, write_future = self._save_checkpoint(
                        checkpointer, checkpoint_path, self.state_dict
                    )
                    self._wait_for_save(stage_future, write_future)

                    # Load with map_location
                    loaded_state_dict = checkpointer.load(
                        checkpoint_path, default_map_location="cpu"
                    )

                    # Verify the loaded state dictionary
                    self.assertIn("model", loaded_state_dict)
                    self.assertEqual(loaded_state_dict["epoch"], 5)

                finally:
                    checkpointer.close()

    def test_partial_load(self):
        """Test partial loading for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_partial_{checkpointer_type}"
                    )

                    # Save the full checkpoint
                    stage_future, write_future = self._save_checkpoint(
                        checkpointer, checkpoint_path, self.state_dict
                    )
                    self._wait_for_save(stage_future, write_future)

                    # Create a partial state dictionary
                    partial_state_dict = {
                        "model": torch.nn.Linear(10, 5).state_dict(),
                        "epoch": None,
                    }

                    # Load only the keys in partial_state_dict
                    loaded_state_dict = checkpointer.load(
                        checkpoint_path, state_dict=partial_state_dict
                    )

                    # Verify partial loading worked
                    self.assertIn("model", loaded_state_dict)
                    self.assertIn("epoch", loaded_state_dict)
                    self.assertEqual(loaded_state_dict["epoch"], 5)
                    self.assertNotIn("step", loaded_state_dict)
                    self.assertNotIn("optimizer", loaded_state_dict)

                finally:
                    checkpointer.close()

    def test_load_strict_mode(self):
        """Test strict mode loading for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_strict_{checkpointer_type}"
                    )

                    # Save a checkpoint with limited keys
                    limited_state_dict = {"model": torch.nn.Linear(10, 5).state_dict()}
                    stage_future, write_future = self._save_checkpoint(
                        checkpointer, checkpoint_path, limited_state_dict
                    )
                    self._wait_for_save(stage_future, write_future)

                    # Try to load with more keys than exist in checkpoint
                    partial_state_dict = {
                        "model": torch.nn.Linear(10, 5).state_dict(),
                        "missing_key": None,
                    }

                    # Should raise error in strict mode
                    with self.assertRaises(RuntimeError) as cm:
                        checkpointer.load(
                            checkpoint_path, state_dict=partial_state_dict, strict=True
                        )

                    self.assertIn("missing keys", str(cm.exception))

                    # Should work without strict mode
                    loaded_state_dict = checkpointer.load(
                        checkpoint_path, state_dict=partial_state_dict, strict=False
                    )
                    self.assertIn("model", loaded_state_dict)

                finally:
                    checkpointer.close()

    def test_save_with_kwargs(self):
        """Test save with additional kwargs for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_kwargs_{checkpointer_type}"
                    )

                    # For sync checkpointer, we can pass arbitrary kwargs to the writer
                    # For async checkpointer, we test without kwargs to avoid conflicts
                    if checkpointer_type == "sync":
                        # Sync checkpointer passes kwargs directly to writer, so arbitrary kwargs are OK
                        stage_future, write_future = self._save_checkpoint(
                            checkpointer,
                            checkpoint_path,
                            self.state_dict,
                            custom_arg="test_value",
                            another_arg=42,
                        )
                    else:
                        # Async checkpointer has complex kwargs handling between stager and writer
                        # Just test basic save without kwargs to avoid conflicts
                        stage_future, write_future = self._save_checkpoint(
                            checkpointer,
                            checkpoint_path,
                            self.state_dict,
                        )

                    self._wait_for_save(stage_future, write_future)

                    # Verify checkpoint was created
                    checkpoint_file = os.path.join(
                        checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
                    )
                    self.assertTrue(os.path.exists(checkpoint_file))

                finally:
                    checkpointer.close()

    def test_nested_dict_partial_load(self):
        """Test loading nested dictionaries partially for both sync and async."""
        for checkpointer_type, checkpointer in self._get_checkpointers():
            with self.subTest(checkpointer_type=checkpointer_type):
                try:
                    # Create a checkpoint with nested dictionaries
                    nested_state_dict = {
                        "model": {
                            "layer1": {
                                "weight": torch.randn(5, 10),
                                "bias": torch.randn(5),
                            },
                            "layer2": {
                                "weight": torch.randn(2, 5),
                                "bias": torch.randn(2),
                            },
                        },
                        "metadata": {"epoch": 10, "step": 2000},
                    }

                    checkpoint_path = os.path.join(
                        self.temp_dir, f"checkpoint_nested_{checkpointer_type}"
                    )

                    # Save the nested state dict
                    stage_future, write_future = self._save_checkpoint(
                        checkpointer, checkpoint_path, nested_state_dict
                    )
                    self._wait_for_save(stage_future, write_future)

                    # Create a partial state dictionary with nested structure
                    partial_state_dict = {
                        "model": {
                            "layer1": {"weight": None},  # Only request layer1.weight
                        },
                        "metadata": {"epoch": None},  # Only request metadata.epoch
                    }

                    # Load only the keys in partial_state_dict
                    loaded_state_dict = checkpointer.load(
                        checkpoint_path, state_dict=partial_state_dict
                    )

                    # Verify that the nested keys were correctly loaded
                    self.assertIn("model", loaded_state_dict)
                    self.assertIn("layer1", loaded_state_dict["model"])
                    self.assertIn("weight", loaded_state_dict["model"]["layer1"])
                    self.assertIn("metadata", loaded_state_dict)
                    self.assertIn("epoch", loaded_state_dict["metadata"])

                    # Verify values were loaded correctly
                    self.assertTrue(
                        torch.allclose(
                            loaded_state_dict["model"]["layer1"]["weight"],
                            nested_state_dict["model"]["layer1"]["weight"],
                        )
                    )
                    self.assertEqual(loaded_state_dict["metadata"]["epoch"], 10)

                    # Verify that keys not in the partial_state_dict are not loaded
                    self.assertNotIn("layer2", loaded_state_dict["model"])
                    self.assertNotIn("step", loaded_state_dict["metadata"])

                finally:
                    checkpointer.close()


class TestAsyncCheckpointerSpecific(TestCase):
    """Tests specific to AsyncCheckpointer functionality."""

    def setUp(self):
        super().setUp()
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )
        self.writer_config = CheckpointWriterConfig()

        # Create reader for testing
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
        )

        # Create test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def _create_async_checkpointer(self) -> AsyncCheckpointer:
        """Helper method to create AsyncCheckpointer with real components."""
        # Create staging config for async operations
        # Use conservative settings to avoid CUDA issues in test environment
        stager_config = CheckpointStagerConfig(
            use_async_staging=True,
            use_pinned_memory=False,  # Disable to avoid CUDA memory issues
            use_shared_memory=True,
            use_non_blocking_copy=False,  # Disable to avoid CUDA issues
        )

        # Create process config
        process_config = CheckpointProcessConfig(
            subprocess_init_timeout_secs=30,
            subprocess_shutdown_timeout_secs=60,
        )

        # Create stager
        checkpoint_stager = DefaultStager(stager_config)

        # Create checkpoint process
        checkpoint_process = CheckpointProcess(
            rank_info=self.rank_info,
            config=process_config,
            subprocess_init_fn=subprocess_init_fn,
            subprocess_init_args=(
                "test-async-checkpointer",
                os.getpid(),
            ),
            checkpoint_writer_init_fn=ckpt_writer_init_fn,
            checkpoint_writer_init_args={
                "config": self.writer_config,
                "rank_info": self.rank_info,
            },
        )

        # Wait for process initialization
        checkpoint_process.process_creation_future.result()

        return AsyncCheckpointer(
            checkpoint_stager=checkpoint_stager,
            checkpoint_process=checkpoint_process,
            reader=self.reader,
        )

    def test_async_returns_futures(self):
        """Test that async save returns futures."""
        checkpointer = self._create_async_checkpointer()
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_futures")

        try:
            # Save the checkpoint asynchronously
            result = checkpointer.save(checkpoint_path, self.state_dict)

            # Verify that futures are returned
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            stage_future, write_future = result
            self.assertIsInstance(stage_future, Future)
            self.assertIsInstance(write_future, Future)

            # Wait for completion
            stage_future.result()
            write_future.result()

        finally:
            checkpointer.close()

    def test_async_sequential_saves_wait(self):
        """Test that sequential async saves wait for previous operations."""
        checkpointer = self._create_async_checkpointer()

        try:
            # First save
            checkpoint_path1 = os.path.join(self.temp_dir, "checkpoint_seq_1")
            stage_future1, write_future1 = checkpointer.save(
                checkpoint_path1, self.state_dict
            )

            # Second save (should wait for first to complete)
            checkpoint_path2 = os.path.join(self.temp_dir, "checkpoint_seq_2")
            modified_state_dict = self.state_dict.copy()
            modified_state_dict["epoch"] = 10
            stage_future2, write_future2 = checkpointer.save(
                checkpoint_path2, modified_state_dict
            )

            # Wait for both to complete
            write_future1.result()
            write_future2.result()

            # Verify both checkpoints were created with correct content
            checkpoint_file1 = os.path.join(
                checkpoint_path1, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            checkpoint_file2 = os.path.join(
                checkpoint_path2, f"checkpoint_{self.rank_info.global_rank}.pt"
            )

            self.assertTrue(os.path.exists(checkpoint_file1))
            self.assertTrue(os.path.exists(checkpoint_file2))

            loaded1 = torch.load(checkpoint_file1)
            loaded2 = torch.load(checkpoint_file2)

            self.assertEqual(loaded1["epoch"], 5)
            self.assertEqual(loaded2["epoch"], 10)

        finally:
            checkpointer.close()

    def test_async_multiple_saves_ordering(self):
        """Test that multiple async saves maintain proper ordering."""
        checkpointer = self._create_async_checkpointer()

        try:
            # Create multiple state dicts
            state_dicts = [
                {"epoch": 1, "model": torch.nn.Linear(5, 3).state_dict()},
                {"epoch": 2, "model": torch.nn.Linear(5, 3).state_dict()},
                {"epoch": 3, "model": torch.nn.Linear(5, 3).state_dict()},
            ]

            # Save multiple checkpoints
            futures = []
            checkpoint_paths = []
            for i, state_dict in enumerate(state_dicts, 1):
                checkpoint_path = os.path.join(self.temp_dir, f"multi_{i}")
                checkpoint_paths.append(checkpoint_path)
                stage_future, write_future = checkpointer.save(
                    checkpoint_path, state_dict
                )
                futures.append((stage_future, write_future))

            # Wait for all to complete
            for stage_future, write_future in futures:
                stage_future.result()
                write_future.result()

            # Verify all checkpoints exist and have correct content
            for i, checkpoint_path in enumerate(checkpoint_paths, 1):
                checkpoint_file = os.path.join(
                    checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
                )
                self.assertTrue(os.path.exists(checkpoint_file))

                loaded = torch.load(checkpoint_file)
                self.assertEqual(loaded["epoch"], i)

        finally:
            checkpointer.close()

    def test_async_error_handling(self):
        """Test error handling in async operations."""
        # Create checkpointer with mocked components to simulate errors
        mock_stager = Mock()
        mock_process = Mock()
        mock_reader = Mock()

        # Mock staging to return a completed future
        mock_staging_future = Future()
        mock_staging_future.set_result({"staged": "data"})
        mock_stager.stage.return_value = mock_staging_future

        # Mock process write to raise an error
        mock_write_future = Future()
        mock_write_future.set_exception(RuntimeError("Write failed"))
        mock_process.write.return_value = mock_write_future

        checkpointer = AsyncCheckpointer(
            checkpoint_stager=mock_stager,
            checkpoint_process=mock_process,
            reader=mock_reader,
        )

        try:
            # This should not raise immediately
            stage_future, write_future = checkpointer.save("/tmp/test", self.state_dict)

            # But waiting for the write future should raise the error
            with self.assertRaises(RuntimeError) as cm:
                write_future.result()

            self.assertIn("Write failed", str(cm.exception))

        finally:
            checkpointer.close()

    def test_async_future_results(self):
        """Test the results returned by async futures."""
        checkpointer = self._create_async_checkpointer()
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_results")

        try:
            # Save checkpoint
            stage_future, write_future = checkpointer.save(
                checkpoint_path, self.state_dict
            )

            # Both futures should complete successfully
            stage_result = stage_future.result()
            write_result = write_future.result()

            # Stage result is wrapped by wrap_future() so it returns None on success
            # This is intentional - the stage_future indicates completion, not data access
            self.assertIsNone(stage_result)

            # Write result should be None (success indicator)
            self.assertIsNone(write_result)

        finally:
            checkpointer.close()


if __name__ == "__main__":
    run_tests()
