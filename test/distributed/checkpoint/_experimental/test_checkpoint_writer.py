# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile
from typing import Any, Optional
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
    WriterHook,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class MockWriterHook(WriterHook):
    """Mock implementation of WriterHook for testing."""

    def __init__(self):
        self.pre_commit_called = False
        self.commit_called = False
        self.pre_commit_path: Optional[str] = None
        self.commit_path: Optional[str] = None
        self.pre_commit_kwargs: Optional[dict[str, Any]] = None
        self.commit_kwargs: Optional[dict[str, Any]] = None

    def pre_commit(self, path: str, **kwargs: Any):
        self.pre_commit_called = True
        self.pre_commit_path = path
        self.pre_commit_kwargs = kwargs

    def post_commit(self, path: str, **kwargs: Any):
        self.commit_called = True
        self.commit_path = path
        self.commit_kwargs = kwargs


class TestCheckpointWriterConfig(TestCase):
    def test_default_values(self):
        """Test that CheckpointWriterConfig has the correct default values."""
        options = CheckpointWriterConfig()
        self.assertEqual(options.write_barrier_timeout_secs, 600)

    def test_custom_values(self):
        """Test that CheckpointWriterConfig can be initialized with custom values."""
        options = CheckpointWriterConfig(write_barrier_timeout_secs=300)
        self.assertEqual(options.write_barrier_timeout_secs, 300)


class TestCheckpointWriter(TestCase):
    def setUp(self):
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )
        self.options = CheckpointWriterConfig()
        self.mock_barrier = MagicMock()
        self.mock_hook = MockWriterHook()

        # Create the checkpoint writer
        self.writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=self.mock_hook,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_write_creates_checkpoint_file(self):
        """Test that write creates a checkpoint file with the correct content."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(self.state_dict, checkpoint_path)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Load the checkpoint and verify its contents
        loaded_state_dict = torch.load(expected_file_path)
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)

    def test_write_calls_barrier(self):
        """Test that write calls the barrier with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(self.state_dict, checkpoint_path)

        # Verify that the barrier was called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_write_calls_commit_hooks(self):
        """Test that write calls the commit hooks with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write with additional kwargs
        kwargs = {"extra": "value"}
        self.writer.write(self.state_dict, checkpoint_path, **kwargs)

        # Verify that the pre_commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.pre_commit_called)
        self.assertEqual(self.mock_hook.pre_commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.pre_commit_kwargs is not None
            and self.mock_hook.pre_commit_kwargs["extra"],
            "value",
        )

        # Verify that the commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.commit_called)
        self.assertEqual(self.mock_hook.commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.commit_kwargs is not None
            and self.mock_hook.commit_kwargs["extra"],
            "value",
        )

    def test_write_without_barrier(self):
        """Test that write works correctly without a barrier."""
        # Create a writer without a barrier
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=None,
            commit_hook=self.mock_hook,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_barrier")

        # Call write
        writer.write(self.state_dict, checkpoint_path)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

    def test_write_without_commit_hook(self):
        """Test that write works correctly without a commit hook."""
        # Create a writer without a commit hook
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=None,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_hook")

        # Call write
        writer.write(self.state_dict, checkpoint_path)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Verify that the barrier was still called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_close(self):
        """Test that close doesn't raise any exceptions."""
        # This is a no-op in the base class, so just verify it doesn't raise
        self.writer.close()


if __name__ == "__main__":
    run_tests()
