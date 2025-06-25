# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.checkpoint._experimental.builder import (
    make_async_checkpointer,
    make_sync_checkpointer,
)
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
    SyncCheckpointer,
)
from torch.distributed.checkpoint._experimental.staging import (
    CheckpointStagerConfig,
    DefaultStager,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSyncCheckpointer(TestCase):
    def setUp(self):
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )
        self.writer_config = CheckpointWriterConfig()
        self.writer = CheckpointWriter(
            config=self.writer_config,
            rank_info=self.rank_info,
        )

        # Create reader for testing
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
        )

        # Create sync checkpointer
        self.checkpointer = SyncCheckpointer(self.writer, self.reader)

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

    def test_sync_save_and_read(self):
        """Test saving and reading a checkpoint synchronously."""
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_sync")

        # Save the checkpoint synchronously
        result = self.checkpointer.save(self.state_dict, checkpoint_path)
        self.assertIsNone(result)  # Sync mode returns None

        # Verify that the checkpoint file exists
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        # Load the checkpoint using the checkpointer
        loaded_state_dict = self.checkpointer.load(checkpoint_path)

        # Verify the loaded state dictionary
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)


class TestAsyncCheckpointer(TestCase):
    def setUp(self):
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )

        # Create reader for testing
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
        )

        # Create staging method
        self.checkpoint_stager = DefaultStager(
            config=CheckpointStagerConfig(use_async_staging=False, use_pinned_memory=False),
        )

        # Create checkpoint process
        self.checkpoint_process = CheckpointProcess(
            rank_info=self.rank_info,
            config=CheckpointProcessConfig(),
            subprocess_init_fn=lambda: None,
            subprocess_init_args=(),
            checkpoint_writer_init_fn=lambda rank_info: CheckpointWriter(
                config=CheckpointWriterConfig(),
                rank_info=rank_info,
            ),
            checkpoint_writer_init_args={},
        )

        # Create async checkpointer
        self.checkpointer = AsyncCheckpointer(
            checkpoint_stager=self.checkpoint_stager,
            checkpoint_process=self.checkpoint_process,
            reader=self.reader,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # Clean up the checkpointer
        self.checkpointer.close()
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_async_save_and_read(self):
        """Test saving and reading a checkpoint with async mode."""
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_async")

        # Save the checkpoint asynchronously
        stage_future, write_future = self.checkpointer.save(self.state_dict, checkpoint_path)

        # Wait for both futures to complete
        stage_future.result()
        write_future.result()

        # Verify that the checkpoint file exists
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        # Load the checkpoint using the checkpointer
        loaded_state_dict = self.checkpointer.load(checkpoint_path)

        # Verify the loaded state dictionary
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)


from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    with_temp_dir,
    with_checkpoint_logging,
)


class TestMultiRankCheckpointer(DTensorTestBase):
    """
    Test checkpointing with multiple ranks and barriers.
    """

    @with_comms
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @with_temp_dir
    @with_checkpoint_logging
    def test_sync_checkpointer_with_barriers(self):
        """Test synchronous checkpointing with barriers across multiple ranks."""
        # Use shared temp directory provided by @with_temp_dir decorator
        temp_dir = self.temp_dir

        # Create rank-specific data
        rank = self.rank
        world_size = self.world_size

        # Create sync checkpointer using the factory function
        checkpointer = make_sync_checkpointer(use_dist_barrier=True)

        # Create rank-specific state dict
        state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
            "rank_specific_data": f"data_from_rank_{rank}",
            "rank_tensor": torch.ones(1) * rank,
        }

        # Save checkpoint from all ranks synchronously
        checkpoint_path = os.path.join(temp_dir, "multi_rank_sync_checkpoint")
        result = checkpointer.save(state_dict, checkpoint_path)
        self.assertIsNone(result)  # Sync mode returns None

        # Verify that checkpoint files exist for all ranks
        for r in range(world_size):
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_{r}.pt")
            self.assertTrue(os.path.exists(checkpoint_file))

        # Load checkpoint for this rank
        loaded_state_dict = checkpointer.load(checkpoint_path)

        # Verify the loaded state dictionary
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)
        self.assertEqual(loaded_state_dict["rank_specific_data"], f"data_from_rank_{rank}")
        self.assertTrue(torch.allclose(loaded_state_dict["rank_tensor"], torch.ones(1) * rank))

    @with_comms
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @with_temp_dir
    @with_checkpoint_logging
    def test_async_checkpointer_with_barriers(self):
        """Test asynchronous checkpointing with barriers across multiple ranks."""
        # Use shared temp directory provided by @with_temp_dir decorator
        temp_dir = self.temp_dir

        # Create rank-specific data
        rank = self.rank
        world_size = self.world_size

        # Create async checkpointer using the factory function
        checkpointer = make_async_checkpointer()

        try:
            # Create rank-specific state dict
            state_dict = {
                "model": torch.nn.Linear(10, 5).state_dict(),
                "optimizer": {"param_groups": [{"lr": 0.01}]},
                "epoch": 5,
                "step": 1000,
                "rank_specific_data": f"data_from_rank_{rank}",
                "rank_tensor": torch.ones(1) * rank,
            }

            # Save checkpoint from all ranks asynchronously
            checkpoint_path = os.path.join(temp_dir, "multi_rank_async_checkpoint")
            stage_future, write_future = checkpointer.save(state_dict, checkpoint_path)

            # Wait for both futures to complete
            stage_future.result()
            write_future.result()
            dist.barrier()

            # Verify that checkpoint files exist for all ranks
            for r in range(world_size):
                checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_{r}.pt")
                self.assertTrue(os.path.exists(checkpoint_file))

            # Load checkpoint for this rank
            loaded_state_dict = checkpointer.load(checkpoint_path)

            # Verify the loaded state dictionary
            self.assertIn("model", loaded_state_dict)
            self.assertIn("optimizer", loaded_state_dict)
            self.assertEqual(loaded_state_dict["epoch"], 5)
            self.assertEqual(loaded_state_dict["step"], 1000)
            self.assertEqual(loaded_state_dict["rank_specific_data"], f"data_from_rank_{rank}")
            self.assertTrue(torch.allclose(loaded_state_dict["rank_tensor"], torch.ones(1) * rank))

        finally:
            # Clean up
            checkpointer.close()


if __name__ == "__main__":
    run_tests()
