# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile

import torch
from torch.distributed.checkpoint._experimental.checkpoint_reader import (
    CheckpointReader,
)
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
)
from torch.distributed.checkpoint._experimental.checkpointer import SyncCheckpointer
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

    def test_read_with_map_location(self):
        """Test reading a checkpoint with a specific map_location."""
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_map_location")

        # Save the checkpoint
        self.checkpointer.save(self.state_dict, checkpoint_path)

        # Load the checkpoint with map_location='cpu'
        loaded_state_dict = self.checkpointer.load(
            checkpoint_path, default_map_location="cpu"
        )

        # Verify the loaded state dictionary
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)

    def test_partial_load(self):
        """Test loading only specific keys from a checkpoint."""
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_partial")

        # Save the full checkpoint
        self.checkpointer.save(self.state_dict, checkpoint_path)

        # Create a partial state dictionary with only some keys
        partial_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "epoch": None,  # Will be loaded from checkpoint
        }

        # Load only the keys in partial_state_dict
        loaded_state_dict = self.checkpointer.load(
            checkpoint_path, state_dict=partial_state_dict, default_map_location="cpu"
        )

        # Verify that the loaded state dictionary contains values from the checkpoint
        self.assertIn("model", loaded_state_dict)
        self.assertIn("epoch", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)  # From checkpoint

        # Verify that keys not in the partial_state_dict are not loaded
        self.assertNotIn("step", loaded_state_dict)
        self.assertNotIn("optimizer", loaded_state_dict)

        # Verify that the loaded state dictionary is the same object as the input
        self.assertIs(loaded_state_dict, partial_state_dict)

    def test_partial_load_with_nested_dict(self):
        """Test loading only specific nested keys from a checkpoint."""
        # Create a checkpoint with nested dictionaries
        nested_state_dict = {
            "model": {
                "layer1": {"weight": torch.randn(5, 10), "bias": torch.randn(5)},
                "layer2": {"weight": torch.randn(2, 5), "bias": torch.randn(2)},
            },
            "metadata": {"epoch": 10, "step": 2000},
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_nested")

        # Create a writer and save the nested state dict
        writer = CheckpointWriter(
            config=self.writer_config,
            rank_info=self.rank_info,
        )
        writer.write(nested_state_dict, checkpoint_path)

        # Create a partial state dictionary with nested structure
        partial_state_dict = {
            "model": {
                "layer1": {"weight": None},  # Only request layer1.weight
            },
            "metadata": {"epoch": None},  # Only request metadata.epoch
        }

        # Load only the keys in partial_state_dict
        loaded_state_dict = self.checkpointer.load(
            checkpoint_path, state_dict=partial_state_dict, default_map_location="cpu"
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


if __name__ == "__main__":
    run_tests()
