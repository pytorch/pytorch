# Owner(s): ["oncall: distributed checkpointing"]
import os
import shutil
import tempfile
from typing import Any

import torch
from torch.distributed.checkpoint._experimental.checkpoint_reader import (
    CheckpointReader,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCheckpointReader(TestCase):
    def setUp(self):
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )

        # Create the checkpoint reader
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": {
                "weight": torch.randn(10, 5),
                "bias": torch.randn(5),
                "test_list": [torch.randn(2), torch.randn(2)],
            },
            "optimizer": {
                "param_groups": [
                    {"lr": 0.01, "test_list": [torch.randn(2), torch.randn(2)]}
                ]
            },
            "epoch": 5,
            "step": 1000,
        }

        # Create a test checkpoint file
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            self.checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        torch.save(self.state_dict, checkpoint_file)

    def move_tensors_to_device(self, state_dict: Any, device: str) -> Any:
        """
        Recursively move all tensors in a nested dictionary to CUDA.

        Args:
            state_dict (dict): A dictionary potentially containing nested dictionaries and tensors.

        Returns:
            dict: A new dictionary with all tensors moved to CUDA.
        """
        if isinstance(state_dict, dict):
            return {
                key: self.move_tensors_to_device(value, device)
                for key, value in state_dict.items()
            }
        elif isinstance(state_dict, list):
            return [self.move_tensors_to_device(item, device) for item in state_dict]
        elif isinstance(state_dict, torch.Tensor):
            return state_dict.cuda() if device == "cpu" else state_dict.cpu()
        else:
            return state_dict

    def deep_compare(self, obj1: Any, obj2: Any) -> bool:
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            if obj1.keys() != obj2.keys():
                return False
            return all(self.deep_compare(obj1[key], obj2[key]) for key in obj1)
        elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(
                self.deep_compare(item1, item2) for item1, item2 in zip(obj1, obj2)
            )
        elif isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
            return torch.equal(obj1, obj2)
        else:
            return obj1 == obj2

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_read_checkpoint(self):
        """Test that read correctly reads a checkpoint file."""

        # Call read
        read_state_dict, missing_keys = self.reader.read(self.checkpoint_path)
        self.assertEqual(missing_keys, [])

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertTrue(self.deep_compare(read_state_dict, self.state_dict))

        # No hooks to verify since we removed them

    def test_read_with_map_location(self):
        """Test that read correctly uses the map_location parameter."""
        # Call read with map_location='cpu'
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        read_state_dict, _ = self.reader.read(
            self.checkpoint_path, map_location=map_location
        )

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertEqual(read_state_dict["epoch"], 5)
        self.assertEqual(read_state_dict["step"], 1000)
        self.assertEqual(read_state_dict["model"]["weight"].device.type, map_location)
        read_state_dict, _ = self.reader.read(
            self.checkpoint_path, map_location=map_location
        )

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertEqual(read_state_dict["epoch"], 5)
        self.assertEqual(read_state_dict["step"], 1000)
        self.assertEqual(read_state_dict["model"]["weight"].device.type, map_location)

    def test_read_nonexistent_checkpoint(self):
        """Test that read raises FileNotFoundError for a nonexistent checkpoint."""
        # Set up a path to a nonexistent checkpoint
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_checkpoint")

        # Call read and expect a FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.reader.read(nonexistent_path)

    def test_read_with_kwargs(self):
        """Test that read correctly passes kwargs."""
        # Call read with additional kwargs
        kwargs = {"extra": "value"}
        self.reader.read(self.checkpoint_path, **kwargs)

    def test_partial_read(self):
        """Test that read with state_dict correctly loads only the requested keys."""
        # Create a partial state dictionary with only some keys
        partial_state_dict = {}
        partial_state_dict["optimizer"] = None
        partial_state_dict["model"] = {"weight": torch.randn(10, 5)}
        partial_state_dict["epoch"] = None
        # Call read with state_dict
        updated_state_dict, _ = self.reader.read(
            self.checkpoint_path,
            partial_state_dict,
        )

        # Verify that the updated state dictionary contains values from both dictionaries
        self.assertIn("model", updated_state_dict)
        self.assertIn("epoch", updated_state_dict)
        self.assertTrue(
            torch.equal(
                updated_state_dict["model"]["weight"],
                self.state_dict["model"]["weight"],
            )
        )

        self.assertTrue(
            self.deep_compare(
                updated_state_dict["optimizer"], self.state_dict["optimizer"]
            )
        )
        self.assertEqual(updated_state_dict["epoch"], 5)  # From checkpoint

        self.assertNotIn("bias", updated_state_dict["model"])
        self.assertNotIn("step", updated_state_dict)

    def test_partial_read_missing_keys(self):
        """Test that partial_read correctly reports missing keys."""
        # Create a partial state dictionary with keys that don't exist in the checkpoint
        partial_state_dict = {
            "model": None,
            "nonexistent_key": None,  # This key doesn't exist in the checkpoint
            "another_missing_key": {"nested": None},  # This key also doesn't exist
        }

        # Call read with state_dict
        _, missing_keys = self.reader.read(
            self.checkpoint_path,
            partial_state_dict,
        )

        # Verify that missing keys are correctly reported
        self.assertIn("nonexistent_key", missing_keys)
        self.assertIn("another_missing_key", missing_keys)

        # Verify that keys that exist in the checkpoint are not in missing_keys
        self.assertNotIn("model", missing_keys)

    def test_partial_read_different_dtypes(self):
        """Test that partial_read correctly handles different tensor dtypes."""
        # Create a state dictionary with tensors of different dtypes
        dtype_state_dict = {
            "float32": torch.randn(10, 10, dtype=torch.float32),
            "float64": torch.randn(10, 10, dtype=torch.float64),
            "int32": torch.randint(-100, 100, (10, 10), dtype=torch.int32),
            "int64": torch.randint(-100, 100, (10, 10), dtype=torch.int64),
            "bool": torch.randint(0, 2, (10, 10), dtype=torch.bool),
        }

        # Save the state dictionary
        dtype_checkpoint_path = os.path.join(self.temp_dir, "dtype_checkpoint")
        os.makedirs(dtype_checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            dtype_checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        torch.save(dtype_state_dict, checkpoint_file)

        # Create a partial state dictionary requesting tensors of each dtype
        partial_state_dict = {
            "float32": torch.randn(10, 10, dtype=torch.float32),
            "float64": None,
            "int32": None,
            "int64": None,
            "bool": None,
        }

        # Load the partial state dictionary
        updated_state_dict, _ = self.reader.read(
            os.path.dirname(checkpoint_file),
            partial_state_dict,
        )

        # Verify that tensors of each dtype were loaded correctly
        for key in dtype_state_dict:
            self.assertIn(key, updated_state_dict)
            self.assertEqual(updated_state_dict[key].dtype, dtype_state_dict[key].dtype)
            self.assertTrue(
                torch.allclose(updated_state_dict[key], dtype_state_dict[key])
            )


if __name__ == "__main__":
    run_tests()
