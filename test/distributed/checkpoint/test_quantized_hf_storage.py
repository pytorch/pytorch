# Owner(s): ["oncall: distributed checkpointing"]

import tempfile
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestQuantizedHfStorage(TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = self.temp_dir.name

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_dequantization(self):
        """Test that quantized tensors are properly dequantized during read operations."""
        reader = QuantizedHuggingFaceStorageReader(self.path, thread_count=1)

        # Test data
        quantized_tensor = torch.ones(4, 4, dtype=torch.float32)
        scale_inv = torch.tensor([[2.0]], dtype=torch.float32)

        # Mock the safetensors file for reading data
        mock_file = MagicMock()

        # Mock get_slice to return a tensor that can be sliced
        def mock_get_slice(tensor_name):
            mock_tensor = MagicMock()
            mock_tensor.__getitem__ = lambda self, slices: quantized_tensor
            return mock_tensor

        mock_file.get_slice = mock_get_slice
        mock_file.get_tensor.return_value = scale_inv

        reader._weight_scale_mapping = {
            "model.layers.0.self_attn.kv_b_proj.weight": "model.layers.0.self_attn.kv_b_proj.weight_scale_inv",
        }

        # Create a read request for quantized tensor
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex(
                fqn="model.layers.0.self_attn.kv_b_proj.weight",
                offset=torch.Size([0, 0]),
            ),
            dest_index=MetadataIndex(
                fqn="model.layers.0.self_attn.kv_b_proj.weight",
                offset=torch.Size([0, 0]),
            ),
            storage_offsets=[0, 0],
            dest_offsets=[0, 0],
            lengths=[4, 4],
        )

        # Mock planner
        target_tensor = torch.zeros(4, 4, dtype=torch.float32)
        mock_planner = MagicMock()
        mock_planner.resolve_tensor.return_value = target_tensor

        # Test the _process_read_request method
        reader._process_read_request(mock_file, read_item, mock_planner)

        # Verify the tensor was dequantized (ones * 2.0 = twos)
        expected_result = torch.ones(4, 4, dtype=torch.float32) * 2.0
        mock_planner.commit_tensor.assert_called_once()

        # Check that target_tensor was updated correctly
        args, _ = mock_planner.commit_tensor.call_args
        committed_tensor = args[1]  # second argument is the tensor
        torch.testing.assert_close(committed_tensor, expected_result)


if __name__ == "__main__":
    run_tests()
