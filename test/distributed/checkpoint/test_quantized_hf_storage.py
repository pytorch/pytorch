# Owner(s): ["oncall: distributed checkpointing"]

import tempfile
from unittest.mock import MagicMock, patch

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
        """Test quantized tensors with weights and scales in both same and different files."""
        reader = QuantizedHuggingFaceStorageReader(self.path, thread_count=1)

        # Test data for two different weights
        quantized_tensor1 = torch.ones(4, 4, dtype=torch.float32)
        quantized_tensor2 = (
            torch.ones(4, 4, dtype=torch.float32) * 3.0
        )  # Different values
        scale_inv1 = torch.tensor([[2.0]], dtype=torch.float32)
        scale_inv2 = torch.tensor([[0.5]], dtype=torch.float32)  # Different scale

        # Define weight and scale tensor names
        weight1_fqn = "model.layers.0.self_attn.q_proj.weight"  # Scale in same file
        scale1_fqn = "model.layers.0.self_attn.q_proj.weight_scale_inv"
        weight2_fqn = (
            "model.layers.0.self_attn.k_proj.weight"  # Scale in different file
        )
        scale2_fqn = "model.layers.0.self_attn.k_proj.weight_scale_inv"

        file1_name = "model-00001-of-00002.safetensors"
        file2_name = "model-00002-of-00002.safetensors"

        # Setup weight-scale mapping and file locations
        reader._weight_scale_mapping = {
            weight1_fqn: scale1_fqn,
            weight2_fqn: scale2_fqn,
        }
        reader._weight_map = {
            weight1_fqn: file1_name,  # Weight in file 1
            scale1_fqn: file1_name,  # Scale also in file 1 (same file scenario)
            weight2_fqn: file1_name,  # Weight in file 1
            scale2_fqn: file2_name,  # Scale in file 2 (different file scenario)
        }

        # Mock the main safetensors file (file1)
        mock_file1 = MagicMock()

        # Mock get_slice to return different tensors based on tensor name
        def mock_get_slice(tensor_name):
            mock_tensor = MagicMock()
            if tensor_name == weight1_fqn:
                mock_tensor.__getitem__ = lambda _, __: quantized_tensor1
            elif tensor_name == weight2_fqn:
                mock_tensor.__getitem__ = lambda _, __: quantized_tensor2
            return mock_tensor

        mock_file1.get_slice = mock_get_slice

        # Mock get_tensor for same-file scale (scale1)
        mock_file1.get_tensor.return_value = scale_inv1

        # Mock the cross-file safetensors file (file2) for scale2
        mock_file2 = MagicMock()
        mock_file2.get_tensor.return_value = scale_inv2

        # Test 1: Same-file scenario (weight1 + scale1 both in file1)
        read_item1 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex(
                fqn=weight1_fqn,
                offset=torch.Size([0, 0]),
            ),
            dest_index=MetadataIndex(
                fqn=weight1_fqn,
                offset=torch.Size([0, 0]),
            ),
            storage_offsets=[0, 0],
            dest_offsets=[0, 0],
            lengths=[4, 4],
        )

        target_tensor1 = torch.zeros(4, 4, dtype=torch.float32)
        mock_planner1 = MagicMock()
        mock_planner1.resolve_tensor.return_value = target_tensor1

        # Process first weight (same file scenario)
        reader._process_read_request(mock_file1, read_item1, mock_planner1)

        # Verify first tensor was dequantized (ones * 2.0 = twos)
        expected_result1 = torch.ones(4, 4, dtype=torch.float32) * 2.0
        mock_planner1.commit_tensor.assert_called_once()

        # Check that target_tensor1 was updated correctly
        args1, _ = mock_planner1.commit_tensor.call_args
        committed_tensor1 = args1[1]
        torch.testing.assert_close(committed_tensor1, expected_result1)

        # Test 2: Cross-file scenario (weight2 in file1, scale2 in file2)
        read_item2 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex(
                fqn=weight2_fqn,
                offset=torch.Size([0, 0]),
            ),
            dest_index=MetadataIndex(
                fqn=weight2_fqn,
                offset=torch.Size([0, 0]),
            ),
            storage_offsets=[0, 0],
            dest_offsets=[0, 0],
            lengths=[4, 4],
        )

        target_tensor2 = torch.zeros(4, 4, dtype=torch.float32)
        mock_planner2 = MagicMock()
        mock_planner2.resolve_tensor.return_value = target_tensor2

        # Mock the entire safetensors module since it may not be available in test environment
        mock_safetensors = MagicMock()
        mock_safe_open = MagicMock()
        mock_safetensors.safe_open = mock_safe_open

        # Set up the mock to return a context manager that yields mock_file2
        mock_safe_open.return_value.__enter__.return_value = mock_file2
        mock_safe_open.return_value.__exit__.return_value = False

        # Mock the module import and safe_open function
        with patch.dict("sys.modules", {"safetensors": mock_safetensors}):
            # Process second weight (cross-file scenario)
            reader._process_read_request(mock_file1, read_item2, mock_planner2)

            # Verify safe_open was called with the correct file path
            expected_path = f"{self.path}/{file2_name}"
            mock_safe_open.assert_called_once()
            call_args = mock_safe_open.call_args[0]
            self.assertEqual(str(call_args[0]), expected_path)

        # Verify the scale tensor was loaded from the correct file
        mock_file2.get_tensor.assert_called_once_with(scale2_fqn)

        # Verify second tensor was dequantized (3.0 * 0.5 = 1.5)
        expected_result2 = torch.ones(4, 4, dtype=torch.float32) * 3.0 * 0.5  # 1.5
        mock_planner2.commit_tensor.assert_called_once()

        # Check that target_tensor2 was updated correctly
        args2, _ = mock_planner2.commit_tensor.call_args
        committed_tensor2 = args2[1]
        torch.testing.assert_close(committed_tensor2, expected_result2)


if __name__ == "__main__":
    run_tests()
