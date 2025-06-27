# Owner(s): ["module: inductor"]
import json
import os
import tempfile
from unittest.mock import patch

import yaml

import torch
import torch._inductor.lookup_table
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase


class MockDevice:
    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index


class MockInputNode:
    def __init__(
        self,
        device_type="cuda",
        device_index=0,
        dtype=torch.float32,
        size_hint=None,
        stride_hint=None,
    ):
        self.device_type = device_type
        self.device_index = device_index
        self.dtype = dtype
        self.size_hint = size_hint or [128, 128]
        self.stride_hint = stride_hint or [128, 1]

    def get_device(self):
        return MockDevice(self.device_type, self.device_index)

    def get_dtype(self):
        return self.dtype


class TestLookupTable(TestCase):
    def setUp(self):
        super().setUp()
        # Reset the global lookup table before each test
        self.original_table = None
        self.addCleanup(self._cleanup_patches)

    def _cleanup_patches(self):
        # Reset any patches after each test
        pass

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table")
    @patch("torch._inductor.lookup_table._lookup_key")
    @patch("torch.cuda.get_device_name")
    def test_device_name_mismatch(
        self, mock_get_device_name, mock_lookup_key, mock_lookup_table
    ):
        """Test when device name doesn't match any entry in lookup table"""
        # Setup
        mock_get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_lookup_key.return_value = "test_lookup_key"

        # Lookup table has different device name (NVIDIA H100)
        mock_lookup_table = {
            "mm": {"NVIDIA H100": {"test_lookup_key": "(128, 128, 64, 2, 2)"}}
        }

        input_nodes = [
            MockInputNode(device_type="cuda", device_index=0),
            MockInputNode(device_type="cuda", device_index=0),
        ]

        with patch(
            "torch._inductor.lookup_table.gemm_config_lookup_table", mock_lookup_table
        ):
            result = torch._inductor.lookup_table.get_lookup_table(input_nodes, "mm")

        # Should return empty dict when device name doesn't match
        self.assertEqual(result, {})
        mock_get_device_name.assert_called_once_with(0)
        mock_lookup_key.assert_called_once_with(input_nodes)

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table")
    @patch("torch._inductor.lookup_table._lookup_key")
    @patch("torch.cuda.get_device_name")
    def test_lookup_key_mismatch(
        self, mock_get_device_name, mock_lookup_key, mock_lookup_table
    ):
        """Test when lookup key doesn't match any entry in lookup table"""
        # Setup
        mock_get_device_name.return_value = "NVIDIA H100"
        mock_lookup_key.return_value = "non_existent_lookup_key"

        # Lookup table has different lookup key
        mock_lookup_table = {
            "mm": {"NVIDIA H100": {"different_lookup_key": "(128, 128, 64, 2, 2)"}}
        }

        input_nodes = [
            MockInputNode(device_type="cuda", device_index=0),
            MockInputNode(device_type="cuda", device_index=0),
        ]

        with patch(
            "torch._inductor.lookup_table.gemm_config_lookup_table", mock_lookup_table
        ):
            result = torch._inductor.lookup_table.get_lookup_table(input_nodes, "mm")

        # Should return empty dict when lookup key doesn't match
        self.assertEqual(result, {})
        mock_get_device_name.assert_called_once_with(0)
        mock_lookup_key.assert_called_once_with(input_nodes)

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table")
    @patch("torch._inductor.lookup_table._lookup_key")
    def test_device_type_mismatch(self, mock_lookup_key, mock_lookup_table):
        """Test when device type is CPU instead of CUDA"""
        # Setup
        mock_lookup_key.return_value = "test_lookup_key"

        # Lookup table has entry for CPU
        mock_lookup_table = {"mm": {"cpu": {"test_lookup_key": "(128, 128, 64, 2, 2)"}}}

        input_nodes = [
            MockInputNode(device_type="cpu", device_index=0),
            MockInputNode(device_type="cpu", device_index=0),
        ]

        with patch(
            "torch._inductor.lookup_table.gemm_config_lookup_table", mock_lookup_table
        ):
            result = torch._inductor.lookup_table.get_lookup_table(input_nodes, "mm")

        # Should return the lookup string for CPU device
        expected_result = "(128, 128, 64, 2, 2)"
        self.assertEqual(result, expected_result)
        mock_lookup_key.assert_called_once_with(input_nodes)

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table")
    @patch("torch._inductor.lookup_table._lookup_key")
    @patch("torch.cuda.get_device_name")
    def test_lookup_dict_hit(
        self, mock_get_device_name, mock_lookup_key, mock_lookup_table
    ):
        """Test successful lookup that returns a lookup string"""
        # Setup
        mock_get_device_name.return_value = "NVIDIA H100"
        mock_lookup_key.return_value = "test_lookup_key"

        # Lookup table with matching entries
        expected_lookup_result = "(128, 128, 64, 2, 2)"
        mock_lookup_table = {
            "mm": {"NVIDIA H100": {"test_lookup_key": expected_lookup_result}}
        }

        input_nodes = [
            MockInputNode(device_type="cuda", device_index=0),
            MockInputNode(device_type="cuda", device_index=0),
        ]

        with patch(
            "torch._inductor.lookup_table.gemm_config_lookup_table", mock_lookup_table
        ):
            result = torch._inductor.lookup_table.get_lookup_table(input_nodes, "mm")

        # Should return the exact lookup string
        self.assertEqual(result, expected_lookup_result)
        mock_get_device_name.assert_called_once_with(0)
        mock_lookup_key.assert_called_once_with(input_nodes)

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table", None)
    @patch("torch._inductor.lookup_table._lookup_key")
    def test_lookup_table_none(self, mock_lookup_key):
        """Test when gemm_config_lookup_table is None"""
        # Setup
        mock_lookup_key.return_value = "test_lookup_key"

        input_nodes = [
            MockInputNode(device_type="cuda", device_index=0),
            MockInputNode(device_type="cuda", device_index=0),
        ]

        result = torch._inductor.lookup_table.get_lookup_table(input_nodes, "mm")

        # Should return None when lookup table is None
        self.assertIsNone(result)
        # Should not call lookup_key when lookup table is None
        mock_lookup_key.assert_not_called()

    @patch("torch._inductor.lookup_table.gemm_config_lookup_table")
    @patch("torch._inductor.lookup_table._lookup_key")
    @patch("torch.cuda.get_device_name")
    def test_method_mismatch(
        self, mock_get_device_name, mock_lookup_key, mock_lookup_table
    ):
        """Test when method doesn't match any entry in lookup table"""
        # Setup
        mock_get_device_name.return_value = "NVIDIA H100"
        mock_lookup_key.return_value = "test_lookup_key"

        # Lookup table has different method
        mock_lookup_table = {
            "bmm": {  # Different method
                "NVIDIA H100": {"test_lookup_key": "(128, 128, 64, 2, 2)"}
            }
        }

        input_nodes = [
            MockInputNode(device_type="cuda", device_index=0),
            MockInputNode(device_type="cuda", device_index=0),
        ]

        with patch(
            "torch._inductor.lookup_table.gemm_config_lookup_table", mock_lookup_table
        ):
            result = torch._inductor.lookup_table.get_lookup_table(
                input_nodes, "mm"
            )  # Using "mm" method

        # Should return empty dict when method doesn't match
        self.assertEqual(result, {})
        mock_get_device_name.assert_called_once_with(0)
        mock_lookup_key.assert_called_once_with(input_nodes)

    def test_lookup_key_generation(self):
        """Test the _lookup_key function with mock input nodes"""
        # Create mock input nodes with specific properties
        input_nodes = [
            MockInputNode(
                dtype=torch.float32, size_hint=[128, 64], stride_hint=[64, 1]
            ),
            MockInputNode(dtype=torch.float16, size_hint=[64, 32], stride_hint=[32, 1]),
        ]

        # Mock the get_size_hint and get_stride_hint functions
        with (
            patch("torch._inductor.lookup_table.get_size_hint") as mock_size_hint,
            patch("torch._inductor.lookup_table.get_stride_hint") as mock_stride_hint,
        ):
            mock_size_hint.side_effect = lambda node: node.size_hint
            mock_stride_hint.side_effect = lambda node: node.stride_hint

            result = torch._inductor.lookup_table._lookup_key(input_nodes)

            # Verify the result is a string representation of the expected tuple
            expected = str(
                tuple(
                    [
                        (torch.float32, [128, 64], [64, 1]),
                        (torch.float16, [64, 32], [32, 1]),
                    ]
                )
            )
            self.assertEqual(result, expected)


class TestLookupTableFileParsing(TestCase):
    """Test class for file parsing logic in _get_lookup_table()"""

    def setUp(self):
        super().setUp()
        # Store original values to restore later
        self.original_table = torch._inductor.lookup_table.gemm_config_lookup_table
        self.original_path_config = inductor_config.triton.gemm_config_lookup_table_path

        # Reset global state
        torch._inductor.lookup_table.gemm_config_lookup_table = None
        inductor_config.triton.gemm_config_lookup_table_path = None

    def tearDown(self):
        # Restore original values
        torch._inductor.lookup_table.gemm_config_lookup_table = self.original_table
        inductor_config.triton.gemm_config_lookup_table_path = self.original_path_config
        super().tearDown()

    def test_json_file_parsing(self):
        """Test that JSON files are parsed correctly"""
        test_data = {
            "mm": {"NVIDIA H100": {"test_key": {"triton": "(128, 128, 64, 2, 2)"}}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.gemm_config_lookup_table_path = temp_file_path

            # Call the function
            result = torch._inductor.lookup_table._get_lookup_table()

            # Verify the result
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file_path)

    def test_yaml_file_parsing(self):
        """Test that YAML files are parsed correctly"""
        test_data = {
            "addmm": {"NVIDIA H100": {"test_key": {"triton": "(64, 64, 32, 2, 2)"}}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.gemm_config_lookup_table_path = temp_file_path

            # Call the function
            result = torch._inductor.lookup_table._get_lookup_table()

            # Verify the result
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file_path)

    def test_yml_file_parsing(self):
        """Test that .yml files are parsed correctly"""
        test_data = {"bmm": {"cpu": {"test_key": {"triton": "(32, 32, 16, 1, 1)"}}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.gemm_config_lookup_table_path = temp_file_path

            # Call the function
            result = torch._inductor.lookup_table._get_lookup_table()

            # Verify the result
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file_path)

    def test_unsupported_file_extension(self):
        """Test that unsupported file extensions throw an AssertionError"""
        test_data = {"test": "data"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fakesuffix", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.gemm_config_lookup_table_path = temp_file_path

            # Should raise AssertionError
            with self.assertRaises(AssertionError) as context:
                torch._inductor.lookup_table._get_lookup_table()

            self.assertIn("Unsupported file format", str(context.exception))
            self.assertIn(".fakesuffix", str(context.exception))
        finally:
            os.unlink(temp_file_path)


class TestLookupTableConfigSettings(TestCase):
    """Test class for _in_use() method functionality"""

    def setUp(self):
        super().setUp()
        # Store original values to restore later
        self.original_table = torch._inductor.lookup_table.gemm_config_lookup_table
        self.original_path_config = inductor_config.triton.gemm_config_lookup_table_path

    def tearDown(self):
        # Restore original values
        torch._inductor.lookup_table.gemm_config_lookup_table = self.original_table
        inductor_config.triton.gemm_config_lookup_table_path = self.original_path_config

    def test_in_use_with_path_only(self):
        """Test: path is set -> _in_use() returns True"""
        # Setup: table is None, path is set
        torch._inductor.lookup_table.gemm_config_lookup_table = None
        inductor_config.triton.gemm_config_lookup_table_path = "/some/path/config.json"

        # Should return True when path is set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_table_only(self):
        """Test: table is set -> _in_use() returns True"""
        # Setup: path is None, table is set
        inductor_config.triton.gemm_config_lookup_table_path = None
        torch._inductor.lookup_table.gemm_config_lookup_table = {"test": "data"}

        # Should return True when table is set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_neither(self):
        """Test: neither path nor table set -> _in_use() returns False"""
        # Setup: both path and table are None
        torch._inductor.lookup_table.gemm_config_lookup_table = None
        inductor_config.triton.gemm_config_lookup_table_path = None

        # Should return False when neither is set
        result = torch._inductor.lookup_table._in_use()
        self.assertFalse(result)

    def test_in_use_with_both_table_and_path(self):
        """Test: both table and path set -> _in_use() returns True, table has precedence"""
        # Setup: both path and table are set
        test_table = {"existing": "table_data"}
        torch._inductor.lookup_table.gemm_config_lookup_table = test_table
        inductor_config.triton.gemm_config_lookup_table_path = "/some/path/config.json"

        # Should return True when both are set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

        # Verify table data has precedence and is not overridden by path
        lookup_result = torch._inductor.lookup_table._get_lookup_table()
        self.assertEqual(lookup_result, test_table)


if __name__ == "__main__":
    run_tests()
