# Owner(s): ["module: inductor"]
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch._inductor.lookup_table
from torch._inductor import config as inductor_config
from torch._inductor.lookup_table import lookup_template_dict
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


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


class BaseLookupTableTest(TestCase):
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        super().setUp()
        self.original_table = torch._inductor.lookup_table.kernel_config_lookup_table
        self.original_path_config = (
            inductor_config.triton.kernel_config_lookup_table_path
        )
        self.addCleanup(self._cleanup_patches)

    def tearDown(self):
        torch._inductor.lookup_table.kernel_config_lookup_table = self.original_table
        inductor_config.triton.kernel_config_lookup_table_path = (
            self.original_path_config
        )
        super().tearDown()

    def _cleanup_patches(self):
        pass

    def create_mock_input_nodes(self, count=2, **kwargs):
        """Create mock input nodes with default or custom properties"""
        return [MockInputNode(**kwargs) for _ in range(count)]

    def create_lookup_table_config(
        self, device_key, method, lookup_key, backend_configs
    ):
        """Create a lookup table configuration"""
        return {device_key: {method: {lookup_key: backend_configs}}}

    def create_triton_config(self, config_list=None, kwargs_dict=None):
        """Create a triton backend configuration"""
        config_list = config_list or [128, 128, 64, 2, 2]
        kwargs_dict = kwargs_dict or {"EVEN_K": True}
        return json.dumps({"config": config_list, "kwargs": kwargs_dict})


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTable(BaseLookupTableTest):
    def _run_lookup_test(
        self,
        dev_key,
        lookup_key,
        method,
        lookup_table_data,
        expected_result,
        should_call_lookup_key=True,
    ):
        """Helper method to run lookup table tests with common patterns"""
        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value=dev_key),
            patch(
                "torch._inductor.lookup_table._gemm_lookup_key", return_value=lookup_key
            ) as mock_lookup_key,
            patch(
                "torch._inductor.lookup_table.kernel_config_lookup_table",
                lookup_table_data,
            ),
        ):
            result = torch._inductor.lookup_table.get_gemm_lookup_table(
                input_nodes, method
            )

            self.assertEqual(result, expected_result)
            if should_call_lookup_key:
                mock_lookup_key.assert_called_once_with(input_nodes)
            else:
                mock_lookup_key.assert_not_called()

    @parametrize(
        "dev_key,lookup_key,method,table_dev_key,table_method,table_lookup_key,expected",
        [
            (
                "NVIDIA RTX 3080(8, 6)",
                "test_key",
                "mm",
                "NVIDIA H100(9, 0)",
                "mm",
                "test_key",
                {},
            ),
            (
                "NVIDIA H100(9, 0)",
                "non_existent_key",
                "mm",
                "NVIDIA H100(9, 0)",
                "mm",
                "different_key",
                {},
            ),
            (
                "NVIDIA H100(9, 0)",
                "test_key",
                "mm",
                "NVIDIA H100(9, 0)",
                "bmm",
                "test_key",
                {},
            ),
        ],
    )
    def test_lookup_mismatches(
        self,
        dev_key,
        lookup_key,
        method,
        table_dev_key,
        table_method,
        table_lookup_key,
        expected,
    ):
        """Test various mismatch scenarios in lookup table"""
        lookup_table_data = {
            table_dev_key: {
                table_method: {
                    table_lookup_key: {"triton": self.create_triton_config()}
                }
            }
        }
        self._run_lookup_test(dev_key, lookup_key, method, lookup_table_data, expected)

    def test_device_type_cpu(self):
        """Test when device type is CPU"""
        expected_result = {"triton": self.create_triton_config()}
        lookup_table_data = {"cpu": {"mm": {"test_key": expected_result}}}
        self._run_lookup_test(
            "cpu", "test_key", "mm", lookup_table_data, expected_result
        )

    def test_lookup_dict_hit(self):
        """Test successful lookup that returns a lookup dict"""
        expected_result = {
            "triton": self.create_triton_config(),
            "tma": json.dumps(
                {"config": [64, 64, 32, 3, 4], "kwargs": {"EVEN_K": True}}
            ),
        }
        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": expected_result}}}
        self._run_lookup_test(
            "NVIDIA H100(9, 0)", "test_key", "mm", lookup_table_data, expected_result
        )

    def test_lookup_table_none(self):
        """Test when kernel_config_lookup_table is None"""
        with patch("torch._inductor.lookup_table.kernel_config_lookup_table", None):
            result = torch._inductor.lookup_table.get_gemm_lookup_table(
                self.create_mock_input_nodes(2), "mm"
            )
            self.assertIsNone(result)

    def test_lookup_key_generation(self):
        """Test the _lookup_key function with mock input nodes"""
        input_nodes = [
            MockInputNode(
                dtype=torch.float32, size_hint=[128, 64], stride_hint=[64, 1]
            ),
            MockInputNode(dtype=torch.float16, size_hint=[64, 32], stride_hint=[32, 1]),
        ]

        with (
            patch("torch._inductor.lookup_table.get_size_hint") as mock_size_hint,
            patch("torch._inductor.lookup_table.get_stride_hint") as mock_stride_hint,
        ):
            mock_size_hint.side_effect = lambda node: node.size_hint
            mock_stride_hint.side_effect = lambda node: node.stride_hint

            result = torch._inductor.lookup_table._gemm_lookup_key(input_nodes)
            expected = str(
                (
                    (torch.float32, [128, 64], [64, 1]),
                    (torch.float16, [64, 32], [32, 1]),
                )
            )
            self.assertEqual(result, expected)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestLookupTableFileParsing(TestCase):
    """Test class for file parsing logic in _get_lookup_table()"""

    def setUp(self):
        super().setUp()
        # Store original values to restore later
        self.original_table = torch._inductor.lookup_table.kernel_config_lookup_table
        self.original_path_config = (
            inductor_config.triton.kernel_config_lookup_table_path
        )

        # Reset global state
        torch._inductor.lookup_table.kernel_config_lookup_table = None
        inductor_config.triton.kernel_config_lookup_table_path = None

    def tearDown(self):
        # Restore original values
        torch._inductor.lookup_table.kernel_config_lookup_table = self.original_table
        inductor_config.triton.kernel_config_lookup_table_path = (
            self.original_path_config
        )
        super().tearDown()

    def test_json_file_parsing(self):
        """Test that JSON files are parsed correctly"""
        test_data = {
            "mm": {
                "NVIDIA H100": {
                    "test_key": {
                        "triton": json.dumps(
                            {"config": [128, 128, 64, 2, 2], "kwargs": {"EVEN_K": True}}
                        ),
                        "tma": json.dumps(
                            {"config": [64, 64, 32, 3, 4], "kwargs": {"EVEN_K": True}}
                        ),
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.kernel_config_lookup_table_path = temp_file_path

            # Call the function
            result = torch._inductor.lookup_table._get_lookup_table()

            # Verify the result
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file_path)

    def test_yaml_file_parsing(self):
        import yaml

        """Test that YAML files are parsed correctly"""
        test_data = {
            "addmm": {
                "NVIDIA H100": {
                    "test_key": {
                        "triton": json.dumps(
                            {"config": [64, 64, 32, 2, 2], "kwargs": {"EVEN_K": True}}
                        ),
                        "bias_addmm": json.dumps(
                            {"config": [32, 32, 16, 1, 2], "kwargs": {"EVEN_K": True}}
                        ),
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.kernel_config_lookup_table_path = temp_file_path

            # Call the function
            result = torch._inductor.lookup_table._get_lookup_table()

            # Verify the result
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_file_path)

    def test_yml_file_parsing(self):
        import yaml

        """Test that .yml files are parsed correctly"""
        test_data = {
            "bmm": {
                "cpu": {
                    "test_key": {
                        "triton": json.dumps(
                            {"config": [32, 32, 16, 1, 1], "kwargs": {"EVEN_K": True}}
                        )
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_file_path = f.name

        try:
            # Configure to use the temp file
            inductor_config.triton.kernel_config_lookup_table_path = temp_file_path

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
            inductor_config.triton.kernel_config_lookup_table_path = temp_file_path

            # Should raise AssertionError
            with self.assertRaises(AssertionError) as context:
                torch._inductor.lookup_table._get_lookup_table()

            self.assertIn("Unsupported file format", str(context.exception))
            self.assertIn(".fakesuffix", str(context.exception))
        finally:
            os.unlink(temp_file_path)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestLookupTableConfigSettings(TestCase):
    """Test class for _in_use() method functionality"""

    def setUp(self):
        super().setUp()
        # Store original values to restore later
        self.original_table = torch._inductor.lookup_table.kernel_config_lookup_table
        self.original_path_config = (
            inductor_config.triton.kernel_config_lookup_table_path
        )

    def tearDown(self):
        # Restore original values
        torch._inductor.lookup_table.kernel_config_lookup_table = self.original_table
        inductor_config.triton.kernel_config_lookup_table_path = (
            self.original_path_config
        )

    def test_in_use_with_path_only(self):
        """Test: path is set -> _in_use() returns True"""
        # Setup: table is None, path is set
        torch._inductor.lookup_table.kernel_config_lookup_table = None
        inductor_config.triton.kernel_config_lookup_table_path = (
            "/some/path/config.json"
        )

        # Should return True when path is set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_table_only(self):
        """Test: table is set -> _in_use() returns True"""
        # Setup: path is None, table is set
        inductor_config.triton.kernel_config_lookup_table_path = None
        torch._inductor.lookup_table.kernel_config_lookup_table = {"test": "data"}

        # Should return True when table is set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_neither(self):
        """Test: neither path nor table set -> _in_use() returns False"""
        # Setup: both path and table are None
        torch._inductor.lookup_table.kernel_config_lookup_table = None
        inductor_config.triton.kernel_config_lookup_table_path = None

        # Should return False when neither is set
        result = torch._inductor.lookup_table._in_use()
        self.assertFalse(result)

    def test_in_use_with_both_table_and_path(self):
        """Test: both table and path set -> _in_use() returns True, table has precedence"""
        # Setup: both path and table are set
        test_table = {"existing": "table_data"}
        torch._inductor.lookup_table.kernel_config_lookup_table = test_table
        inductor_config.triton.kernel_config_lookup_table_path = (
            "/some/path/config.json"
        )

        # Should return True when both are set
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

        # Verify table data has precedence and is not overridden by path
        lookup_result = torch._inductor.lookup_table._get_lookup_table()
        self.assertEqual(lookup_result, test_table)


@instantiate_parametrized_tests
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestLookupTemplateDict(TestCase):
    """Test class for lookup_template_dict function"""

    def setUp(self):
        super().setUp()
        # Store original _in_use function to restore later
        self.original_in_use = torch._inductor.lookup_table._in_use

    def tearDown(self):
        # Restore original _in_use function
        torch._inductor.lookup_table._in_use = self.original_in_use
        super().tearDown()

    def test_lookup_template_dict_not_in_use(self):
        """Test lookup_template_dict when lookup table is not in use"""
        # Mock _in_use to return False
        with patch("torch._inductor.lookup_table._in_use", return_value=False):
            result = lookup_template_dict({"triton": "some_data"}, "triton")
            self.assertIsNone(result)

    def test_lookup_template_dict_none_lookup_dict(self):
        """Test lookup_template_dict with None lookup_dict"""
        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(None, "triton")
            expected = {"config": [], "kwargs": {}}
            self.assertEqual(result, expected)

    def test_lookup_template_dict_key_not_found(self):
        """Test lookup_template_dict when key is not in lookup_dict"""
        lookup_dict = {
            "tma": json.dumps(
                {"config": [64, 64, 32, 3, 4], "kwargs": {"EVEN_K": True}}
            )
        }

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")  # Key not in dict
            expected = {"config": [], "kwargs": {}}
            self.assertEqual(result, expected)

    def test_lookup_template_dict_valid_json_config(self):
        """Test lookup_template_dict with valid JSON config"""
        config_data = {"config": [128, 128, 64, 2, 4], "kwargs": {"EVEN_K": True}}
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_valid_json_with_6_params(self):
        """Test lookup_template_dict with valid JSON config containing 6 parameters"""
        config_data = {
            "config": [64, 64, 32, 2, 4, 8],  # 6 parameters including group_m
            "kwargs": {"EVEN_K": True, "USE_FAST_ACCUM": False},
        }
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_empty_config(self):
        """Test lookup_template_dict with empty config list"""
        config_data = {"config": [], "kwargs": {"EVEN_K": True}}  # Empty config
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_empty_kwargs(self):
        """Test lookup_template_dict with empty kwargs"""
        config_data = {"config": [32, 32, 16, 1, 2], "kwargs": {}}  # Empty kwargs
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    @parametrize(
        "invalid_json",
        [
            "not_json_at_all",
            "{invalid_json}",
            '{"kwargs": {"EVEN_K": true}}',  # Missing config (required)
            '{"config": "not_a_list"}',  # Config not a list
            '{"config": [], "kwargs": "not_a_dict"}',  # Kwargs not a dict
            "[]",  # Not a dictionary
            '"just_a_string"',  # Not a dictionary
        ],
    )
    def test_lookup_template_dict_invalid_json(self, invalid_json):
        """Test lookup_template_dict with invalid JSON that should raise ValueError"""
        lookup_dict = {"triton": invalid_json}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            with self.assertRaises(ValueError) as cm:
                lookup_template_dict(lookup_dict, "triton")

            # Verify the error message contains expected information
            error_message = str(cm.exception)
            self.assertIn("Invalid JSON structure for lookup table", error_message)

    def test_lookup_template_dict_multiple_backends(self):
        """Test lookup_template_dict with multiple backend configurations"""
        triton_config = {"config": [128, 128, 64, 2, 4], "kwargs": {"EVEN_K": True}}
        tma_config = {
            "config": [64, 64, 32, 3, 8],
            "kwargs": {"EVEN_K": True, "USE_TMA": True},
        }
        lookup_dict = {
            "triton": json.dumps(triton_config),
            "tma": json.dumps(tma_config),
        }

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            # Test triton backend
            result_triton = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result_triton, triton_config)

            # Test tma backend
            result_tma = lookup_template_dict(lookup_dict, "tma")
            self.assertEqual(result_tma, tma_config)

    def test_lookup_template_dict_complex_kwargs(self):
        """Test lookup_template_dict with complex kwargs containing various types"""
        config_data = {
            "config": [64, 64, 32, 2, 4],
            "kwargs": {
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ALLOW_TF32": True,
            },
        }
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_missing_kwargs(self):
        """Test lookup_template_dict with missing kwargs (should be allowed and auto-added)"""
        config_data = {"config": [128, 128, 64, 2, 4]}  # No kwargs key
        lookup_dict = {"triton": json.dumps(config_data)}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            expected = {
                "config": [128, 128, 64, 2, 4],
                "kwargs": {},
            }  # kwargs should be auto-added
            self.assertEqual(result, expected)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_CUDA:
        run_tests()
