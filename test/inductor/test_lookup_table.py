# Owner(s): ["module: inductor"]
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
        self.original_table = torch._inductor.config.template_lookup_table.table
        self.addCleanup(self._cleanup_patches)

    def tearDown(self):
        torch._inductor.config.template_lookup_table.table = self.original_table
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

    def create_triton_config(
        self, block_m=128, block_n=128, block_k=64, num_stages=2, num_warps=2, **kwargs
    ):
        """Create a triton backend configuration as direct dictionary matching mm_options() expectations"""
        # This should match what mm_options() expects to receive after processing
        config = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "EVEN_K": True,  # This gets computed by mm_options()
            "ALLOW_TF32": True,  # This gets computed by mm_options()
            "USE_FAST_ACCUM": False,  # Default from mm_options()
            "ACC_TYPE": "tl.float32",  # This gets computed by mm_options()
            "GROUP_M": 8,  # Default from mm_options()
        }
        config.update(kwargs)
        return config


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
                "torch._inductor.lookup_table._template_lookup_key",
                return_value=lookup_key,
            ) as mock_lookup_key,
            patch.object(
                inductor_config.template_lookup_table, "table", lookup_table_data
            ),
        ):
            result = torch._inductor.lookup_table.get_template_lookup_table(
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
                    table_lookup_key: {"triton": [self.create_triton_config()]}
                }
            }
        }
        self._run_lookup_test(dev_key, lookup_key, method, lookup_table_data, expected)

    def test_lookup_dict_hit(self):
        """Test successful lookup that returns a lookup dict"""
        expected_result = {
            "triton": [self.create_triton_config()],
            "tma": [
                self.create_triton_config(
                    block_m=64, block_k=32, num_stages=3, num_warps=4, ALLOW_TF32=True
                )
            ],
        }
        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": expected_result}}}
        self._run_lookup_test(
            "NVIDIA H100(9, 0)", "test_key", "mm", lookup_table_data, expected_result
        )

    def test_lookup_table_none(self):
        """Test when template lookup table is empty"""
        with patch.object(inductor_config.template_lookup_table, "table", {}):
            result = torch._inductor.lookup_table.get_template_lookup_table(
                self.create_mock_input_nodes(2), "mm"
            )
            self.assertIsNone(result)

    def test_lookup_key_generation(self):
        """Test the _template_lookup_key function with mock input nodes"""
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

            result = torch._inductor.lookup_table._template_lookup_key(input_nodes)
            expected = str(
                (
                    (torch.float32, [128, 64], [64, 1]),
                    (torch.float16, [64, 32], [32, 1]),
                )
            )
            self.assertEqual(result, expected)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestTemplateLookupTableConfig(TestCase):
    """Test class for new template lookup table configuration"""

    def setUp(self):
        super().setUp()
        # Store original values to restore later
        self.original_table = inductor_config.template_lookup_table.table

    def tearDown(self):
        # Restore original values
        inductor_config.template_lookup_table.table = self.original_table

    def test_in_use_with_table_data(self):
        """Test: table has data -> _in_use() returns True"""
        # Setup: table has data
        inductor_config.template_lookup_table.table = {"test": "data"}

        # Should return True when table has data
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_empty_table(self):
        """Test: table is empty -> _in_use() returns False"""
        # Setup: table is empty
        inductor_config.template_lookup_table.table = {}

        # Should return False when table is empty
        result = torch._inductor.lookup_table._in_use()
        self.assertFalse(result)

    def test_get_lookup_table_returns_config_table(self):
        """Test: _get_lookup_table() returns the config table"""
        test_table = {"device": {"op": {"key": {"template": "options"}}}}
        inductor_config.template_lookup_table.table = test_table

        result = torch._inductor.lookup_table._get_lookup_table()
        self.assertEqual(result, test_table)


@instantiate_parametrized_tests
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestLookupTemplateDict(TestCase):
    """Test class for lookup_template_dict function with new direct dictionary system"""

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
            result = lookup_template_dict({"triton": {"BLOCK_M": 128}}, "triton")
            self.assertIsNone(result)

    def test_lookup_template_dict_none_lookup_dict(self):
        """Test lookup_template_dict with None lookup_dict"""
        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(None, "triton")
            self.assertIsNone(result)

    def test_lookup_template_dict_key_not_found(self):
        """Test lookup_template_dict when key is not in lookup_dict"""
        lookup_dict = {
            "tma": {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 3,
                "num_warps": 4,
            }
        }

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")  # Key not in dict
            self.assertIsNone(result)

    def test_lookup_template_dict_valid_config(self):
        """Test lookup_template_dict with valid direct dictionary config"""
        config_data = [
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 2,
                "num_warps": 4,
                "ALLOW_TF32": True,
            }
        ]
        lookup_dict = {"triton": config_data}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_multiple_templates(self):
        """Test lookup_template_dict with multiple template configurations"""
        triton_config = [
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 2,
                "num_warps": 4,
                "ALLOW_TF32": True,
            }
        ]
        tma_config = [
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 3,
                "num_warps": 8,
                "ALLOW_TF32": True,
            }
        ]
        lookup_dict = {
            "triton": triton_config,
            "tma": tma_config,
        }

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            # Test triton backend
            result_triton = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result_triton, triton_config)

            # Test tma backend
            result_tma = lookup_template_dict(lookup_dict, "tma")
            self.assertEqual(result_tma, tma_config)

    def test_lookup_template_dict_empty_config(self):
        """Test lookup_template_dict with empty config dictionary"""
        config_data = {}  # Empty config for bias_addmm case
        lookup_dict = {"bias_addmm": config_data}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "bias_addmm")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_decompose_k_config(self):
        """Test lookup_template_dict with decompose_k config (single k value)"""
        config_data = [{"k": 4}]  # decompose_k format as list
        lookup_dict = {"decompose_k": config_data}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "decompose_k")
            self.assertEqual(result, config_data)

    def test_lookup_template_dict_multiple_configs_same_template(self):
        """Test lookup_template_dict with multiple configurations for same template"""
        config1 = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_stages": 2,
            "num_warps": 4,
            "ALLOW_TF32": True,
        }
        config2 = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_stages": 3,
            "num_warps": 8,
            "ALLOW_TF32": True,
        }
        config_list = [config1, config2]
        lookup_dict = {"triton": config_list}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, config_list)
            self.assertEqual(len(result), 2)
            self.assertIn(config1, result)
            self.assertIn(config2, result)

    def test_lookup_template_dict_mixed_single_and_multiple_configs(self):
        """Test lookup_template_dict with mixed scenarios - some templates with single config, others with multiple"""
        triton_configs = [
            {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8},
            {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 16},
        ]
        tma_configs = [
            {"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 8},
        ]
        lookup_dict = {
            "triton": triton_configs,
            "tma": tma_configs,
        }

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            # Test triton backend with multiple configs
            result_triton = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result_triton, triton_configs)
            self.assertEqual(len(result_triton), 3)

            # Test tma backend with single config
            result_tma = lookup_template_dict(lookup_dict, "tma")
            self.assertEqual(result_tma, tma_configs)
            self.assertEqual(len(result_tma), 1)

    def test_lookup_template_dict_empty_config_list(self):
        """Test lookup_template_dict with empty config list"""
        lookup_dict = {"triton": []}

        # Mock _in_use to return True
        with patch("torch._inductor.lookup_table._in_use", return_value=True):
            result = lookup_template_dict(lookup_dict, "triton")
            self.assertEqual(result, [])
            self.assertEqual(len(result), 0)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_CUDA:
        run_tests()
