# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch
import torch._inductor.lookup_table
from torch._inductor import config as inductor_config
from torch._inductor.lookup_table import lookup_key_suffix, lookup_template_configs
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class MockInputNode:
    """Simplified mock input node for testing"""

    def __init__(self, device_type="cuda", dtype=torch.float32, size_hint=None):
        self.device_type = device_type
        self.dtype = dtype
        self.size_hint = size_hint or [128, 128]
        self.stride_hint = [128, 1]

    def get_device(self):
        class Device:
            def __init__(self, device_type, index=0):
                self.type = device_type
                self.index = index

        return Device(self.device_type)

    def get_dtype(self):
        return self.dtype

    def get_size(self):
        return self.size_hint

    def get_stride(self):
        return self.stride_hint


class BaseLookupTableTest(TestCase):
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        super().setUp()
        self.original_table = torch._inductor.config.template_lookup_table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        inductor_config.max_autotune = True

    def tearDown(self):
        torch._inductor.config.template_lookup_table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        super().tearDown()

    def create_mock_input_nodes(self, count=2, **kwargs):
        """Create mock input nodes with default or custom properties"""
        return [MockInputNode(**kwargs) for _ in range(count)]

    def create_lookup_key(self, device_key, method, lookup_key):
        """Create a lookup key"""
        flat_key = f"{device_key}+{method}+{lookup_key}+{lookup_key_suffix()}"
        return flat_key

    def create_config(self, template_id, **kwargs):
        """Create a backend configuration with template_id field"""
        config = {"template_id": template_id}

        # Add minimal defaults based on template type
        if template_id == "triton":
            config.update(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "tma":
            config.update(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 4,
                    "num_warps": 8,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "decompose_k":
            config.update({"k": 4})

        config.update(kwargs)
        return config


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTable(BaseLookupTableTest):
    """Consolidated tests for lookup table functionality"""

    def test_lookup_mismatch(self):
        """Test mismatch scenario in lookup table"""
        # Test device key mismatch
        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): [
                self.create_config("triton")
            ]
        }

        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch(
                "torch._inductor.lookup_table._dev_key", return_value="NVIDIA RTX 3080"
            ),
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            result = lookup_template_configs(input_nodes, "mm", "triton")
            self.assertEqual(result, [])

    def test_successful_lookup_with_template_filtering(self):
        """Test successful lookup that filters configs by template_id"""
        config_list = [
            self.create_config("triton", BLOCK_M=128, BLOCK_N=128),
            self.create_config("triton", BLOCK_M=64, BLOCK_N=64),
            self.create_config("tma", BLOCK_M=256, BLOCK_N=128),
            self.create_config("decompose_k", k=4),
        ]

        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): config_list
        }

        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            # Test triton template filtering
            result = lookup_template_configs(input_nodes, "mm", "triton")
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result), 2)
            for config in result:
                self.assertNotIn("template_id", config)
                self.assertIn("BLOCK_M", config)

            # Test tma template filtering
            result = lookup_template_configs(input_nodes, "mm", "tma")
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result), 1)
            self.assertNotIn("template_id", result[0])
            self.assertEqual(result[0]["BLOCK_M"], 256)

            # Test decompose_k template filtering
            result = lookup_template_configs(input_nodes, "mm", "decompose_k")
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result), 1)
            self.assertNotIn("template_id", result[0])
            self.assertEqual(result[0]["k"], 4)

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        with patch.object(inductor_config, "template_lookup_table", {}):
            result = lookup_template_configs(
                self.create_mock_input_nodes(2), "mm", "triton"
            )
            self.assertEqual(result, [])

    def test_validation_error(self):
        """Test validation error for invalid config"""
        invalid_config = {"BLOCK_M": 128}  # missing template_id
        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): [invalid_config]
        }

        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            with self.assertRaises(ValueError) as cm:
                lookup_template_configs(input_nodes, "mm", "triton")
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_cpu_input_returns_none(self):
        """Test that CPU tensor input returns None"""
        cpu_input_nodes = [MockInputNode(device_type="cpu")]

        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): [
                self.create_config("triton")
            ]
        }

        with (
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            result = lookup_template_configs(cpu_input_nodes, "mm", "triton")
            self.assertIsNone(result)

    def test_in_use_functionality(self):
        """Test _in_use() function behavior"""
        # Test with data
        inductor_config.template_lookup_table = {"test": "data"}
        inductor_config.max_autotune = True
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

        # Test with empty table
        inductor_config.template_lookup_table = {}
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

        # Test error when max_autotune disabled
        inductor_config.max_autotune = False
        with self.assertRaises(RuntimeError) as cm:
            torch._inductor.lookup_table._in_use()
        self.assertIn("template lookup table requires max-autotune", str(cm.exception))

    @parametrize(
        "allow_tf32,tf32_configs,expected_count",
        [
            (True, [True, False], 2),  # No filtering when allowed
            (False, [True, False], 1),  # Filter TF32=True when not allowed
            (False, [True, True], 0),  # Filter all when all TF32=True
        ],
    )
    def test_tf32_filtering(self, allow_tf32, tf32_configs, expected_count):
        """Test TF32 filtering scenarios"""
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        configs = [
            self.create_config("triton", BLOCK_M=128, ALLOW_TF32=tf32_val)
            for tf32_val in tf32_configs
        ]

        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): configs
        }

        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            result = lookup_template_configs(input_nodes, "mm", "triton")
            if expected_count > 0:
                assert result is not None, "Result should not be None"
                self.assertEqual(len(result), expected_count)
            else:
                self.assertEqual(result, [])

    def test_multiple_calls_work(self):
        """Test that calling lookup functions multiple times works correctly"""
        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
        ]
        lookup_table_data = {
            self.create_lookup_key("NVIDIA H100", "mm", "test_key"): config_list
        }

        input_nodes = self.create_mock_input_nodes(2)

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._inputs_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            # First calls
            result1 = lookup_template_configs(input_nodes, "mm", "triton")
            result2 = lookup_template_configs(input_nodes, "mm", "tma")
            assert result1 is not None, "Result1 should not be None"
            assert result2 is not None, "Result2 should not be None"
            self.assertEqual(len(result1), 1)
            self.assertEqual(len(result2), 1)

            # Second calls should work the same
            result3 = lookup_template_configs(input_nodes, "mm", "triton")
            result4 = lookup_template_configs(input_nodes, "mm", "tma")
            assert result3 is not None, "Result3 should not be None"
            assert result4 is not None, "Result4 should not be None"
            self.assertEqual(len(result3), 1)
            self.assertEqual(len(result4), 1)


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
