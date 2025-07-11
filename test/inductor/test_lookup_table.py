# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch
import torch._inductor.lookup_table
from torch._inductor import config as inductor_config
from torch._inductor.lookup_table import (
    lookup_op_config_entries,
    lookup_template_configs_from_op,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


class MockDevice:
    """Simplified mock device for testing"""

    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index


class MockInputNode:
    """Simplified mock input node for testing"""

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
        # Store all original values for restoration
        self.original_table = torch._inductor.config.template_lookup_table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        self.original_max_autotune_gemm = getattr(
            inductor_config, "max_autotune_gemm", False
        )
        self.original_allow_tf32 = getattr(
            torch.backends.cuda.matmul, "allow_tf32", True
        )

    def tearDown(self):
        # Restore all original values
        torch._inductor.config.template_lookup_table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        inductor_config.max_autotune_gemm = self.original_max_autotune_gemm
        torch.backends.cuda.matmul.allow_tf32 = self.original_allow_tf32
        super().tearDown()

    def assert_configs_equal_without_template_id(self, result, expected):
        """
        Custom comparator for configs that handles template_id removal.

        Args:
            result: The actual result from lookup_template_configs_from_op
            expected: The expected config(s) with template_id field
        """
        if isinstance(expected, list):
            # Handle list of configs
            self.assertEqual(len(result), len(expected))
            for result_config, expected_config in zip(result, expected):
                self._assert_single_config_equal_without_template_id(
                    result_config, expected_config
                )
        else:
            # Handle single config
            self._assert_single_config_equal_without_template_id(result, expected)

    def _assert_single_config_equal_without_template_id(
        self, result_config, expected_config
    ):
        """Helper method to compare a single config pair"""
        # Assert template_id is not in result
        self.assertNotIn(
            "template_id",
            result_config,
            "Result config should not contain template_id field",
        )

        # Create expected config without template_id for comparison
        expected_without_template_id = expected_config.copy()
        if "template_id" in expected_without_template_id:
            expected_without_template_id.pop("template_id")

        # Compare the configs without template_id
        self.assertEqual(result_config, expected_without_template_id)

    def create_mock_input_nodes(self, count=2, **kwargs):
        """Create mock input nodes with default or custom properties"""
        return [MockInputNode(**kwargs) for _ in range(count)]

    def create_lookup_table_config(
        self, device_key, method, lookup_key, backend_configs
    ):
        """Create a lookup table configuration"""
        return {device_key: {method: {lookup_key: backend_configs}}}

    def create_config(self, template_id, **kwargs):
        """Create a backend configuration with template_id field for any template type"""
        # Define default configurations for each template type
        common = {
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
        }
        template_defaults = {
            "triton": {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 2,
                "num_warps": 2,
                **common,
            },
            "tma": {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 4,
                "num_warps": 8,
                **common,
            },
            "decompose_k": {
                "k": 4,
            },
            "bias_addmm": {},
        }

        # Start with template_id and defaults for the template type
        config = {"template_id": template_id}
        if template_id in template_defaults:
            config.update(template_defaults[template_id])

        # Override with any provided kwargs
        config.update(kwargs)
        return config

    # Convenience methods for backward compatibility
    def create_triton_config(self, **kwargs):
        """Create a triton backend configuration with template_id field"""
        return self.create_config("triton", **kwargs)

    def create_tma_config(self, **kwargs):
        """Create a TMA backend configuration with template_id field"""
        return self.create_config("tma", **kwargs)

    def create_decompose_k_config(self, k=4):
        """Create a decompose_k backend configuration with template_id field"""
        return self.create_config("decompose_k", k=k)

    def create_bias_addmm_config(self):
        """Create a bias_addmm backend configuration with template_id field"""
        return self.create_config("bias_addmm")

    def _run_lookup_test(
        self,
        dev_key,
        lookup_key,
        method,
        lookup_table_data,
        expected_result,
        should_call_lookup_key=True,
        input_nodes=None,
        enable_max_autotune=True,
    ):
        """Helper method to run lookup table tests with common patterns"""
        if input_nodes is None:
            input_nodes = self.create_mock_input_nodes(2)

        if enable_max_autotune:
            inductor_config.max_autotune = True

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value=dev_key),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value=lookup_key,
            ) as mock_lookup_key,
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            result = lookup_op_config_entries(input_nodes, method)

            self.assertEqual(result, expected_result)
            if should_call_lookup_key:
                mock_lookup_key.assert_called_once_with(input_nodes)
            else:
                mock_lookup_key.assert_not_called()

    def _run_template_lookup_test(
        self,
        lookup_dict,
        template_id,
        expected_result,
        enable_max_autotune=True,
        allow_tf32=True,
        should_log_warning=False,
        warning_count=0,
    ):
        """Helper method to run template lookup tests with TF32 filtering"""
        if enable_max_autotune:
            inductor_config.max_autotune = True
            inductor_config.template_lookup_table = {"test": "data"}
        else:
            inductor_config.max_autotune = False
            inductor_config.template_lookup_table = {}

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        if should_log_warning:
            with patch("torch._inductor.lookup_table.log.warning") as mock_warning:
                result = lookup_template_configs_from_op(lookup_dict, template_id)
                self.assertEqual(mock_warning.call_count, warning_count)
        else:
            result = lookup_template_configs_from_op(lookup_dict, template_id)

        if expected_result is None:
            self.assertIsNone(result)
        elif isinstance(expected_result, list):
            if len(expected_result) == 0:
                self.assertEqual(result, [])
            else:
                self.assert_configs_equal_without_template_id(result, expected_result)
        else:
            self.assert_configs_equal_without_template_id(result, expected_result)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableCore(BaseLookupTableTest):
    """Consolidated tests for core lookup table functionality"""

    @parametrize(
        "dev_key,lookup_key,method,table_dev_key,table_method,table_lookup_key,expected",
        [
            (
                "NVIDIA RTX 3080",
                "test_key",
                "mm",
                "NVIDIA H100",
                "mm",
                "test_key",
                {},
            ),
            (
                "NVIDIA H100",
                "non_existent_key",
                "mm",
                "NVIDIA H100",
                "mm",
                "different_key",
                {},
            ),
            (
                "NVIDIA H100",
                "test_key",
                "mm",
                "NVIDIA H100",
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
                table_method: {table_lookup_key: [self.create_triton_config()]}
            }
        }
        self._run_lookup_test(dev_key, lookup_key, method, lookup_table_data, expected)

    def test_successful_lookup_with_grouping(self):
        """Test successful lookup that groups configs by template_id"""
        # Create configs with different template_ids
        config_list = [
            self.create_triton_config(block_m=128, block_n=128),
            self.create_triton_config(block_m=64, block_n=64),
            self.create_tma_config(block_m=256, block_n=128),
            self.create_decompose_k_config(k=4),
        ]

        # Expected result should be grouped by template_id
        expected_result = {
            "triton": [
                self.create_triton_config(block_m=128, block_n=128),
                self.create_triton_config(block_m=64, block_n=64),
            ],
            "tma": [self.create_tma_config(block_m=256, block_n=128)],
            "decompose_k": [self.create_decompose_k_config(k=4)],
        }

        lookup_table_data = {"NVIDIA H100": {"mm": {"test_key": config_list}}}
        self._run_lookup_test(
            "NVIDIA H100", "test_key", "mm", lookup_table_data, expected_result
        )

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        with patch.object(inductor_config, "template_lookup_table", {}):
            result = lookup_op_config_entries(self.create_mock_input_nodes(2), "mm")
            self.assertIsNone(result)

    @parametrize(
        "invalid_config,expected_error_message",
        [
            ({"BLOCK_M": 128, "BLOCK_N": 128}, "missing required 'template_id' field"),
            ("not_a_dict", "is not a dictionary"),
        ],
    )
    def test_validation_errors(self, invalid_config, expected_error_message):
        """Test validation errors for invalid configs"""
        config_list = [invalid_config]
        lookup_table_data = {"NVIDIA H100": {"mm": {"test_key": config_list}}}

        input_nodes = self.create_mock_input_nodes(2)
        inductor_config.max_autotune = True

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            with self.assertRaises(ValueError) as cm:
                torch._inductor.lookup_table._get_op_lookup_table(input_nodes, "mm")
            self.assertIn(expected_error_message, str(cm.exception))

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

    def test_cpu_input_with_h100_lookup_table(self):
        """Test that CPU tensor input through H100 lookup table returns None"""
        # Create CPU input nodes
        cpu_input_nodes = [
            MockInputNode(device_type="cpu", device_index=0),
            MockInputNode(device_type="cpu", device_index=0),
        ]

        # Create lookup table configured for H100
        h100_lookup_table_data = {
            "NVIDIA H100": {
                "mm": {
                    "test_key": [self.create_triton_config(block_m=128, block_n=128)]
                }
            }
        }

        inductor_config.max_autotune = True

        with (
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(
                inductor_config, "template_lookup_table", h100_lookup_table_data
            ),
        ):
            result = lookup_op_config_entries(cpu_input_nodes, "mm")
            # CPU device should not match H100 lookup table, result should be None
            self.assertIsNone(result)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestTemplateLookupTableConfig(BaseLookupTableTest):
    """Test class for new template lookup table configuration"""

    def test_in_use_with_table_data(self):
        """Test: table has data -> _in_use() returns True"""
        # Setup: table has data and max_autotune enabled
        inductor_config.template_lookup_table = {"test": "data"}
        inductor_config.max_autotune = True

        # Should return True when table has data
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_in_use_with_empty_table(self):
        """Test: table is empty -> _in_use() returns False"""
        # Setup: table is empty
        inductor_config.template_lookup_table = {}

        # Should return False when table is empty
        result = torch._inductor.lookup_table._in_use()
        self.assertFalse(result)

    def test_in_use_raises_error_when_table_exists_but_max_autotune_disabled(self):
        """Test: RuntimeError is raised when table has data but max_autotune is not enabled"""
        # Setup: table has data but max_autotune is disabled
        inductor_config.template_lookup_table = {"test": "data"}
        inductor_config.max_autotune = False
        inductor_config.max_autotune_gemm = False

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            torch._inductor.lookup_table._in_use()

        self.assertIn(
            "The template lookup table requires max-autotune to be enabled",
            str(cm.exception),
        )
        self.assertIn(
            "Please set inductor_config.max_autotune=True, or remove the lookup table",
            str(cm.exception),
        )

    def test_in_use_works_with_max_autotune_gemm_enabled(self):
        """Test: _in_use() works when max_autotune_gemm is enabled instead of max_autotune"""
        # Setup: table has data and max_autotune_gemm enabled
        inductor_config.template_lookup_table = {"test": "data"}
        inductor_config.max_autotune = False
        inductor_config.max_autotune_gemm = True

        # Should return True when table has data and max_autotune_gemm is enabled
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_get_lookup_table_returns_config_table(self):
        """Test: _get_lookup_table() returns the config table"""
        test_table = {"device": {"op": {"key": {"template": "options"}}}}
        inductor_config.template_lookup_table = test_table
        inductor_config.max_autotune = True

        result = torch._inductor.lookup_table._get_lookup_table()
        self.assertEqual(result, test_table)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupConfigsForTemplateId(BaseLookupTableTest):
    """Test class for lookup_configs_for_template_id function with TF32 filtering"""

    @parametrize(
        "enable_max_autotune,lookup_table,lookup_dict,template_id,expected_result",
        [
            # Test when not in use
            (False, {}, {"triton": "configs"}, "triton", None),
            # Test with None lookup_dict
            (True, {"test": "data"}, None, "triton", None),
        ],
    )
    def test_lookup_configs_edge_cases(
        self,
        enable_max_autotune,
        lookup_table,
        lookup_dict,
        template_id,
        expected_result,
    ):
        """Test edge cases for template lookup"""
        if lookup_dict == "configs":
            lookup_dict = {"triton": [self.create_triton_config()]}

        self._run_template_lookup_test(
            lookup_dict, template_id, expected_result, enable_max_autotune
        )

    @parametrize(
        "allow_tf32,configs_setup,expected_count,warning_count",
        [
            # No TF32 filtering when allowed
            (True, "mixed_tf32", 2, 0),
            # TF32 filtering when not allowed
            (False, "mixed_tf32", 1, 1),
            # All filtered out
            (False, "all_tf32_true", 0, 2),
            # No ALLOW_TF32 field - not filtered
            (False, "no_tf32_field", 1, 0),
            # ALLOW_TF32=False - not filtered
            (False, "tf32_false", 1, 0),
        ],
    )
    def test_lookup_configs_tf32_filtering(
        self, allow_tf32, configs_setup, expected_count, warning_count
    ):
        """Test TF32 filtering scenarios"""
        configs_map = {
            "mixed_tf32": [
                self.create_triton_config(block_m=128, ALLOW_TF32=True),
                self.create_triton_config(block_m=64, ALLOW_TF32=False),
            ],
            "all_tf32_true": [
                self.create_triton_config(block_m=128, ALLOW_TF32=True),
                self.create_triton_config(block_m=64, ALLOW_TF32=True),
            ],
            "no_tf32_field": [
                {"template_id": "triton", "BLOCK_M": 128, "BLOCK_N": 128}
            ],
            "tf32_false": [self.create_triton_config(block_m=128, ALLOW_TF32=False)],
        }

        configs = configs_map[configs_setup]
        lookup_dict = {"triton": configs}

        # Handle expected results based on the specific test case
        if expected_count == 0:
            expected_result = []
        elif configs_setup == "mixed_tf32" and not allow_tf32:
            # For mixed TF32 with filtering, expect the config with ALLOW_TF32=False (second config)
            expected_result = [configs[1]]
        else:
            expected_result = configs[:expected_count]

        self._run_template_lookup_test(
            lookup_dict,
            "triton",
            expected_result,
            allow_tf32=allow_tf32,
            should_log_warning=warning_count > 0,
            warning_count=warning_count,
        )

    def test_lookup_configs_removes_template_id(self):
        """Test that template_id is removed from returned configs"""
        expected_configs = [
            self.create_triton_config(block_m=128, block_n=64),
            self.create_triton_config(block_m=256, block_n=128),
        ]
        lookup_dict = {"triton": expected_configs}

        self._run_template_lookup_test(lookup_dict, "triton", expected_configs)

    def test_lookup_configs_mixed_templates(self):
        """Test with multiple template types and TF32 filtering"""
        triton_configs = [
            self.create_triton_config(block_m=128, ALLOW_TF32=True),  # Will be filtered
            self.create_triton_config(block_m=64, ALLOW_TF32=False),  # Will remain
        ]
        tma_configs = [
            self.create_tma_config(block_m=256, ALLOW_TF32=True),  # Will be filtered
        ]
        lookup_dict = {"triton": triton_configs, "tma": tma_configs}

        # Test triton configs - should return 1 config (the one without TF32)
        self._run_template_lookup_test(
            lookup_dict,
            "triton",
            [triton_configs[1]],
            allow_tf32=False,
            should_log_warning=True,
            warning_count=1,
        )

        # Test tma configs - should return empty list (all filtered)
        self._run_template_lookup_test(
            lookup_dict,
            "tma",
            [],
            allow_tf32=False,
            should_log_warning=True,
            warning_count=1,
        )

    def test_lookup_configs_multiple_calls_dont_interfere(self):
        """Test that calling lookup functions twice doesn't break due to template_id removal"""
        inductor_config.max_autotune = True
        torch.backends.cuda.matmul.allow_tf32 = True

        input_nodes = self.create_mock_input_nodes(2)
        config_list = [
            self.create_triton_config(block_m=128, block_n=64),
            self.create_tma_config(block_m=512, block_n=256),
        ]
        lookup_table_data = {"NVIDIA H100": {"mm": {"test_key": config_list}}}

        with (
            patch("torch._inductor.lookup_table._dev_key", return_value="NVIDIA H100"),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(inductor_config, "template_lookup_table", lookup_table_data),
        ):
            # First call - should find entries
            first_result = lookup_op_config_entries(input_nodes, "mm")
            self.assertIsNotNone(first_result)
            self.assertEqual(len(first_result["triton"]), 1)
            self.assertEqual(len(first_result["tma"]), 1)

            # Second call - should find same entries again
            second_result = lookup_op_config_entries(input_nodes, "mm")
            self.assertIsNotNone(second_result)
            self.assertEqual(len(second_result["triton"]), 1)
            self.assertEqual(len(second_result["tma"]), 1)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_CUDA:
        run_tests()
