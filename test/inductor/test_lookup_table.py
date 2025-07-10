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

    def create_triton_config(
        self,
        block_m=128,
        block_n=128,
        block_k=64,
        num_stages=2,
        num_warps=2,
        template_id="triton",
        **kwargs,
    ):
        """Create a triton backend configuration with template_id field"""
        # This should match what mm_options() expects to receive after processing
        config = {
            "template_id": template_id,
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

    def create_tma_config(
        self, block_m=256, block_n=128, block_k=64, num_stages=4, num_warps=8, **kwargs
    ):
        """Create a TMA backend configuration with template_id field"""
        config = {
            "template_id": "tma",
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
        }
        config.update(kwargs)
        return config

    def create_decompose_k_config(self, k=4):
        """Create a decompose_k backend configuration with template_id field"""
        return {"template_id": "decompose_k", "k": k}

    def create_bias_addmm_config(self):
        """Create a bias_addmm backend configuration with template_id field"""
        return {"template_id": "bias_addmm"}


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableCore(BaseLookupTableTest):
    """Consolidated tests for core lookup table functionality"""

    def setUp(self):
        super().setUp()
        self.original_max_autotune = inductor_config.max_autotune
        self.original_max_autotune_gemm = inductor_config.max_autotune_gemm

    def tearDown(self):
        inductor_config.max_autotune = self.original_max_autotune
        inductor_config.max_autotune_gemm = self.original_max_autotune_gemm
        super().tearDown()

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
        inductor_config.max_autotune = True

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
            result = lookup_op_config_entries(input_nodes, method)

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

        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": config_list}}}
        self._run_lookup_test(
            "NVIDIA H100(9, 0)", "test_key", "mm", lookup_table_data, expected_result
        )

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        with patch.object(inductor_config.template_lookup_table, "table", {}):
            result = lookup_op_config_entries(self.create_mock_input_nodes(2), "mm")
            self.assertIsNone(result)

    def test_validation_missing_template_id(self):
        """Test validation error when template_id is missing"""
        invalid_config = {"BLOCK_M": 128, "BLOCK_N": 128}
        config_list = [invalid_config]
        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": config_list}}}
        input_nodes = self.create_mock_input_nodes(2)
        inductor_config.max_autotune = True

        with (
            patch(
                "torch._inductor.lookup_table._dev_key",
                return_value="NVIDIA H100(9, 0)",
            ),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(
                inductor_config.template_lookup_table, "table", lookup_table_data
            ),
        ):
            with self.assertRaises(ValueError) as cm:
                torch._inductor.lookup_table._get_op_lookup_table(input_nodes, "mm")
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_validation_non_dict_config(self):
        """Test validation error when config is not a dict"""
        invalid_config = "not_a_dict"
        config_list = [invalid_config]
        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": config_list}}}
        input_nodes = self.create_mock_input_nodes(2)
        inductor_config.max_autotune = True

        with (
            patch(
                "torch._inductor.lookup_table._dev_key",
                return_value="NVIDIA H100(9, 0)",
            ),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(
                inductor_config.template_lookup_table, "table", lookup_table_data
            ),
        ):
            with self.assertRaises(ValueError) as cm:
                torch._inductor.lookup_table._get_op_lookup_table(input_nodes, "mm")
            self.assertIn("is not a dictionary", str(cm.exception))

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
        self.original_max_autotune = inductor_config.max_autotune
        self.original_max_autotune_gemm = inductor_config.max_autotune_gemm

    def tearDown(self):
        # Restore original values
        inductor_config.template_lookup_table.table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        inductor_config.max_autotune_gemm = self.original_max_autotune_gemm

    def test_in_use_with_table_data(self):
        """Test: table has data -> _in_use() returns True"""
        # Setup: table has data and max_autotune enabled
        inductor_config.template_lookup_table.table = {"test": "data"}
        inductor_config.max_autotune = True

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

    def test_in_use_raises_error_when_table_exists_but_max_autotune_disabled(self):
        """Test: RuntimeError is raised when table has data but max_autotune is not enabled"""
        # Setup: table has data but max_autotune is disabled
        inductor_config.template_lookup_table.table = {"test": "data"}
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
        inductor_config.template_lookup_table.table = {"test": "data"}
        inductor_config.max_autotune = False
        inductor_config.max_autotune_gemm = True

        # Should return True when table has data and max_autotune_gemm is enabled
        result = torch._inductor.lookup_table._in_use()
        self.assertTrue(result)

    def test_get_lookup_table_returns_config_table(self):
        """Test: _get_lookup_table() returns the config table"""
        test_table = {"device": {"op": {"key": {"template": "options"}}}}
        inductor_config.template_lookup_table.table = test_table
        inductor_config.max_autotune = True

        result = torch._inductor.lookup_table._get_lookup_table()
        self.assertEqual(result, test_table)


@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestLookupConfigsForTemplateId(BaseLookupTableTest):
    """Test class for lookup_configs_for_template_id function with TF32 filtering"""

    def setUp(self):
        super().setUp()
        self.original_max_autotune = inductor_config.max_autotune
        self.original_allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    def tearDown(self):
        inductor_config.max_autotune = self.original_max_autotune
        torch.backends.cuda.matmul.allow_tf32 = self.original_allow_tf32
        super().tearDown()

    def test_lookup_configs_not_in_use(self):
        """Test lookup_configs_for_template_id when not in use"""
        inductor_config.max_autotune = False
        inductor_config.template_lookup_table.table = {}

        lookup_dict = {"triton": [self.create_triton_config()]}
        result = lookup_template_configs_from_op(lookup_dict, "triton")
        self.assertIsNone(result)

    def test_lookup_configs_none_lookup_dict(self):
        """Test lookup_configs_for_template_id with None lookup_dict"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}

        result = lookup_template_configs_from_op(None, "triton")
        self.assertIsNone(result)

    def test_lookup_configs_success_no_tf32_filtering(self):
        """Test successful lookup without TF32 filtering"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 allowed

        configs = [
            self.create_triton_config(block_m=128, ALLOW_TF32=True),
            self.create_triton_config(block_m=64, ALLOW_TF32=False),
        ]
        lookup_dict = {"triton": configs}

        result = lookup_template_configs_from_op(lookup_dict, "triton")
        self.assertEqual(len(result), 2)
        self.assert_configs_equal_without_template_id(result, configs)

    def test_lookup_configs_tf32_filtering_enabled(self):
        """Test TF32 filtering when torch.backends.cuda.matmul.allow_tf32 is False"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not allowed

        config_with_tf32 = self.create_triton_config(block_m=128, ALLOW_TF32=True)
        config_without_tf32 = self.create_triton_config(block_m=64, ALLOW_TF32=False)
        configs = [config_with_tf32, config_without_tf32]
        lookup_dict = {"triton": configs}

        with patch("torch._inductor.lookup_table.log.warning") as mock_warning:
            result = lookup_template_configs_from_op(lookup_dict, "triton")

            # Should only return the config without TF32
            self.assertEqual(len(result), 1)
            self.assert_configs_equal_without_template_id(
                result[0], config_without_tf32
            )

            # Should log warning about filtered config
            mock_warning.assert_called_once()
            warning_call = mock_warning.call_args[0]
            self.assertIn("Filtering out config with ALLOW_TF32=True", warning_call[0])
            self.assertIn(
                "torch.backends.cuda.matmul.allow_tf32 is False", warning_call[0]
            )

    def test_lookup_configs_all_filtered_out(self):
        """Test when all configs are filtered out due to TF32"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not allowed

        # All configs have ALLOW_TF32=True
        configs = [
            self.create_triton_config(block_m=128, ALLOW_TF32=True),
            self.create_triton_config(block_m=64, ALLOW_TF32=True),
        ]
        lookup_dict = {"triton": configs}

        with patch("torch._inductor.lookup_table.log.warning") as mock_warning:
            result = lookup_template_configs_from_op(lookup_dict, "triton")

            # Should return empty list when all configs are filtered
            self.assertEqual(result, [])

            # Should log warning for each filtered config
            self.assertEqual(mock_warning.call_count, 2)

    def test_lookup_configs_no_allow_tf32_field(self):
        """Test configs without ALLOW_TF32 field are not filtered"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not allowed

        # Config without ALLOW_TF32 field
        config_no_tf32_field = {"template_id": "triton", "BLOCK_M": 128, "BLOCK_N": 128}
        configs = [config_no_tf32_field]
        lookup_dict = {"triton": configs}

        result = lookup_template_configs_from_op(lookup_dict, "triton")

        # Should return the config since it doesn't have ALLOW_TF32 field
        self.assertEqual(len(result), 1)
        self.assert_configs_equal_without_template_id(result, configs)

    def test_lookup_configs_allow_tf32_false_not_filtered(self):
        """Test configs with ALLOW_TF32=False are not filtered"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not allowed

        config_tf32_false = self.create_triton_config(block_m=128, ALLOW_TF32=False)
        configs = [config_tf32_false]
        lookup_dict = {"triton": configs}

        result = lookup_template_configs_from_op(lookup_dict, "triton")

        # Should return the config since ALLOW_TF32=False
        self.assertEqual(len(result), 1)
        self.assert_configs_equal_without_template_id(result, configs)

    def test_lookup_configs_mixed_templates(self):
        """Test with multiple template types and TF32 filtering"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not allowed

        triton_configs = [
            self.create_triton_config(block_m=128, ALLOW_TF32=True),  # Will be filtered
            self.create_triton_config(block_m=64, ALLOW_TF32=False),  # Will remain
        ]
        tma_configs = [
            self.create_tma_config(block_m=256, ALLOW_TF32=True),  # Will be filtered
        ]

        lookup_dict = {"triton": triton_configs, "tma": tma_configs}

        with patch("torch._inductor.lookup_table.log.warning") as mock_warning:
            # Test triton configs
            triton_result = lookup_template_configs_from_op(lookup_dict, "triton")
            self.assertEqual(len(triton_result), 1)
            self.assertEqual(triton_result[0]["BLOCK_M"], 64)

            # Test tma configs
            tma_result = lookup_template_configs_from_op(lookup_dict, "tma")
            self.assertEqual(tma_result, [])  # All filtered out

            # Should log warnings for both filtered configs
            self.assertEqual(mock_warning.call_count, 2)

    def test_lookup_configs_removes_template_id(self):
        """Test that lookup_op_configs_for_template_id removes template_id from returned configs"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 to avoid filtering

        # Create configs with template_id field
        expected_configs = [
            self.create_triton_config(block_m=128, block_n=64, template_id="triton"),
            self.create_triton_config(block_m=256, block_n=128, template_id="triton"),
        ]

        # Verify original configs have template_id
        for config in expected_configs:
            self.assertIn("template_id", config)
            self.assertEqual(config["template_id"], "triton")

        lookup_dict = {"triton": expected_configs}
        result = lookup_template_configs_from_op(lookup_dict, "triton")

        # Use custom comparator to verify template_id removal and config equality
        self.assert_configs_equal_without_template_id(result, expected_configs)

    def test_lookup_configs_multiple_calls_dont_interfere(self):
        """Test that calling lookup functions twice doesn't break due to template_id removal"""
        inductor_config.max_autotune = True
        inductor_config.template_lookup_table.table = {"test": "data"}
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 to avoid filtering

        # Create mock input nodes
        input_nodes = self.create_mock_input_nodes(2)

        # Create configs with template_id fields
        config_list = [
            self.create_triton_config(block_m=128, block_n=64, template_id="triton"),
            self.create_triton_config(block_m=256, block_n=128, template_id="triton"),
            self.create_tma_config(block_m=512, block_n=256, template_id="tma"),
        ]

        # Set up the lookup table
        lookup_table_data = {"NVIDIA H100(9, 0)": {"mm": {"test_key": config_list}}}

        with (
            patch(
                "torch._inductor.lookup_table._dev_key",
                return_value="NVIDIA H100(9, 0)",
            ),
            patch(
                "torch._inductor.lookup_table._template_lookup_key",
                return_value="test_key",
            ),
            patch.object(
                inductor_config.template_lookup_table, "table", lookup_table_data
            ),
        ):
            # First call sequence
            first_lookup_dict = lookup_op_config_entries(input_nodes, "mm")
            self.assertIsNotNone(first_lookup_dict)
            self.assertIn("triton", first_lookup_dict)
            self.assertIn("tma", first_lookup_dict)

            # Verify configs have template_id before calling lookup_template_configs_from_op
            for config in first_lookup_dict["triton"]:
                self.assertIn("template_id", config)
            for config in first_lookup_dict["tma"]:
                self.assertIn("template_id", config)

            first_triton_result = lookup_template_configs_from_op(
                first_lookup_dict, "triton"
            )
            first_tma_result = lookup_template_configs_from_op(first_lookup_dict, "tma")

            # Verify first call results using custom comparator
            expected_triton_configs = [
                self.create_triton_config(
                    block_m=128, block_n=64, template_id="triton"
                ),
                self.create_triton_config(
                    block_m=256, block_n=128, template_id="triton"
                ),
            ]
            expected_tma_configs = [
                self.create_tma_config(block_m=512, block_n=256, template_id="tma")
            ]

            self.assert_configs_equal_without_template_id(
                first_triton_result, expected_triton_configs
            )
            self.assert_configs_equal_without_template_id(
                first_tma_result, expected_tma_configs
            )

            # Second call sequence - this should work even after template_id removal
            second_lookup_dict = lookup_op_config_entries(input_nodes, "mm")
            self.assertIsNotNone(second_lookup_dict)
            self.assertIn("triton", second_lookup_dict)
            self.assertIn("tma", second_lookup_dict)

            # Verify configs still have template_id for the second call
            for config in second_lookup_dict["triton"]:
                self.assertIn("template_id", config)
            for config in second_lookup_dict["tma"]:
                self.assertIn("template_id", config)

            second_triton_result = lookup_template_configs_from_op(
                second_lookup_dict, "triton"
            )
            second_tma_result = lookup_template_configs_from_op(
                second_lookup_dict, "tma"
            )

            # Verify second call results using custom comparator
            self.assert_configs_equal_without_template_id(
                second_triton_result, expected_triton_configs
            )
            self.assert_configs_equal_without_template_id(
                second_tma_result, expected_tma_configs
            )

            # Verify that the results are equivalent between calls
            self.assertEqual(len(first_triton_result), len(second_triton_result))
            self.assertEqual(len(first_tma_result), len(second_tma_result))
            self.assertEqual(first_triton_result, second_triton_result)
            self.assertEqual(first_tma_result, second_tma_result)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_CUDA:
        run_tests()
