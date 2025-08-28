# Owner(s): ["module: inductor"]
import unittest
from typing import Optional, Union
from unittest.mock import patch

import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.lookup_table.core import lookup_key_suffix, lookup_template_configs
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class MockTensorNode:
    """Mock input node that wraps a real tensor for testing"""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_device(self) -> torch.device:
        return self.tensor.device

    def get_dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def get_size(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    def get_stride(self) -> tuple[int, ...]:
        return tuple(self.tensor.stride())


class MockMMKernelInputs(MMKernelInputs):
    """Mock MMKernelInputs that subclasses the real class and uses real tensors"""

    def __init__(
        self,
        tensors: list[torch.Tensor],
        scalars: Optional[dict[str, Union[float, int]]] = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        """Initialize with real tensors, creating mock nodes for the base class"""
        mock_nodes = [MockTensorNode(t) for t in tensors]
        super().__init__(mock_nodes, scalars, mat1_idx=mat1_idx, mat2_idx=mat2_idx)
        self.tensors = tensors  # Keep reference to original tensors

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to symbolic since real tensors already have int shapes"""
        return self.shapes_symbolic()

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to symbolic since real tensors already have int strides"""
        return self.strides_symbolic()

    def mnk_hinted(self) -> tuple[int, int, int]:
        """Delegate to symbolic since real tensors already have int dimensions"""
        return self.mnk_symbolic()

    @property
    def device_type(self) -> Optional[str]:
        return self.tensors[0].device.type


class BaseLookupTableTest(TestCase):
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        super().setUp()
        self.original_table = torch._inductor.config.template_config_lookup_table.table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        inductor_config.max_autotune = True
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        torch._inductor.config.template_config_lookup_table.table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        super().tearDown()

    def create_mock_mm_kernel_inputs(
        self,
        shapes: Optional[list[tuple[int, ...]]] = None,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float32,
        scalars: Optional[dict[str, Union[float, int]]] = None,
    ) -> MockMMKernelInputs:
        """Create MockMMKernelInputs with real tensors"""
        if shapes is None:
            shapes = [(128, 128), (128, 128)]  # Default MM shapes

        tensors = []
        for shape in shapes:
            # Create a real tensor with the specified shape, device, and dtype
            tensor = torch.randn(shape, device=device, dtype=dtype)
            tensors.append(tensor)

        return MockMMKernelInputs(tensors, scalars)

    def create_lookup_key(self, method, kernel_inputs):
        """Create a lookup key that matches core.py's make_lookup_key"""
        # This matches exactly what make_lookup_key does in core.py
        flat_key = f"{kernel_inputs.key}+{method}+{lookup_key_suffix()}"
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
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Mock a different device to create mismatch
        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # looking for addmm but created the entry with mm - should mismatch the key and return
            # an empty result
            result = lookup_template_configs(kernel_inputs, "addmm", ["triton"])
            self.assertEqual(result, {})

    def test_successful_lookup_with_template_filtering(self):
        """Test successful lookup that filters configs by template_id"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128, BLOCK_N=128),
            self.create_config("triton", BLOCK_M=64, BLOCK_N=64),
            self.create_config("tma", BLOCK_M=256, BLOCK_N=128),
            self.create_config("decompose_k", k_split=4),
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # Test triton template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["triton"]), 2)
            for config in result["triton"]:
                self.assertNotIn("template_id", config)
                self.assertIn("BLOCK_M", config)

            # Test tma template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["tma"]), 1)
            self.assertNotIn("template_id", result["tma"][0])
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

            # Test decompose_k template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["decompose_k"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["decompose_k"]), 1)
            self.assertNotIn("template_id", result["decompose_k"][0])
            self.assertEqual(result["decompose_k"][0]["k_split"], 4)

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        with patch.object(inductor_config.template_config_lookup_table, "table", {}):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertEqual(result, {})

    def test_validation_error(self):
        """Test validation error for invalid config"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        invalid_config = {"BLOCK_M": 128}  # missing template_id
        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [invalid_config]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            with self.assertRaises(ValueError) as cm:
                lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_cpu_input_returns_empty(self):
        """Test that CPU tensor input returns empty dict"""
        # Create kernel inputs with CPU tensors
        kernel_inputs = self.create_mock_mm_kernel_inputs(device=torch.device("cpu"))

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertEqual(result, {})  # Should return empty dict for CPU

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
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        configs = [
            self.create_config("triton", BLOCK_M=128, ALLOW_TF32=tf32_val)
            for tf32_val in tf32_configs
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): configs}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            if expected_count > 0:
                assert result is not None, "Result should not be None"
                self.assertEqual(len(result["triton"]), expected_count)
            else:
                self.assertEqual(result, {})

    def test_multiple_calls_work(self):
        """Test that calling lookup functions multiple times works correctly"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
        ]
        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # First calls
            result1 = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            result2 = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result1 is not None, "Result1 should not be None"
            assert result2 is not None, "Result2 should not be None"
            self.assertEqual(len(result1["triton"]), 1)
            self.assertEqual(len(result2["tma"]), 1)

            # Second calls should work the same
            result3 = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            result4 = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result3 is not None, "Result3 should not be None"
            assert result4 is not None, "Result4 should not be None"
            self.assertEqual(len(result3["triton"]), 1)
            self.assertEqual(len(result4["tma"]), 1)

    def test_batch_lookup_mixed_entries(self):
        """Test batch lookup where some templates have entries and others don't"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
            # No decompose_k config in lookup table
        ]
        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # Test batch lookup with mixed results
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton", "tma", "decompose_k"]
            )
            assert result is not None, "Result should not be None"

            # Should have entries for triton and tma, but not decompose_k
            self.assertIn("triton", result)
            self.assertIn("tma", result)
            self.assertNotIn("decompose_k", result)

            self.assertEqual(len(result["triton"]), 1)
            self.assertEqual(len(result["tma"]), 1)
            self.assertEqual(result["triton"][0]["BLOCK_M"], 128)
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

    @parametrize(
        "config_hash,template_hash,expected_kept",
        [
            # Hash matching (config kept)
            ("hash123", "hash123", True),
            # Hash mismatch (config filtered)
            ("hash123", "hash456", False),
            # Config without hash (config kept)
            (None, "hash123", True),
            # Template without hash (config kept)
            ("hash123", None, True),
            # Both None (config kept)
            (None, None, True),
        ],
    )
    def test_template_hash_checking(self, config_hash, template_hash, expected_kept):
        """Test template hash validation behavior"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config = self.create_config("triton", BLOCK_M=128, BLOCK_N=64)
        if config_hash is not None:
            config["template_hash"] = config_hash

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        template_hash_map = (
            {"triton": template_hash} if template_hash is not None else {}
        )

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table, "check_src_hash", True
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if expected_kept:
                assert result is not None, "Result should not be None"
                self.assertIn("triton", result)
                self.assertEqual(len(result["triton"]), 1)
                # template_hash should be removed from returned config
                self.assertNotIn("template_hash", result["triton"][0])
            else:
                # Config was filtered out due to hash mismatch
                self.assertEqual(result, {})

    def test_template_hash_checking_disabled(self):
        """Test that hash checking is skipped when config flag is disabled"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create config with mismatching hash
        config = self.create_config("triton", BLOCK_M=128, template_hash="hash123")

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        # Provide different template hash that would normally cause filtering
        template_hash_map = {"triton": "hash456"}

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table,
                "check_src_hash",
                False,
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config even with mismatching hash since checking is disabled
            assert result is not None, "Result should not be None"
            self.assertIn("triton", result)
            self.assertEqual(len(result["triton"]), 1)
            # template_hash should still be removed from returned config
            self.assertNotIn("template_hash", result["triton"][0])

    def test_template_hash_mixed_scenarios(self):
        """Test mixed hash scenarios with multiple configs"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config(
                "triton", BLOCK_M=128, template_hash="correct_hash"
            ),  # Should be kept
            self.create_config(
                "triton", BLOCK_M=64, template_hash="wrong_hash"
            ),  # Should be filtered
            self.create_config("triton", BLOCK_M=32),  # No hash, should be kept
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        template_hash_map = {"triton": "correct_hash"}

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table, "check_src_hash", True
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            assert result is not None, "Result should not be None"
            self.assertIn("triton", result)
            # Should keep 2 configs: the one with correct hash and the one without hash
            self.assertEqual(len(result["triton"]), 2)

            # Check that kept configs have expected BLOCK_M values
            kept_block_ms = [config["BLOCK_M"] for config in result["triton"]]
            self.assertIn(128, kept_block_ms)  # Config with correct hash
            self.assertIn(32, kept_block_ms)  # Config without hash
            self.assertNotIn(
                64, kept_block_ms
            )  # Config with wrong hash should be filtered


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
