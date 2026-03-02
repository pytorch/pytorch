# Owner(s): ["module: inductor"]
import re
import unittest
from functools import partial
from typing import Any, Optional, Union
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
    ExternKernelCaller,
    TritonTemplateCaller,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, get_num_sms, TMA_DESCRIPTOR_SIZE
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON, HAS_GPU
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tma_device


# Conditional patch for decompose_k tests - override to 10 on ROCm, no-op elsewhere
_DECOMPOSE_K_PATCH_ROCM = (
    {"triton.num_decompose_k_splits": 10} if torch.version.hip else {}
)


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
        return self.strides_symbolic()  # pyre-ignore

    def mnk_hinted(self) -> tuple[int, int, int]:
        """Delegate to symbolic since real tensors already have int dimensions"""
        return self.mnk_symbolic()  # pyre-ignore

    @property
    def device_type(self) -> Optional[str]:
        return self.tensors[0].device.type


class BaseLookupTableTest(TestCase):
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        super().setUp()
        self.original_table = inductor_config.lookup_table.table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        inductor_config.max_autotune = True
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        inductor_config.lookup_table.table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        super().tearDown()

    def create_mock_mm_kernel_inputs(
        self,
        shapes: Optional[list[tuple[int, ...]]] = None,
        device: torch.device = torch.device("cuda"),
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
        """Create a lookup key using LookupTableChoices"""
        choices = LookupTableChoices()
        return choices.make_lookup_key(kernel_inputs, method)

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

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            # looking for addmm but created the entry with mm - should mismatch the key and return
            # an empty result
            result = test_choices.lookup_template_configs(
                kernel_inputs, "addmm", ["triton"]
            )
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

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # Test triton template filtering
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["triton"]), 2)
            for config in result["triton"]:
                self.assertNotIn("template_id", config)
                self.assertIn("BLOCK_M", config)

            # Test tma template filtering
            result = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["tma"]), 1)
            self.assertNotIn("template_id", result["tma"][0])
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

            # Test decompose_k template filtering
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["decompose_k"]
            )
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["decompose_k"]), 1)
            self.assertNotIn("template_id", result["decompose_k"][0])
            self.assertEqual(result["decompose_k"][0]["k_split"], 4)

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        with patch.object(inductor_config.lookup_table, "table", {}):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            self.assertEqual(result, {})

    def test_validation_error(self):
        """Test validation error for invalid config"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        invalid_config = {"BLOCK_M": 128}  # missing template_id

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [invalid_config]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            with self.assertRaises(ValueError) as cm:
                test_choices.lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_cpu_input_returns_empty(self):
        """Test that CPU tensor input returns empty dict"""
        # Create kernel inputs with CPU tensors
        kernel_inputs = self.create_mock_mm_kernel_inputs(device=torch.device("cpu"))

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            self.assertEqual(result, {})  # Should return empty dict for CPU

    def test_multiple_calls_work(self):
        """Test that calling lookup functions multiple times works correctly"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # First calls
            result1 = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            result2 = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result1 is None:
                raise AssertionError("Result1 should not be None")
            if result2 is None:
                raise AssertionError("Result2 should not be None")
            self.assertEqual(len(result1["triton"]), 1)
            self.assertEqual(len(result2["tma"]), 1)

            # Second calls should work the same
            result3 = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            result4 = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result3 is None:
                raise AssertionError("Result3 should not be None")
            if result4 is None:
                raise AssertionError("Result4 should not be None")
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

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # Test batch lookup with mixed results
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton", "tma", "decompose_k"]
            )
            if result is None:
                raise AssertionError("Result should not be None")

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

        template_hash_map = (
            {"triton": template_hash} if template_hash is not None else {}
        )

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", True),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if expected_kept:
                if result is None:
                    raise AssertionError("Result should not be None")
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

        # Provide different template hash that would normally cause filtering
        template_hash_map = {"triton": "hash456"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(
                inductor_config.lookup_table,
                "check_src_hash",
                False,
            ),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config even with mismatching hash since checking is disabled
            if result is None:
                raise AssertionError("Result should not be None")
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

        template_hash_map = {"triton": "correct_hash"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", True),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if result is None:
                raise AssertionError("Result should not be None")
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

            # template_hash should be removed from returned configs
            for config in result["triton"]:
                self.assertNotIn("template_hash", config)

    @parametrize(
        "config_hash,description",
        [
            ("definitely_malformed_hash_!@#$%", "malformed hash"),
            (12345, "non-string hash"),
            ("", "empty string hash"),
            (None, "missing hash field"),
        ],
    )
    def test_hash_checking_disabled_edge_cases(self, config_hash, description):
        """Test that configs are kept when hash checking is disabled, regardless of hash validity"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create config with potentially problematic hash
        config = self.create_config("triton", BLOCK_M=128)
        if config_hash is not None:
            config["template_hash"] = config_hash
        # If config_hash is None, don't add template_hash field at all

        # Provide a valid template hash that would normally be used for comparison
        template_hash_map = {"triton": "valid_template_hash_abc123"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", False),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config regardless of hash validity since checking is disabled
            if result is None:
                raise AssertionError(f"Result should not be None for {description}")
            self.assertIn(
                "triton", result, f"Should have triton result for {description}"
            )
            self.assertEqual(
                len(result["triton"]), 1, f"Should have 1 config for {description}"
            )
            # template_hash should be removed from returned config
            self.assertNotIn(
                "template_hash",
                result["triton"][0],
                f"template_hash should be removed from result for {description}",
            )
            # Other config fields should be preserved
            self.assertEqual(
                result["triton"][0]["BLOCK_M"],
                128,
                f"BLOCK_M should be preserved for {description}",
            )

    @parametrize(
        "table_has_device_key,lookup_device_matches,expected_found",
        [
            # Device-specific key in table, same device -> found
            (True, True, True),
            # Device-specific key in table, different device -> not found
            (True, False, False),
            # Device-agnostic key in table, same device -> found
            (False, True, True),
            # Device-agnostic key in table, different device -> found (device-agnostic)
            (False, False, True),
        ],
    )
    def test_device_key_lookup_scenarios(
        self, table_has_device_key, lookup_device_matches, expected_found
    ):
        """Test lookup behavior with device-specific vs device-agnostic keys"""
        # Create kernel inputs for "device_1" (our reference device)
        kernel_inputs_device1 = self.create_mock_mm_kernel_inputs()

        # Create config
        config = self.create_config("triton", BLOCK_M=128)

        # Create a test choices class for generating the table key
        class TableKeyChoices(LookupTableChoices):
            @staticmethod
            def _get_device_key(device):
                if device.type != "cuda":
                    return None
                return "device_1"  # Always device_1 for table key generation

        table_key_choices = TableKeyChoices()

        # Generate table key based on whether it should include device
        if table_has_device_key:
            table_key = table_key_choices.make_lookup_key(
                kernel_inputs_device1, "mm", include_device=True
            )
        else:
            table_key = table_key_choices.make_lookup_key(
                kernel_inputs_device1, "mm", include_device=False
            )

        lookup_table_data = {table_key: [config]}

        # Create test choices class for the actual lookup with different device behavior
        if lookup_device_matches:

            class TestChoices(LookupTableChoices):
                @staticmethod
                def _get_device_key(device):
                    if device.type != "cuda":
                        return None
                    return "device_1"

        else:

            class TestChoices(LookupTableChoices):
                @staticmethod
                def _get_device_key(device):
                    if device.type != "cuda":
                        return None
                    return "device_2"

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = TestChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs_device1, "mm", ["triton"]
            )

        if expected_found:
            if result is None:
                raise AssertionError(
                    f"Result should not be None when expected_found={expected_found}"
                )
            self.assertIn("triton", result, "Should have triton result when found")
            self.assertEqual(len(result["triton"]), 1, "Should have exactly 1 config")
            self.assertEqual(
                result["triton"][0]["BLOCK_M"], 128, "Config should be preserved"
            )
        else:
            self.assertEqual(
                result,
                {},
                f"Should return empty dict when expected_found={expected_found}",
            )

    def test_device_key_priority(self):
        """Test that device-specific keys take priority over device-agnostic keys"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create two different configs
        device_specific_config = self.create_config(
            "triton", BLOCK_M=256
        )  # Different BLOCK_M
        device_agnostic_config = self.create_config("triton", BLOCK_M=128)

        # Create a test choices instance to generate keys
        key_choices = LookupTableChoices()

        # Create both key types for the same inputs
        device_key = key_choices.make_lookup_key(
            kernel_inputs, "mm", include_device=True
        )
        device_agnostic_key = key_choices.make_lookup_key(
            kernel_inputs, "mm", include_device=False
        )

        # Put both in the table
        lookup_table_data = {
            device_key: [device_specific_config],
            device_agnostic_key: [device_agnostic_config],
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )

            # Should get device-specific config (BLOCK_M=256), not device-agnostic (BLOCK_M=128)
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertIn("triton", result)
            self.assertEqual(len(result["triton"]), 1)
            self.assertEqual(
                result["triton"][0]["BLOCK_M"],
                256,
                "Should use device-specific config when both exist",
            )

    def test_make_lookup_key_variants(self):
        """Test the make_lookup_key_variants helper function"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        test_choices = LookupTableChoices()
        device_key, device_agnostic_key = test_choices.make_lookup_key_variants(
            kernel_inputs, "mm"
        )

        # Both should be strings
        self.assertIsInstance(device_key, str)
        self.assertIsInstance(device_agnostic_key, str)

        # Device key should be longer (contains device info)
        self.assertGreater(len(device_key), len(device_agnostic_key))

        # Device-agnostic key should be contained in device key (as a substring after device part)
        self.assertIn(device_agnostic_key.split("+mm")[0], device_key)


class UnifiedModel(nn.Module):
    """Unified model for different matrix operations"""

    def __init__(self, operation="mm"):
        super().__init__()
        self.operation = operation

    def forward(self, *args):
        if self.operation == "mm":
            return torch.mm(args[0], args[1])
        elif self.operation == "addmm":
            return torch.addmm(args[0], args[1], args[2])
        elif self.operation == "bmm":
            return torch.bmm(args[0], args[1])
        elif self.operation == "mm_plus_mm":
            return torch.mm(args[0], args[1]) + torch.mm(args[2], args[3])
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


def verify_choice_names(choices: list[Any], pattern: str, expected_count: int = 1):
    """Verify choices match expected pattern and count"""
    if len(choices) != expected_count:
        raise ValueError(f"Expected {expected_count} choices, got {len(choices)}")
    for choice in choices:
        if not re.search(pattern, choice.name):
            raise ValueError(
                f"Choice name '{choice.name}' doesn't match pattern '{pattern}'"
            )
    return choices


class BaseE2ELookupTableTest(BaseLookupTableTest):
    """Base class for E2E lookup table tests"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.dev_key = LookupTableChoices._get_device_key(self.device)
        self.original_lookup_table = inductor_config.lookup_table.table
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        inductor_config.lookup_table.table = self.original_lookup_table
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        clear_preprocessing_fns()

    def create_tensors(self, operation, b=8, m=64, n=64, k=32):
        """Create test tensors for operations with configurable dimensions"""
        if operation in ["mm", "addmm", "mm_plus_mm"]:
            A = torch.randn(m, k, device=self.device, dtype=torch.float16)
            B = torch.randn(k, n, device=self.device, dtype=torch.float16)
            if operation == "mm":
                return [A, B]
            if operation == "addmm":
                return [
                    torch.randn((m, n), device=self.device, dtype=torch.float16),
                    A,
                    B,
                ]
            elif operation == "mm_plus_mm":
                return [
                    A,
                    B,
                    torch.randn(m, k, device=self.device, dtype=torch.float16),
                    torch.randn(k, n, device=self.device, dtype=torch.float16),
                ]
        elif operation == "bmm":
            return [
                torch.randn(b, m, k, device=self.device, dtype=torch.float16),
                torch.randn(b, k, n, device=self.device, dtype=torch.float16),
            ]
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def setup_lookup_table(self, operation, tensors, configs):
        """Setup lookup table with configuration"""
        scalars = {}
        if operation in ["addmm", "baddbmm"]:
            scalars["beta"] = 1
            scalars["alpha"] = 1
        mock_kernel_inputs = MockMMKernelInputs(tensors, scalars)
        flat_key = self.create_lookup_key(operation, mock_kernel_inputs)
        inductor_config.lookup_table.table = {flat_key: configs}

    def run_model(self, operation, tensors, config_patches=None):
        """Run compiled model with configuration"""
        config = {"max_autotune_gemm": True, "test_configs.max_mm_configs": 4}
        if config_patches:
            config.update(config_patches)

        model = UnifiedModel(operation)
        with inductor_config.patch(config):
            compiled_model = torch.compile(model.to(self.device))
            return compiled_model(*tensors)

    def create_basic_config(self, template_id):
        """Create basic configuration for template"""
        configs = {
            torch._inductor.kernel.mm.mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.bmm.bmm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
                "A_ROW_MAJOR": True,
                "B_ROW_MAJOR": True,
                "NUM_SMS": get_num_sms(),
                "TMA_SIZE": TMA_DESCRIPTOR_SIZE,
                "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
            },
            torch._inductor.kernel.mm.aten_bias_addmm.uid: {},
            torch._inductor.kernel.mm.decompose_k_subgraph_template.uid: {"k_split": 4},
        }
        return {"template_id": template_id, **configs.get(template_id, {})}

    def _create_simple_matmul_model(self):
        """Create a simple matmul model for recording tests"""

        class SimpleMatmul(nn.Module):
            def forward(self, a, b):
                return torch.mm(a, b)

        return SimpleMatmul()

    def _create_test_inputs(self, device="cuda"):
        """Create test inputs for matmul"""
        return [
            torch.randn(512, 512, device=device, dtype=torch.float32),
            torch.randn(512, 512, device=device, dtype=torch.float32),
        ]


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableE2E(BaseE2ELookupTableTest):
    """E2E tests for lookup table functionality"""

    @parametrize("max_autotune", [True, False])
    @fresh_cache()
    def test_no_lookup_table_entry_autotune_modes(self, max_autotune):
        """Test when there's no lookup table entry with different autotune modes"""
        tensors = self.create_tensors("mm")

        # Setup lookup table with different key to force no match
        self.setup_lookup_table(
            "mm",
            [
                torch.randn(64, 64, device=self.device),
                torch.randn(64, 64, device=self.device),
            ],
            [],
        )

        # Inline validation function
        def validate_choices(choices):
            if max_autotune:
                if len(choices) <= 2:
                    raise AssertionError(
                        f"Max-autotune should have >2 choices, got {len(choices)}"
                    )
                if not any(isinstance(c, ExternKernelCaller) for c in choices):
                    raise AssertionError("Should have ExternKernelCaller")
                if not any(isinstance(c, TritonTemplateCaller) for c in choices):
                    raise AssertionError("Should have TritonTemplateCaller")
            else:
                if len(choices) != 1:
                    raise AssertionError(
                        f"No max-autotune should have 1 choice, got {len(choices)}"
                    )
                if not isinstance(choices[0], ExternKernelCaller):
                    raise AssertionError(
                        f"Should be ExternKernelCaller, got {type(choices[0])}"
                    )
            return choices

        add_preprocessing_fn(validate_choices)
        self.run_model(
            "mm",
            tensors,
            {"max_autotune_gemm": max_autotune, "max_autotune": max_autotune},
        )

    @parametrize("operation", ["mm", "addmm", "bmm", "mm_plus_mm"])
    @fresh_cache()
    def test_valid_lookup_table_entry(self, operation):
        """Test when there's a valid entry for the operation"""
        k = 256 if operation == "mm_plus_mm" else 64
        tensors = self.create_tensors(operation, k=k)

        # Map operation to actual template UID
        template_mapping = {
            "mm": torch._inductor.kernel.mm.mm_template.uid,
            "addmm": torch._inductor.kernel.mm.mm_template.uid,
            "bmm": torch._inductor.kernel.bmm.bmm_template.uid,
            "mm_plus_mm": torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.uid,
        }
        template_id = template_mapping[operation]
        config = self.create_basic_config(template_id)

        self.setup_lookup_table(operation, tensors, [config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=1)
        )
        self.run_model(operation, tensors)

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @parametrize("operation", ["mm", "addmm"])
    @fresh_cache()
    def test_tma_lookup_table_entry(self, operation):
        """Test TMA template entry"""
        tensors = self.create_tensors(operation)
        config = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )

        self.setup_lookup_table(operation, tensors, [config])
        add_preprocessing_fn(
            partial(
                verify_choice_names,
                pattern="triton_mm_persistent_tma_",
                expected_count=1,
            )
        )
        self.run_model(
            operation, tensors, {"triton.enable_persistent_tma_matmul": True}
        )

    # Enable decompose_k for this test (disabled by default on ROCm)
    @fresh_cache()
    def test_decompose_k_lookup_table_entry(self):
        """Test decompose_k template entry"""
        with inductor_config.patch(_DECOMPOSE_K_PATCH_ROCM):
            tensors = self.create_tensors("mm", m=32, n=32, k=32 * 32)
            config = self.create_basic_config(
                torch._inductor.kernel.mm.decompose_k_subgraph_template.uid
            )

            self.setup_lookup_table("mm", tensors, [config])
            add_preprocessing_fn(
                partial(
                    verify_choice_names,
                    pattern="decompose_k|bmm_dtype",
                    expected_count=1,
                )
            )

            self.run_model("mm", tensors)

    @fresh_cache()
    def test_bias_addmm_lookup_table_entry(self):
        """Test bias_addmm template entry"""
        # Create bias with stride[0] == 0 for bias_addmm eligibility
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        expanded_bias = bias_unexpanded.expand(64, 64)
        tensors = [
            expanded_bias,
            torch.randn(64, 32, device=self.device, dtype=torch.float16),
            torch.randn(32, 64, device=self.device, dtype=torch.float16),
        ]

        config = self.create_basic_config(torch._inductor.kernel.mm.aten_bias_addmm.uid)
        self.setup_lookup_table("addmm", tensors, [config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="bias_addmm", expected_count=1)
        )

        # Run with original unexpanded bias
        with inductor_config.patch(
            {"max_autotune_gemm": True, "triton.autotune_cublasLt": True}
        ):
            model = UnifiedModel("addmm")
            compiled_model = torch.compile(model.to(self.device), mode="max-autotune")
            compiled_model(bias_unexpanded, tensors[1], tensors[2])

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @fresh_cache()
    def test_multiple_configs_same_template(self):
        """Test multiple configurations for same template"""
        tensors = self.create_tensors("mm")

        config1 = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        config1.update({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8})

        config2 = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        config2.update({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4})

        self.setup_lookup_table("mm", tensors, [config1, config2])
        add_preprocessing_fn(
            partial(
                verify_choice_names,
                pattern="triton_mm_persistent_tma_",
                expected_count=2,
            )
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @fresh_cache()
    def test_mixed_template_configs(self):
        """Test mixing different template types"""
        tensors = self.create_tensors("mm")

        triton_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        triton_config.update({"BLOCK_M": 128, "num_warps": 8})

        tma_config = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        tma_config.update({"BLOCK_M": 256, "num_warps": 4})

        self.setup_lookup_table("mm", tensors, [triton_config, tma_config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=2)
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})

    @fresh_cache()
    def test_template_hash_filtering_e2e(self):
        """Test end-to-end template hash filtering in real MM operation"""
        tensors = self.create_tensors("mm")

        # Get the actual src_hash from the template
        actual_hash = torch._inductor.kernel.mm.mm_template.src_hash

        # Create configs - one with correct hash, one with wrong hash
        correct_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        correct_config.update(
            {"BLOCK_M": 128, "template_hash": actual_hash}  # Use actual hash
        )

        wrong_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        wrong_config.update(
            {
                "BLOCK_M": 64,
                "template_hash": "definitely_wrong_hash_12345",  # Wrong hash
            }
        )

        self.setup_lookup_table("mm", tensors, [correct_config, wrong_config])

        # Should only get 1 choice since the wrong hash config gets filtered
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=1)
        )

        # Ensure hash checking is enabled
        with patch.object(inductor_config.lookup_table, "check_src_hash", True):
            self.run_model("mm", tensors)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
