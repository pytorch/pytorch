# Owner(s): ["module: inductor"]
import json
import re
import unittest
from functools import partial
from typing import Any

import torch
import torch._inductor.lookup_table
import torch.nn as nn
from torch._inductor import config
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA, HAS_GPU
from torch.utils._triton import has_triton_tma_device


class UnifiedModel(nn.Module):
    """Unified model that can perform different operations based on operation type"""

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


def generate_lookup_key_from_tensors(tensors):
    """Generate lookup key using the same logic as _lookup_key but from real tensors"""
    return str(
        tuple(
            (tensor.dtype, list(tensor.shape), list(tensor.stride()))
            for tensor in tensors
        )
    )


def _post_test_checking_function(
    choices: list[Any], choice_name_re: str, num_expected_choices: int = 1
):
    """Function to register and get feedback

    Args:
        choices: List of choices that are being autotuned, supplied by select_algorithm
        choice_name_re: Regex pattern to check all choice names against
        num_expected_choices: if len(choices) != num_expected_choices, then fail
    """
    if len(choices) != num_expected_choices:
        raise ValueError(f"Expected {num_expected_choices} choices, got {len(choices)}")
    for choice in choices:
        if not re.search(choice_name_re, choice.name):
            raise ValueError(
                f"Expected choice name to match regex '{choice_name_re}', got {choice.name}"
            )

    # This is to comply with the interface that expects these to be preprocessing functions
    return choices


def _test_kwargs_passed_through(
    choices: list[Any], choice_name_re: str, num_expected_choices: int = 1
):
    """Function to check that kwargs are passed through to the choice caller"""
    if len(choices) != num_expected_choices:
        raise ValueError(f"Expected {num_expected_choices} choices, got {len(choices)}")

    for choice in choices:
        if not re.search(choice_name_re, choice.name):
            raise ValueError(
                f"Expected choice name to match regex '{choice_name_re}', got {choice.name}"
            )

        # Check that the choice has the expected kwargs in its info_dict
        info_dict = choice.info_dict()
        if "allow_tf32" not in info_dict or info_dict["allow_tf32"] != "True":
            raise ValueError(
                f"Expected allow_tf32=True in choice info_dict, got {info_dict}"
            )

    return choices


class BaseE2ELookupTableTest(TestCase):
    """Base class for E2E lookup table tests with common setup and utilities"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.dev_key = torch._inductor.lookup_table._dev_key(self.device)
        self.original_lookup_table = (
            torch._inductor.lookup_table.kernel_config_lookup_table
        )

    def tearDown(self):
        torch._inductor.lookup_table.kernel_config_lookup_table = (
            self.original_lookup_table
        )
        clear_preprocessing_fns()

    def create_test_tensors(self, operation):
        """Create test tensors based on operation type"""
        if operation == "mm":
            return {
                "tensors": [
                    torch.randn(64, 2048, device=self.device, dtype=torch.float16),
                    torch.randn(2048, 64, device=self.device, dtype=torch.float16),
                ],
                "model": UnifiedModel("mm"),
            }
        elif operation == "addmm":
            return {
                "tensors": [
                    torch.randn(64, 64, device=self.device, dtype=torch.float16),
                    torch.randn(64, 32, device=self.device, dtype=torch.float16),
                    torch.randn(32, 64, device=self.device, dtype=torch.float16),
                ],
                "model": UnifiedModel("addmm"),
            }
        elif operation == "bmm":
            return {
                "tensors": [
                    torch.randn(8, 128, 64, device=self.device, dtype=torch.float16),
                    torch.randn(8, 64, 128, device=self.device, dtype=torch.float16),
                ],
                "model": UnifiedModel("bmm"),
            }
        elif operation == "mm_plus_mm":
            return {
                "tensors": [
                    torch.randn(64, 32, device=self.device, dtype=torch.float16),
                    torch.randn(32, 64, device=self.device, dtype=torch.float16),
                    torch.randn(64, 32, device=self.device, dtype=torch.float16),
                    torch.randn(32, 64, device=self.device, dtype=torch.float16),
                ],
                "model": UnifiedModel("mm_plus_mm"),
            }
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def setup_lookup_table(self, operation, lookup_key, backend_configs):
        """Setup lookup table with given configuration"""
        torch._inductor.lookup_table.kernel_config_lookup_table = {
            self.dev_key: {operation: {lookup_key: backend_configs}}
        }

    def run_compiled_model(self, model, tensors, config_patches=None):
        """Run compiled model with given configuration patches"""
        default_config = {"max_autotune_gemm": True}
        if config_patches:
            default_config.update(config_patches)

        with config.patch(default_config):
            compiled_model = torch.compile(model.to(self.device), mode="max-autotune")
            return compiled_model(*tensors)

    def _test_no_lookup_table_entry(self, operation, expected_regex):
        """Generic test for when there's no lookup table entry"""
        test_data = self.create_test_tensors(operation)

        # Setup lookup table with different key
        self.setup_lookup_table(
            operation,
            "different_lookup_key",
            {"triton": json.dumps({"config": [64, 64, 64, 2, 2], "kwargs": {}})},
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re=expected_regex,
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])

    def _test_valid_lookup_table_entry(self, operation, config_data):
        """Generic test for valid lookup table entry"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        self.setup_lookup_table(
            operation, lookup_key, {"triton": json.dumps(config_data)}
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_",
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])

    def _test_valid_lookup_table_entry_with_kwargs(self, operation, config_data):
        """Generic test for valid lookup table entry with kwargs"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        self.setup_lookup_table(
            operation, lookup_key, {"triton": json.dumps(config_data)}
        )

        add_preprocessing_fn(
            partial(
                _test_kwargs_passed_through,
                choice_name_re="triton_",
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])

    def _test_invalid_lookup_table_entry(self, operation):
        """Generic test for invalid lookup table entry"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        self.setup_lookup_table(
            operation,
            lookup_key,
            {"triton": json.dumps({"invalid_field": "invalid_value"})},
        )

        with self.assertRaises(Exception):
            self.run_compiled_model(test_data["model"], test_data["tensors"])


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestUnifiedLookupTableE2E(BaseE2ELookupTableTest):
    """Unified test class for all lookup table end-to-end scenarios"""

    @parametrize(
        "operation,expected_regex",
        [
            ("mm", "mm"),
            ("addmm", "addmm"),
            ("bmm", "bmm"),
            ("mm_plus_mm", "_mm_plus_mm"),
        ],
    )
    @fresh_cache()
    def test_no_lookup_table_entry(self, operation, expected_regex):
        """Test when there's no lookup table entry for input_nodes()"""
        self._test_no_lookup_table_entry(operation, expected_regex)

    @parametrize(
        "operation,config",
        [
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {}}),
            ("addmm", {"config": [64, 64, 32, 2, 2], "kwargs": {}}),
            ("bmm", {"config": [64, 64, 64, 2, 2], "kwargs": {}}),
            ("mm_plus_mm", {"config": [64, 64, 16, 2, 2], "kwargs": {}}),
        ],
    )
    @fresh_cache()
    def test_valid_lookup_table_entry(self, operation, config):
        """Test when there's a valid entry for input_nodes()"""
        self._test_valid_lookup_table_entry(operation, config)

    @parametrize(
        "operation,config",
        [
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            ("addmm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            ("bmm", {"config": [64, 64, 64, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            (
                "mm_plus_mm",
                {"config": [64, 64, 16, 2, 2], "kwargs": {"ALLOW_TF32": True}},
            ),
        ],
    )
    @fresh_cache()
    def test_valid_lookup_table_entry_with_kwargs(self, operation, config):
        """Test when there's a valid entry with kwargs for input_nodes()"""
        self._test_valid_lookup_table_entry_with_kwargs(operation, config)

    @parametrize("operation", ["mm", "addmm", "bmm", "mm_plus_mm"])
    @fresh_cache()
    def test_invalid_lookup_table_entry(self, operation):
        """Test when there's an invalid entry that fails to parse"""
        self._test_invalid_lookup_table_entry(operation)

    @parametrize("operation", ["mm", "addmm"])
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_valid_tma_lookup_table_entry(self, operation):
        """Test when there's a valid TMA entry for input_nodes()"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        self.setup_lookup_table(
            operation,
            lookup_key,
            {"tma": json.dumps({"config": [64, 64, 32, 2, 2], "kwargs": {}})},
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_mm_persistent_tma_",
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(
            test_data["model"],
            test_data["tensors"],
            {"triton.enable_persistent_tma_matmul": True},
        )

    @fresh_cache()
    def test_mm_valid_decompose_k_lookup_table_entry(self):
        """Test MM when there's a valid decompose_k entry for input_nodes()"""
        test_data = self.create_test_tensors("mm")
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        self.setup_lookup_table(
            "mm", lookup_key, {"decompose_k": json.dumps({"config": [4]})}
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="decompose_k|bmm_dtype",
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])

    @fresh_cache()
    def test_addmm_bias_addmm_lookup_table_entry(self):
        """Test AddMM when there's a bias_addmm entry in the lookup table"""
        # Create bias as column vector with stride[0] == 0 for bias_addmm to be eligible
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        bias_expanded = bias_unexpanded.expand(64, 64)

        test_tensors = [
            bias_expanded,
            torch.randn(64, 32, device=self.device, dtype=torch.float16),
            torch.randn(32, 64, device=self.device, dtype=torch.float16),
        ]

        lookup_key = generate_lookup_key_from_tensors(test_tensors)
        self.setup_lookup_table(
            "addmm", lookup_key, {"bias_addmm": json.dumps({"config": []})}
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="bias_addmm",
                num_expected_choices=1,
            )
        )

        model = UnifiedModel("addmm")
        self.run_compiled_model(
            model,
            [bias_unexpanded, test_tensors[1], test_tensors[2]],
            {"triton.autotune_cublasLt": True},
        )


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableHelperFunctions(BaseE2ELookupTableTest):
    """Test the helper functions _extract_backend_key and _extract_config_dict"""

    def _test_helper_functions_consistency(self, operation, backend_key, config_data):
        """Generic test for helper function consistency"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        # Setup lookup table with the config
        self.setup_lookup_table(
            operation, lookup_key, {backend_key: json.dumps(config_data)}
        )

        # Capture the choice that gets selected
        captured_choice = None

        def capture_choice_preprocessing_fn(choices):
            nonlocal captured_choice
            if captured_choice is None:
                # This is a workaround to avoid capturing the choice multiple
                # times this happens for decompose_k, as we select_algorithm
                # with bmm_dtype again
                assert len(choices) == 1, f"Expected 1 choice, got {len(choices)}"
                captured_choice = choices[0]
            return choices

        add_preprocessing_fn(capture_choice_preprocessing_fn)

        # Run the model to trigger choice selection
        self.run_compiled_model(test_data["model"], test_data["tensors"])

        # Verify we captured a choice
        self.assertIsNotNone(
            captured_choice, "Failed to capture choice during compilation"
        )
        assert captured_choice is not None  # Type narrowing for mypy

        # Test _extract_backend_key
        extracted_backend_key = torch._inductor.lookup_table._extract_backend_key(
            captured_choice
        )
        self.assertEqual(
            extracted_backend_key,
            backend_key,
            f"Backend key mismatch: expected {backend_key}, got {extracted_backend_key}",
        )

        # Test _extract_config_dict
        extracted_config_dict = torch._inductor.lookup_table._extract_config_dict(
            captured_choice
        )
        generated_config_json = json.dumps(extracted_config_dict, sort_keys=True)

        # Get the original config from the lookup table
        lookup_table = torch._inductor.lookup_table._get_lookup_table()
        self.assertIsNotNone(lookup_table, "Lookup table should not be None")
        assert lookup_table is not None  # Type narrowing for mypy
        original_config_json = lookup_table[self.dev_key][operation][lookup_key][
            backend_key
        ]

        # Compare the JSON strings
        self.assertEqual(
            generated_config_json,
            original_config_json,
            f"Config dict mismatch for {backend_key}:\nExpected: {original_config_json}\nGenerated: {generated_config_json}",
        )

    @parametrize(
        "operation,config",
        [
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": False}}),
            (
                "addmm",
                {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": False}},
            ),
            ("bmm", {"config": [64, 64, 64, 2, 2], "kwargs": {"ALLOW_TF32": False}}),
            (
                "mm_plus_mm",
                {"config": [64, 64, 16, 2, 2], "kwargs": {"ALLOW_TF32": False}},
            ),
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            ("addmm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            ("bmm", {"config": [64, 64, 64, 2, 2], "kwargs": {"ALLOW_TF32": True}}),
            (
                "mm_plus_mm",
                {"config": [64, 64, 16, 2, 2], "kwargs": {"ALLOW_TF32": True}},
            ),
        ],
    )
    @fresh_cache()
    def test_triton_helper_functions(self, operation, config):
        """Test helper functions for triton backend"""
        self._test_helper_functions_consistency(operation, "triton", config)

    @parametrize(
        "operation,config_data",
        [
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": False}}),
            ("mm", {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": False}}),
            (
                "addmm",
                {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}},
            ),
            (
                "addmm",
                {"config": [64, 64, 32, 2, 2], "kwargs": {"ALLOW_TF32": True}},
            ),
        ],
    )
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_tma_helper_functions(self, operation, config_data):
        """Test helper functions for TMA backend"""
        # Need to enable TMA for this test
        with config.patch({"triton.enable_persistent_tma_matmul": True}):
            self._test_helper_functions_consistency(operation, "tma", config_data)

    @fresh_cache()
    def test_decompose_k_helper_functions(self):
        """Test helper functions for decompose_k backend"""
        config_data = {"config": [4], "kwargs": {}}
        self._test_helper_functions_consistency("mm", "decompose_k", config_data)

    @fresh_cache()
    def test_bias_addmm_helper_functions(self):
        """Test helper functions for bias_addmm backend"""
        # Create bias as column vector with stride[0] == 0 for bias_addmm to be eligible
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        bias_expanded = bias_unexpanded.expand(64, 64)

        test_tensors = [
            bias_expanded,
            torch.randn(64, 32, device=self.device, dtype=torch.float16),
            torch.randn(32, 64, device=self.device, dtype=torch.float16),
        ]

        lookup_key = generate_lookup_key_from_tensors(test_tensors)
        config_data = {"config": [], "kwargs": {}}

        self.setup_lookup_table(
            "addmm", lookup_key, {"bias_addmm": json.dumps(config_data)}
        )

        # Capture the choice that gets selected
        captured_choice = None

        def capture_choice_preprocessing_fn(choices):
            nonlocal captured_choice
            assert len(choices) == 1, f"Expected 1 choice, got {len(choices)}"
            captured_choice = choices[0]
            return choices

        add_preprocessing_fn(capture_choice_preprocessing_fn)

        # Run the model to trigger choice selection
        model = UnifiedModel("addmm")
        with config.patch({"triton.autotune_cublasLt": True}):
            self.run_compiled_model(
                model, [bias_unexpanded, test_tensors[1], test_tensors[2]]
            )

        # Verify we captured a choice
        self.assertIsNotNone(
            captured_choice, "Failed to capture choice during compilation"
        )
        assert captured_choice is not None  # Type narrowing for mypy

        # Test _extract_backend_key
        extracted_backend_key = torch._inductor.lookup_table._extract_backend_key(
            captured_choice
        )
        self.assertEqual(extracted_backend_key, "bias_addmm")

        # Test _extract_config_dict
        extracted_config_dict = torch._inductor.lookup_table._extract_config_dict(
            captured_choice
        )
        generated_config_json = json.dumps(extracted_config_dict, sort_keys=True)

        # Get the original config from the lookup table
        lookup_table = torch._inductor.lookup_table._get_lookup_table()
        self.assertIsNotNone(lookup_table, "Lookup table should not be None")
        assert lookup_table is not None  # Type narrowing for mypy
        original_config_json = lookup_table[self.dev_key]["addmm"][lookup_key][
            "bias_addmm"
        ]

        # Compare the JSON strings
        self.assertEqual(generated_config_json, original_config_json)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
