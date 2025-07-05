# Owner(s): ["module: inductor"]
import re
import unittest
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch._inductor import config
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA, HAS_GPU
from torch.utils._triton import has_triton_tma_device


class MMModel(nn.Module):
    """Simple model that performs matrix multiplication"""

    def forward(self, a, b):
        return torch.mm(a, b)


class AddMMModel(nn.Module):
    """Simple model that performs additive matrix multiplication"""

    def forward(self, bias, a, b):
        return torch.addmm(bias, a, b)


class BMMModel(nn.Module):
    """Simple model that performs batch matrix multiplication"""

    def forward(self, a, b):
        return torch.bmm(a, b)


class MMPlusMMModel(nn.Module):
    """Simple model that performs two matrix multiplications and adds them"""

    def forward(self, a, b, c, d):
        return torch.mm(a, b) + torch.mm(c, d)


def generate_lookup_key_from_tensors(tensors):
    """Generate lookup key using the same logic as _lookup_key but from real tensors"""
    return str(
        tuple(
            (tensor.dtype, list(tensor.shape), list(tensor.stride()))
            for tensor in tensors
        )
    )


def test_checking_function(
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


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestMMLookupTableE2E(TestCase):
    """Test class for MM method lookup table end-to-end scenarios"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        # Store original value to restore later
        self.original_lookup_table = (
            torch._inductor.lookup_table.gemm_config_lookup_table
        )

        # Create test tensors for MM (K is 32 * larger than M and N)
        self.mm_a = torch.randn(64, 2048, device=self.device, dtype=torch.float16)
        self.mm_b = torch.randn(2048, 64, device=self.device, dtype=torch.float16)
        self.mm_lookup_key = generate_lookup_key_from_tensors([self.mm_a, self.mm_b])

        # Create model
        self.model = MMModel().to(self.device)

    def tearDown(self):
        # Restore original value
        torch._inductor.lookup_table.gemm_config_lookup_table = (
            self.original_lookup_table
        )
        clear_preprocessing_fns()

    @fresh_cache()
    def test_mm_no_lookup_table_entry(self):
        """Test MM when there's no lookup table entry for input_nodes()"""
        # Setup lookup table with no entry for this specific lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm": {
                self.device_name: {
                    "different_lookup_key": {"triton": "(64, 64, 64, 2, 2)"}
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name mm to be chosen here
        add_preprocessing_fn(
            partial(test_checking_function, choice_name_re="mm", num_expected_choices=1)
        )

        # Run the compiled model - should work even without lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.mm_a, self.mm_b)

    @fresh_cache()
    def test_mm_valid_lookup_table_entry(self):
        """Test MM when there's a valid entry for input_nodes()"""
        # Setup pruning table with valid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm": {
                self.device_name: {self.mm_lookup_key: {"triton": "(64, 64, 32, 2, 2)"}}
            }
        }
        # We expect the TritonTemplateCaller to be chosen here as we have a valid config
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="triton_", num_expected_choices=1
            )
        )

        # Run the compiled model with valid lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.mm_a, self.mm_b)

    @fresh_cache()
    def test_mm_invalid_lookup_table_entry(self):
        """Test MM when there's an invalid entry that fails to parse"""
        # Setup pruning table with invalid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm": {
                self.device_name: {
                    self.mm_lookup_key: {"triton": "invalid_config_format"}
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name mm to be chosen here
        add_preprocessing_fn(
            partial(test_checking_function, choice_name_re="mm", num_expected_choices=1)
        )

        # Run the compiled model - should fallback gracefully with invalid config
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.mm_a, self.mm_b)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_mm_valid_tma_lookup_table_entry(self):
        """Test MM when there's a valid TMA entry for input_nodes()"""
        # Setup pruning table with valid tma entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm": {
                self.device_name: {self.mm_lookup_key: {"tma": "(64, 64, 32, 2, 2)"}}
            }
        }
        # We expect the TritonPersistentTemplateCaller to be chosen here as we have a valid TMA config
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="triton_mm_persistent_tma_",
                num_expected_choices=1,
            )
        )

        # Run the compiled model with valid TMA lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
                "triton.enable_persistent_tma_matmul": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.mm_a, self.mm_b)

    @fresh_cache()
    def test_mm_valid_decompose_k_lookup_table_entry(self):
        """Test MM when there's a valid decompose_k entry for input_nodes()"""
        # Setup pruning table with valid decompose_k entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm": {self.device_name: {self.mm_lookup_key: {"decompose_k": "4"}}}
        }
        # we go through bmm for decompose, even though it's not being autotuned
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="decompose_k|bmm_dtype",
                num_expected_choices=1,
            )
        )

        # Run the compiled model with valid decompose_k lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.mm_a, self.mm_b)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestAddMMLookupTableE2E(TestCase):
    """Test class for AddMM method lookup table end-to-end scenarios"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        # Store original value to restore later
        self.original_lookup_table = (
            torch._inductor.lookup_table.gemm_config_lookup_table
        )

        # Create test tensors for AddMM (bias, a, b)
        self.addmm_bias = torch.randn(64, 64, device=self.device, dtype=torch.float16)
        self.addmm_a = torch.randn(64, 32, device=self.device, dtype=torch.float16)
        self.addmm_b = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        self.addmm_lookup_key = generate_lookup_key_from_tensors(
            [self.addmm_bias, self.addmm_a, self.addmm_b]
        )

        # Create model
        self.model = AddMMModel().to(self.device)

    def tearDown(self):
        # Restore original value
        torch._inductor.lookup_table.gemm_config_lookup_table = (
            self.original_lookup_table
        )
        clear_preprocessing_fns()

    @fresh_cache()
    def test_addmm_no_lookup_table_entry(self):
        """Test AddMM when there's no lookup table entry for input_nodes()"""
        # Setup pruning table with no entry for this specific lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "addmm": {
                self.device_name: {
                    "different_lookup_key": {"triton": "(64, 64, 32, 2, 3)"}
                }
            }
        }
        # We expect the ExternChoiceCaller with the default name addmm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="addmm", num_expected_choices=1
            )
        )

        # Run the compiled model - should work even without lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.addmm_bias, self.addmm_a, self.addmm_b)

    @fresh_cache()
    def test_addmm_valid_lookup_table_entry(self):
        """Test AddMM when there's a valid entry for input_nodes()"""
        # Setup pruning table with valid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "addmm": {
                self.device_name: {
                    self.addmm_lookup_key: {"triton": "(64, 64, 32, 2, 2)"}
                }
            }
        }

        # We expect the TritonTemplateCaller to be chosen here as we have a valid config
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="triton_", num_expected_choices=1
            )
        )

        # Run the compiled model with valid lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.addmm_bias, self.addmm_a, self.addmm_b)

    @fresh_cache()
    def test_addmm_invalid_lookup_table_entry(self):
        """Test AddMM when there's an invalid entry that fails to parse"""
        # Setup pruning table with invalid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "addmm": {
                self.device_name: {
                    self.addmm_lookup_key: {"triton": ""}  # Empty string
                }
            }
        }

        # We expect the ExternChoiceCaller with the default name addmm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="addmm", num_expected_choices=1
            )
        )

        # Run the compiled model - should fallback gracefully with invalid config
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.addmm_bias, self.addmm_a, self.addmm_b)

    @fresh_cache()
    def test_addmm_bias_addmm_lookup_table_entry(self):
        """Test AddMM when there's a bias_addmm entry in the lookup table"""
        # Create bias as column vector with stride[0] == 0 for bias_addmm to be eligible
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        bias_expanded = bias_unexpanded.expand(64, 64)

        # Calculate lookup key with expanded bias (stride[0] == 0)
        bias_addmm_lookup_key = generate_lookup_key_from_tensors(
            [bias_expanded, self.addmm_a, self.addmm_b]
        )

        # Setup pruning table with bias_addmm entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "addmm": {self.device_name: {bias_addmm_lookup_key: {"bias_addmm": "1"}}}
        }

        # We expect the ExternChoiceCaller with the bias_addmm name to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="bias_addmm",
                num_expected_choices=1,
            )
        )

        # Run the compiled model with bias_addmm lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
                "triton.autotune_cublasLt": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            # Pass the unexpanded bias to the model
            _ = compiled_model(bias_unexpanded, self.addmm_a, self.addmm_b)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_addmm_valid_tma_lookup_table_entry(self):
        """Test AddMM when there's a valid TMA entry for input_nodes()"""
        # Setup pruning table with valid tma entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "addmm": {
                self.device_name: {self.addmm_lookup_key: {"tma": "(64, 64, 32, 2, 2)"}}
            }
        }

        # We expect the TritonPersistentTemplateCaller to be chosen here as we have a valid TMA config
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="triton_mm_persistent_tma_",
                num_expected_choices=1,
            )
        )

        # Run the compiled model with valid TMA lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
                "triton.enable_persistent_tma_matmul": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.addmm_bias, self.addmm_a, self.addmm_b)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestBMMLookupTableE2E(TestCase):
    """Test class for BMM method lookup table end-to-end scenarios"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        # Store original value to restore later
        self.original_lookup_table = (
            torch._inductor.lookup_table.gemm_config_lookup_table
        )

        # Create test tensors for BMM (batch matrix multiplication)
        self.bmm_a = torch.randn(8, 128, 64, device=self.device, dtype=torch.float16)
        self.bmm_b = torch.randn(8, 64, 128, device=self.device, dtype=torch.float16)
        self.bmm_lookup_key = generate_lookup_key_from_tensors([self.bmm_a, self.bmm_b])

        # Create model
        self.model = BMMModel().to(self.device)

    def tearDown(self):
        # Restore original value
        torch._inductor.lookup_table.gemm_config_lookup_table = (
            self.original_lookup_table
        )
        clear_preprocessing_fns()

    @fresh_cache()
    def test_bmm_no_lookup_table_entry(self):
        """Test BMM when there's no lookup table entry for input_nodes()"""
        # Setup pruning table with no entry for this specific lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "bmm": {
                self.device_name: {
                    "different_lookup_key": {"triton": "(64, 64, 64, 2, 2)"}
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name bmm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="bmm", num_expected_choices=1
            )
        )

        # Run the compiled model - should work even without lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.bmm_a, self.bmm_b)

    @fresh_cache()
    def test_bmm_valid_lookup_table_entry(self):
        """Test BMM when there's a valid entry for input_nodes()"""
        # Setup pruning table with valid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "bmm": {
                self.device_name: {
                    self.bmm_lookup_key: {"triton": "(64, 64, 64, 2, 2)"}
                }
            }
        }

        # We expect the TritonTemplateCaller to be chosen here as we have a valid config
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="triton_", num_expected_choices=1
            )
        )

        # Run the compiled model with valid lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.bmm_a, self.bmm_b)

    @fresh_cache()
    def test_bmm_invalid_lookup_table_entry(self):
        """Test BMM when there's an invalid entry that fails to parse"""
        # Setup pruning table with invalid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "bmm": {
                self.device_name: {
                    self.bmm_lookup_key: {
                        "triton": "(256, 256, 128, 1"
                    }  # Missing closing parenthesis
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name bmm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="bmm", num_expected_choices=1
            )
        )

        # Run the compiled model - should fallback gracefully with invalid config
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(self.bmm_a, self.bmm_b)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA, "CUDA not available")
class TestMMPlusMMMLookupTableE2E(TestCase):
    """Test class for MM_PLUS_MM method lookup table end-to-end scenarios"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        # Store original value to restore later
        self.original_lookup_table = (
            torch._inductor.lookup_table.gemm_config_lookup_table
        )

        # Create test tensors for MM_PLUS_MM (four matrices for two mm operations)
        self.mm_plus_mm_a = torch.randn(64, 32, device=self.device, dtype=torch.float16)
        self.mm_plus_mm_b = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        self.mm_plus_mm_c = torch.randn(64, 32, device=self.device, dtype=torch.float16)
        self.mm_plus_mm_d = torch.randn(32, 64, device=self.device, dtype=torch.float16)
        self.mm_plus_mm_lookup_key = generate_lookup_key_from_tensors(
            [self.mm_plus_mm_a, self.mm_plus_mm_b, self.mm_plus_mm_c, self.mm_plus_mm_d]
        )

        # Create model
        self.model = MMPlusMMModel().to(self.device)

    def tearDown(self):
        # Restore original value
        torch._inductor.lookup_table.gemm_config_lookup_table = (
            self.original_lookup_table
        )
        clear_preprocessing_fns()

    @fresh_cache()
    def test_mm_plus_mm_no_lookup_table_entry(self):
        """Test MM_PLUS_MM when there's no lookup table entry for input_nodes()"""
        # Setup pruning table with no entry for this specific lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm_plus_mm": {
                self.device_name: {
                    "different_lookup_key": {"triton": "(64, 64, 32, 2, 2)"}
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name mm_plus_mm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="_mm_plus_mm",
                num_expected_choices=1,
            )
        )

        # Run the compiled model - should work even without lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(
                self.mm_plus_mm_a,
                self.mm_plus_mm_b,
                self.mm_plus_mm_c,
                self.mm_plus_mm_d,
            )

    @fresh_cache()
    def test_mm_plus_mm_valid_lookup_table_entry(self):
        """Test MM_PLUS_MM when there's a valid entry for input_nodes()"""
        # Setup pruning table with valid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm_plus_mm": {
                self.device_name: {
                    # important here that K != 32, as that will make the TritonTemplate not eligible
                    self.mm_plus_mm_lookup_key: {"triton": "(64, 64, 16, 2, 2)"}
                }
            }
        }

        # We expect the TritonTemplateCaller to be chosen here as we have a valid config
        add_preprocessing_fn(
            partial(
                test_checking_function, choice_name_re="triton_", num_expected_choices=1
            )
        )

        # Run the compiled model with valid lookup table entry
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(
                self.mm_plus_mm_a,
                self.mm_plus_mm_b,
                self.mm_plus_mm_c,
                self.mm_plus_mm_d,
            )

    @fresh_cache()
    def test_mm_plus_mm_invalid_lookup_table_entry(self):
        """Test MM_PLUS_MM when there's an invalid entry that fails to parse"""
        # Setup pruning table with invalid triton entry for this lookup key
        torch._inductor.lookup_table.gemm_config_lookup_table = {
            "mm_plus_mm": {
                self.device_name: {
                    self.mm_plus_mm_lookup_key: {
                        "triton": "(abc, def, ghi, jkl, mno)"
                    }  # Non-numeric values
                }
            }
        }

        # we expect the ExternChoiceCaller with the default name mm_plus_mm to be chosen here
        add_preprocessing_fn(
            partial(
                test_checking_function,
                choice_name_re="_mm_plus_mm",
                num_expected_choices=1,
            )
        )

        # Run the compiled model - should fallback gracefully with invalid config
        with config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            compiled_model = torch.compile(self.model, mode="max-autotune")
            _ = compiled_model(
                self.mm_plus_mm_a,
                self.mm_plus_mm_b,
                self.mm_plus_mm_c,
                self.mm_plus_mm_d,
            )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
