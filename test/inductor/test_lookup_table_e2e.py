# Owner(s): ["module: inductor"]
import re
import unittest
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.lookup_table import _dev_key, lookup_key_suffix
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, get_num_sms, TMA_DESCRIPTOR_SIZE
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


class BaseE2ELookupTableTest(TestCase):
    """Base class for E2E lookup table tests with common setup and utilities"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.dev_key = _dev_key(self.device)
        self.original_lookup_table = inductor_config.template_lookup_table

    def tearDown(self):
        inductor_config.template_lookup_table = self.original_lookup_table
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
        # following the logic inside torch._inductor.lookup_table.make_lookup_key

        flat_key = f"{self.dev_key}+{operation}+{lookup_key}+{lookup_key_suffix()}"
        inductor_config.template_lookup_table = {flat_key: backend_configs}

    def run_compiled_model(self, model, tensors, config_patches=None):
        """Run compiled model with given configuration patches"""
        default_config = {"max_autotune_gemm": True}
        if config_patches:
            default_config.update(config_patches)

        with inductor_config.patch(default_config):
            compiled_model = torch.compile(model.to(self.device), mode="max-autotune")
            return compiled_model(*tensors)

    def _test_no_lookup_table_entry(self, operation, expected_regex):
        """Generic test for when there's no lookup table entry"""
        test_data = self.create_test_tensors(operation)

        # Setup lookup table with different key
        self.setup_lookup_table(
            operation,
            "different_lookup_key",
            [
                {
                    "template_id": "mm",
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 64,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            ],
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
            operation,
            lookup_key,
            [{"template_id": config_data["template_id"], **config_data}],
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_",
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])

    def _test_invalid_lookup_table_entry(self, operation):
        """Generic test for invalid lookup table entry"""
        test_data = self.create_test_tensors(operation)
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])
        tid = operation
        if operation == "addmm":
            # They use the same template
            tid = "mm"
        self.setup_lookup_table(
            operation,
            lookup_key,
            [{"template_id": tid, "invalid_field": "invalid_value"}],
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
            (
                "mm",
                {
                    "template_id": "mm",
                    "template_hash": torch._inductor.kernel.mm.mm_template.src_hash,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                },
            ),
            (
                "mm",
                {
                    "template_id": "mm",
                    # NOTE: This this one uses no template_hash and therefore will not check
                    # and should always pass. This is the user saying "I don't care about the hash"
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                },
            ),
            (
                "addmm",
                {
                    "template_id": "mm",
                    # NOTE: This is a template hash for mm_template, if this test
                    # breaks, update it with the hash in the log
                    # TODO(coconutruben): once the templates are easy to import, read this out of the template
                    "template_hash": torch._inductor.kernel.mm.mm_template.src_hash,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                },
            ),
            (
                "bmm",
                {
                    "template_id": "bmm",
                    "template_hash": torch._inductor.kernel.bmm.bmm_template.src_hash,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 64,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                },
            ),
            (
                "mm_plus_mm",
                {
                    "template_id": "mm_plus_mm",
                    "template_hash": torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.src_hash,
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 16,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                },
            ),
        ],
    )
    @fresh_cache()
    def test_valid_lookup_table_entry(self, operation, config):
        """Test when there's a valid entry for input_nodes()"""
        self._test_valid_lookup_table_entry(operation, config)

    # Note: no mm_plus_mm test here as mm_plus_mm will gracefully
    # avoid Triton if BLOCK_K is missing in the config, so we'll
    # avoid a very convoluted test for this behavior
    @parametrize("operation", ["mm", "addmm", "bmm"])
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
        from torch.utils._triton import has_triton_stable_tma_api

        self.setup_lookup_table(
            operation,
            lookup_key,
            [
                {
                    "template_id": "mm_persistent_tma",
                    # NOTE: This is a template hash for mm_persistent_tma, if this test
                    # breaks, update it with the hash in the log
                    # TODO(coconutruben): once the templates are easy to import, read this out of the template
                    "template_hash": "88ec6cbe557df819512c09fa9094e91d1c631130be236800fa695acabfc96996",
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                    # TMA-specific fields that persistent_mm_options() would add
                    "A_ROW_MAJOR": True,
                    "B_ROW_MAJOR": True,
                    "NUM_SMS": get_num_sms(),
                    "TMA_SIZE": TMA_DESCRIPTOR_SIZE,  # TMA_DESCRIPTOR_SIZE
                    "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),  # From tma_options()
                }
            ],
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
            "mm", lookup_key, [{"template_id": "decompose_k", "k": 4}]
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
        # Note: bias_addmm requires an empty single config as it does not have params
        self.setup_lookup_table("addmm", lookup_key, [{"template_id": "bias_addmm"}])

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

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_multiple_configs_same_template(self):
        """Test when there are multiple TMA configurations for the same template"""
        test_data = self.create_test_tensors("mm")
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        from torch.utils._triton import has_triton_stable_tma_api

        config1 = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_stages": 3,
            "num_warps": 8,
            "EVEN_K": True,
            "ALLOW_TF32": False,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
            # TMA-specific fields
            "A_ROW_MAJOR": True,
            "B_ROW_MAJOR": True,
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,  # TMA_DESCRIPTOR_SIZE
            "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
        }
        config2 = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_stages": 4,
            "num_warps": 4,
            "EVEN_K": True,
            "ALLOW_TF32": False,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
            # TMA-specific fields
            "A_ROW_MAJOR": True,
            "B_ROW_MAJOR": True,
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,  # TMA_DESCRIPTOR_SIZE
            "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
        }

        self.setup_lookup_table(
            "mm",
            lookup_key,
            [
                {"template_id": "mm_persistent_tma", **config1},
                {"template_id": "mm_persistent_tma", **config2},
            ],
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_mm_persistent_tma_",
                num_expected_choices=2,  # Expect 2 TMA choices now
            )
        )

        self.run_compiled_model(
            test_data["model"],
            test_data["tensors"],
            {"triton.enable_persistent_tma_matmul": True},
        )

    @fresh_cache()
    def test_two_configs_one_per_template(self):
        """Test when there are two configs, one per template (triton and tma)"""
        test_data = self.create_test_tensors("mm")
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        triton_config = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_stages": 3,
            "num_warps": 8,
            "EVEN_K": True,
            "ALLOW_TF32": False,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
        }

        tma_config = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_stages": 4,
            "num_warps": 4,
            "EVEN_K": True,
            "ALLOW_TF32": False,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
            "GROUP_M": 8,
            "A_ROW_MAJOR": True,
            "B_ROW_MAJOR": True,
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,  # TMA_DESCRIPTOR_SIZE
            "TMA_EXPERIMENTAL_API": True,
        }

        self.setup_lookup_table(
            "mm",
            lookup_key,
            [
                {"template_id": "mm", **triton_config},
                {"template_id": "mm_persistent_tma", **tma_config},
            ],
        )

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_",  # Both triton and tma templates should match this pattern
                num_expected_choices=2,  # Expect 2 choices: one triton, one tma
            )
        )

        self.run_compiled_model(
            test_data["model"],
            test_data["tensors"],
            {"triton.enable_persistent_tma_matmul": True},
        )

    @fresh_cache()
    def test_three_configs_mixed_templates(self):
        """Test autotuning considers all configurations provided in lookup table"""
        test_data = self.create_test_tensors("mm")
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        # Setup with 3 different configurations: 2 for triton, 1 for tma
        triton_configs = [
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "ALLOW_TF32": False,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 4,
                "num_warps": 4,
                "EVEN_K": True,
                "ALLOW_TF32": False,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
        ]

        tma_configs = [
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "num_stages": 4,
                "num_warps": 4,
                "EVEN_K": True,
                "ALLOW_TF32": False,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
                "A_ROW_MAJOR": True,
                "B_ROW_MAJOR": True,
                "NUM_SMS": get_num_sms(),
                "TMA_SIZE": TMA_DESCRIPTOR_SIZE,  # TMA_DESCRIPTOR_SIZE
                "TMA_EXPERIMENTAL_API": True,
            }
        ]

        # Convert to new flattened structure with template_id
        config_list = []
        for config in triton_configs:
            config_list.append({"template_id": "mm", **config})
        for config in tma_configs:
            config_list.append({"template_id": "mm_persistent_tma", **config})

        self.setup_lookup_table("mm", lookup_key, config_list)

        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="triton_",  # Should match both triton and tma templates
                num_expected_choices=3,  # Expect exactly 3 choices
            )
        )

        self.run_compiled_model(
            test_data["model"],
            test_data["tensors"],
            {"triton.enable_persistent_tma_matmul": True},
        )

    @fresh_cache()
    def test_mm_invalid_template_hash_fallback_to_aten(self):
        """Test MM with valid config but invalid template_hash falls back to aten"""
        test_data = self.create_test_tensors("mm")
        lookup_key = generate_lookup_key_from_tensors(test_data["tensors"])

        # Setup valid mm config but with invalid template_hash
        self.setup_lookup_table(
            "mm",
            lookup_key,
            [
                {
                    "template_id": "mm",
                    "template_hash": "rubbish",  # Invalid hash should cause filtering
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            ],
        )

        # Should fall back to aten mm since config gets filtered out
        add_preprocessing_fn(
            partial(
                _post_test_checking_function,
                choice_name_re="mm",  # Should match aten mm fallback
                num_expected_choices=1,
            )
        )

        self.run_compiled_model(test_data["model"], test_data["tensors"])


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
