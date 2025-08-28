# Owner(s): ["module: inductor"]
import re
import unittest
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.lookup_table.core import _dev_key
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
    ExternKernelCaller,
    TritonTemplateCaller,
)
from torch._inductor.test_case import run_tests
from torch._inductor.utils import fresh_cache, get_num_sms, TMA_DESCRIPTOR_SIZE
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON, HAS_GPU
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tma_device

from .test_lookup_table import BaseLookupTableTest, MockMMKernelInputs


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
        self.dev_key = _dev_key(self.device)
        self.original_lookup_table = inductor_config.template_config_lookup_table.table
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        inductor_config.template_config_lookup_table.table = self.original_lookup_table
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
        inductor_config.template_config_lookup_table.table = {flat_key: configs}

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
                "ALLOW_TF32": False,
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
                "ALLOW_TF32": False,
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
                "ALLOW_TF32": False,
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
                "ALLOW_TF32": False,
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


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
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
                assert len(choices) > 2, (
                    f"Max-autotune should have >2 choices, got {len(choices)}"
                )
                assert any(isinstance(c, ExternKernelCaller) for c in choices), (
                    "Should have ExternKernelCaller"
                )
                assert any(isinstance(c, TritonTemplateCaller) for c in choices), (
                    "Should have TritonTemplateCaller"
                )
            else:
                assert len(choices) == 1, (
                    f"No max-autotune should have 1 choice, got {len(choices)}"
                )
                assert isinstance(choices[0], ExternKernelCaller), (
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

    @fresh_cache()
    def test_decompose_k_lookup_table_entry(self):
        """Test decompose_k template entry"""
        tensors = self.create_tensors("mm", m=32, n=32, k=32 * 32)
        config = self.create_basic_config(
            torch._inductor.kernel.mm.decompose_k_subgraph_template.uid
        )

        self.setup_lookup_table("mm", tensors, [config])
        add_preprocessing_fn(
            partial(
                verify_choice_names, pattern="decompose_k|bmm_dtype", expected_count=1
            )
        )
        self.run_model("mm", tensors)

    @fresh_cache()
    def test_bias_addmm_lookup_table_entry(self):
        """Test bias_addmm template entry"""
        # Create bias with stride[0] == 0 for bias_addmm eligibility
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        bias_expanded = bias_unexpanded.expand(64, 64)
        tensors = [
            bias_expanded,
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
        with inductor_config.patch(
            {"template_config_lookup_table.check_src_hash": True}
        ):
            self.run_model("mm", tensors)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
