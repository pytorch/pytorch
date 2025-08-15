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
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON, HAS_GPU
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tma_device


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


def create_lookup_key_from_tensors(tensors):
    """Generate lookup key from tensors"""
    # normalize out when shape = 1 that stride on that dimension is 0
    components = []
    for c in [[t.dtype, list(t.shape), list(t.stride())] for t in tensors]:
        d, s, st = c
        for i, (x, y) in enumerate(zip(s, st)):
            if x == 1:
                st[i] = 0
        components.append((d, s, st))
    return str(tuple(components))


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


class BaseE2ELookupTableTest(TestCase):
    """Base class for E2E lookup table tests"""

    def setUp(self):
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.dev_key = _dev_key(self.device)
        self.original_lookup_table = inductor_config.template_lookup_table

    def tearDown(self):
        inductor_config.template_lookup_table = self.original_lookup_table
        clear_preprocessing_fns()

    def create_tensors(self, operation, b=8, m=64, n=64, k=32, decompose_k=False):
        """Create test tensors for operations with configurable dimensions"""
        if decompose_k:
            k = 32 * max(m, n)  # decompose_k needs k to be 32x larger than m or n

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
        lookup_key = create_lookup_key_from_tensors(tensors)
        flat_key = f"{self.dev_key}+{operation}+{lookup_key}+{lookup_key_suffix()}"
        inductor_config.template_lookup_table = {flat_key: configs}

    def run_model(self, operation, tensors, config_patches=None):
        """Run compiled model with configuration"""
        config = {"max_autotune_gemm": True}
        if config_patches:
            config.update(config_patches)

        model = UnifiedModel(operation)
        with inductor_config.patch(config):
            compiled_model = torch.compile(model.to(self.device), mode="max-autotune")
            return compiled_model(*tensors)

    def create_basic_config(self, template_id, include_hash=True):
        """Create basic configuration for template"""
        configs = {
            "mm": {
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
                "template_hash": torch._inductor.kernel.mm.mm_template.src_hash,
                "template_id": torch._inductor.kernel.mm.mm_template.name,
            },
            "mm_plus_mm": {
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
                "template_hash": torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.src_hash,
                "template_id": torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.name,
            },
            "bmm": {
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
                "template_hash": torch._inductor.kernel.bmm.bmm_template.src_hash,
                "template_id": torch._inductor.kernel.bmm.bmm_template.name,
            },
            "mm_persistent_tma": {
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
                "template_hash": torch._inductor.kernel.mm.persistent_tma_mm_template.src_hash,
                "template_id": torch._inductor.kernel.mm.persistent_tma_mm_template.name,
            },
            "bias_addmm": {"template_id": "bias_addmm"},
            "decompose_k": {"template_id": "decompose_k", "k": 4},
        }
        c = configs.get(template_id)
        if not include_hash:
            del c["template_hash"]
        return c


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support lookup table")
@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableE2E(BaseE2ELookupTableTest):
    """E2E tests for lookup table functionality"""

    @fresh_cache()
    def test_no_lookup_table_entry(self):
        """Test when there's no lookup table entry - uses default autotune"""
        tensors = self.create_tensors("mm")

        # Setup lookup table with different key
        self.setup_lookup_table("mm", [torch.randn(1, 1, device=self.device)], [])

        add_preprocessing_fn(
            partial(verify_choice_names, pattern="mm", expected_count=1)
        )
        self.run_model("mm", tensors)

    @parametrize("operation", ["mm", "addmm", "bmm", "mm_plus_mm"])
    @fresh_cache()
    def test_valid_lookup_table_entry(self, operation):
        """Test when there's a valid entry for the operation"""
        k = 256 if operation == "mm_plus_mm" else 64
        tensors = self.create_tensors(operation, k=k)
        template_id = "mm" if operation in ["mm", "addmm"] else operation
        config = self.create_basic_config(template_id)

        self.setup_lookup_table(operation, tensors, [config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=1)
        )
        self.run_model(operation, tensors)

    @fresh_cache()
    def test_invalid_lookup_table_entry(self):
        """Test when there's an invalid entry that fails to parse"""
        tensors = self.create_tensors("mm")
        invalid_config = {"template_id": "mm", "invalid_field": "invalid_value"}

        self.setup_lookup_table("mm", tensors, [invalid_config])
        with self.assertRaises(Exception):
            self.run_model("mm", tensors)

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @parametrize("operation", ["mm", "addmm"])
    @fresh_cache()
    def test_tma_lookup_table_entry(self, operation):
        """Test TMA template entry"""
        tensors = self.create_tensors(operation)
        config = self.create_basic_config("mm_persistent_tma")

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
        config = self.create_basic_config("decompose_k")

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

        config = self.create_basic_config("bias_addmm")
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

        config1 = self.create_basic_config("mm_persistent_tma")
        config1.update({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8})

        config2 = self.create_basic_config("mm_persistent_tma")
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

        triton_config = self.create_basic_config("mm")
        triton_config.update({"BLOCK_M": 128, "num_warps": 8})

        tma_config = self.create_basic_config("mm_persistent_tma")
        tma_config.update({"BLOCK_M": 256, "num_warps": 4})

        self.setup_lookup_table("mm", tensors, [triton_config, tma_config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=2)
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})

    @fresh_cache()
    def test_mm_without_template_hash_still_works(self):
        """Test mm operation where config has no template_hash - should still work"""
        tensors = self.create_tensors("mm")

        # Create config without template_hash
        config = self.create_basic_config("mm", include_hash=False)

        self.setup_lookup_table("mm", tensors, [config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=1)
        )
        self.run_model("mm", tensors)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
