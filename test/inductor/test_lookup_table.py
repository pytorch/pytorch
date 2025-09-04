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
from torch._inductor.lookup_table.core import (
    _dev_key,
    lookup_key_suffix,
    lookup_template_configs,
)
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
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON, HAS_GPU
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tma_device


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
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=2)
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
