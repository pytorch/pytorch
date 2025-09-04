# Owner(s): ["module: inductor"]
import random
import unittest
from typing import Any, Union

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.performance_model.choices import PerformanceModelChoices
from torch._inductor.performance_model.registry import (
    clear_registry,
    register_performance_model_fn,
)
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


# Custom exception to ensure we don't compile choices during tests
class CompilationBlockerException(Exception):
    pass


def mock_performance_model_function(choices: list[Any], op_name: str) -> list[Any]:
    """
    Mock performance model function that ranks all choices randomly.

    Args:
        choices: List of KernelTemplateChoice objects
        op_name: Operation name (e.g., "mm")

    Returns:
        Same list of choices with performance_prediction field populated for all
    """
    # Rank all choices randomly
    for i in range(len(choices)):
        # Assign random performance prediction (lower is better)
        choices[i].performance_prediction = random.uniform(0.1, 10.0)

    return choices


def compilation_blocker_validator(choices: list[Any]) -> list[Any]:
    """
    Validator that raises an exception to prevent actual compilation.
    This ensures we're only testing choice generation, not compilation.
    """
    raise CompilationBlockerException(
        f"Compilation blocked as expected. Generated {len(choices)} choices."
    )


class SimpleModel(nn.Module):
    """Simple model that performs matrix multiplication"""

    def forward(self, a, b):
        return torch.mm(a, b)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support performance model interface")
@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestPerformanceModelInterfaceE2E(TestCase):
    """E2E tests for performance model interface functionality"""

    def setUp(self):
        torch._dynamo.reset()
        self.device = torch.device("cuda")
        # Clear any existing performance model registrations
        clear_registry()
        # Set the performance model choices handler
        V.set_choices_handler(PerformanceModelChoices())

    def tearDown(self):
        # Clear performance model registry
        clear_registry()
        # Clear any preprocessing functions
        clear_preprocessing_fns()
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())

    def get_device_name(self):
        """Get proper device name using same logic as KernelInputs"""
        if self.device.type == "cuda":
            device_properties = torch.cuda.get_device_properties(self.device)
            return device_properties.gcnArchName
        return self.device.type

    def create_tensors(self, m=1024, n=1024, k=512):
        """Create test tensors for mm operation"""
        A = torch.randn(m, k, device=self.device, dtype=torch.float16)
        B = torch.randn(k, n, device=self.device, dtype=torch.float16)
        return [A, B]

    def register_mock_performance_models(self, include_aten=False):
        """Register mock performance model functions for mm templates"""
        hardware_name = self.get_device_name()
        clear_registry()
        # Register for triton templates (mm and mm_persistent_tma)
        triton_template_uids = [
            torch._inductor.kernel.mm.mm_template.uid,
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid,
        ]
        if include_aten:
            triton_template_uids.append(torch._inductor.kernel.mm.aten_mm.uid)
        for template_uid in triton_template_uids:
            register_performance_model_fn(
                mock_performance_model_function,
                template_id=template_uid,
                op="mm",
                hardware_name=hardware_name,
            )

    def run_model(self, tensors, config_patches=None):
        """Run compiled model with configuration"""

        model = SimpleModel()
        with inductor_config.patch(config_patches):
            compiled_model = torch.compile(model.to(self.device))
            return compiled_model(*tensors)

    @parametrize("discard_unranked", [True, False])
    @parametrize("topk", [100, 200])
    @fresh_cache()
    def test_performance_model_choice_filtering(self, discard_unranked, topk):
        """
        Test that performance model interface correctly filters choices.

        - Registers mock performance model for triton templates only (not aten)
        - Verifies choice count: should be topk+1 or topk depending on discard_unranked
        - The +1 comes from aten_mm which we don't register a model for
        """
        # Register mock performance models (triton only, not aten)
        self.register_mock_performance_models(include_aten=False)

        # Create test tensors
        tensors = self.create_tensors(m=1024, n=1024, k=512)

        # Setup validation function to check choice counts
        def validate_choice_counts(choices, topk=topk):
            num_choices = len(choices)

            if discard_unranked:
                # Should have exactly topk choices (all triton ones ranked)
                expected_count = topk
                if num_choices != expected_count:
                    raise AssertionError(
                        f"Expected {expected_count} choices with discard_unranked=True, topk={topk}, "
                        f"but got {num_choices}"
                    )
            else:
                # Should have topk+1 choices (topk triton + 1 unranked aten)
                expected_count = topk + 1
                if num_choices != expected_count:
                    raise AssertionError(
                        f"Expected {expected_count} choices with discard_unranked=False, topk={topk}, "
                        f"but got {num_choices}"
                    )

            # Block compilation to prevent actual kernel execution
            raise CompilationBlockerException(
                f"Validation passed. Generated {num_choices} choices as expected."
            )

        add_preprocessing_fn(validate_choice_counts)

        # Run model with performance model configuration
        config_patches = {
            "max_autotune_gemm": True,
            "max_autotune": True,
            "performance_model.topk": topk,
            "performance_model.discard_unranked": discard_unranked,
            "max_autotune_gemm_backends": "ATEN,TRITON",
        }

        # Expect CompilationBlockerException to be raised during validation
        with self.assertRaisesRegex(Exception, r"CompilationBlockerException"):
            self.run_model(tensors, config_patches)

    @fresh_cache()
    def test_topk_minus_one_behavior(self):
        """
        Test special -1 topk behavior:
        1. First run without any registered model functions - should get normal choices
        2. Then register models for all templates and set topk=-1 - choices should match
        """
        clear_registry()
        # Create test tensors
        tensors = self.create_tensors(m=1024, n=1024, k=512)

        # First test: no registered models, intercept choice count
        choices_without_models: list[Union[None, int]] = [None]

        def intercept_choices_no_models(choices):
            nonlocal choices_without_models
            choices_without_models[0] = len(choices)
            # Block compilation to prevent actual kernel execution
            raise CompilationBlockerException(
                f"No models test: Generated {len(choices)} choices as expected."
            )

        add_preprocessing_fn(intercept_choices_no_models)

        # Run without any registered performance models
        config_patches = {
            "max_autotune_gemm": True,
            "max_autotune": True,
            "performance_model.topk": 100,  # Regular topk for baseline
            "performance_model.discard_unranked": False,
            "max_autotune_gemm_backends": "ATEN,TRITON",
        }

        with self.assertRaisesRegex(Exception, r"CompilationBlockerException"):
            self.run_model(tensors, config_patches)

        baseline_choice_count = choices_without_models[0]

        torch._dynamo.reset()
        clear_preprocessing_fns()

        # Second test: register models for ALL templates and set topk=-1
        self.register_mock_performance_models(include_aten=True)

        choices_with_topk_minus_one: list[Union[int, None]] = [None]

        def intercept_choices_topk_minus_one(choices):
            nonlocal choices_with_topk_minus_one
            choices_with_topk_minus_one[0] = len(choices)
            # Block compilation to prevent actual kernel execution
            raise CompilationBlockerException(
                f"Topk=-1 test: Generated {len(choices)} choices as expected."
            )

        add_preprocessing_fn(intercept_choices_topk_minus_one)

        # Run with all models registered and topk=-1
        config_patches_minus_one = {
            "max_autotune_gemm": True,
            "max_autotune": True,
            "performance_model.topk": -1,  # Special -1 behavior
            "performance_model.discard_unranked": False,
            "max_autotune_gemm_backends": "ATEN,TRITON",
        }

        with self.assertRaisesRegex(Exception, r"CompilationBlockerException"):
            self.run_model(tensors, config_patches_minus_one)

        topk_minus_one_choice_count = choices_with_topk_minus_one[0]

        # The choice counts should match when topk=-1 with all models registered
        if baseline_choice_count != topk_minus_one_choice_count:
            raise AssertionError(
                f"Expected topk=-1 to give same count as no models: "
                f"baseline={baseline_choice_count}, topk=-1={topk_minus_one_choice_count}"
            )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_CUDA_AND_TRITON and is_big_gpu():
        run_tests()
