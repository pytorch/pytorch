# Owner(s): ["oncall: distributed checkpointing"]

from concurrent.futures import Future
from unittest import skipIf

import torch
from torch.distributed.checkpoint._experimental.staging import (
    CheckpointStagerConfig,
    DefaultStager,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class _StagerTestMixin:
    def setUp(self) -> None:
        super().setUp()
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
            "tensor": torch.randn(3, 4),
            "nested": {"inner_tensor": torch.ones(2, 2), "inner_value": 42},
        }


class TestDefaultStagerGeneric(_StagerTestMixin, TestCase):
    hw_classification = HardwareClassification.GENERIC

    @skipIf(
        torch.accelerator.is_available(),
        reason="requires no available accelerator",
    )
    def test_non_blocking_without_accelerator(self) -> None:
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_non_blocking_copy=True,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "Non-blocking copy requires that the current accelerator is available",
        ):
            DefaultStager(options)

    def test_different_option_combinations(self) -> None:
        test_cases = (
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
        )

        for options in test_cases:
            with self.subTest(options=options):
                stager = DefaultStager(options)
                staged_dict = stager.stage(self.state_dict)
                self.assertIsInstance(staged_dict, dict)
                self.assertIn("model", staged_dict)
                stager.close()

    def test_resource_cleanup(self) -> None:
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_non_blocking_copy=False,
        )
        stager = DefaultStager(options)
        self.assertIsNotNone(stager._state_dict_stager)
        stager.close()

    def test_multiple_staging_operations(self) -> None:
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_non_blocking_copy=False,
        )
        stager = DefaultStager(options)
        state_dicts = [
            {"model1": torch.nn.Linear(5, 3).state_dict()},
            {"model2": torch.nn.Conv2d(3, 16, 3).state_dict()},
            {"optimizer": {"lr": 0.001, "momentum": 0.9}},
        ]

        staged_results = [stager.stage(state_dict) for state_dict in state_dicts]

        self.assertEqual(len(staged_results), 3)
        for result, state_dict in zip(staged_results, state_dicts):
            self.assertIsInstance(result, dict)
            for key in state_dict:
                self.assertIn(key, result)
        stager.close()


class TestDefaultStagerAccelerator(_StagerTestMixin, TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_accelerator_tensors_staging(self, device) -> None:
        state_dict = {
            "accelerator_tensor": torch.randn(3, 4, device=device),
            "cpu_tensor": torch.randn(2, 3),
            "mixed_model": {
                "weight": torch.randn(5, 5, device=device),
                "bias": torch.randn(5, device=device),
            },
        }
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_non_blocking_copy=False,
        )
        stager = DefaultStager(options)

        staged_dict = stager.stage(state_dict)
        self.assertIsInstance(staged_dict, dict)
        self.assertIn("accelerator_tensor", staged_dict)
        self.assertIn("cpu_tensor", staged_dict)
        self.assertIn("mixed_model", staged_dict)
        self.assertEqual(staged_dict["accelerator_tensor"].device.type, "cpu")
        self.assertEqual(staged_dict["cpu_tensor"].device.type, "cpu")
        self.assertEqual(staged_dict["mixed_model"]["weight"].device.type, "cpu")
        self.assertEqual(staged_dict["mixed_model"]["bias"].device.type, "cpu")
        stager.close()


class TestDefaultStagerStreamAccelerator(_StagerTestMixin, TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_async_staging(self, device) -> None:
        state_dict = {"tensor": torch.randn(3, 4, device=device)}
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=True,
            use_non_blocking_copy=False,
        )
        stager = DefaultStager(options)

        result = stager.stage(state_dict)
        self.assertIsInstance(result, Future)
        staged_dict = result.result()
        self.assertEqual(staged_dict["tensor"].device.type, "cpu")
        stager.close()


class TestDefaultStagerCUDA(_StagerTestMixin, TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_sync_staging(self, device) -> None:
        """Test synchronous staging with the original optimized defaults."""
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=False))

        staged_dict = stager.stage(self.state_dict)

        self.assertIsInstance(staged_dict, dict)
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)
        self.assertIn("tensor", staged_dict)
        self.assertIn("nested", staged_dict)
        stager.close()


instantiate_device_type_tests(
    TestDefaultStagerAccelerator,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestDefaultStagerStreamAccelerator,
    globals(),
    only_for=("cuda", "xpu"),
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestDefaultStagerCUDA,
    globals(),
    only_for=("cuda",),
)

if __name__ == "__main__":
    run_tests()
