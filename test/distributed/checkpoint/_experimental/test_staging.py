#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from concurrent.futures import Future

from torch.distributed.checkpoint._experimental.staging import (
    CheckpointStager,
    CheckpointStagerConfig,
    DefaultStager,
)
from torch.distributed.checkpoint._experimental.types import STATE_DICT


class MockStager(CheckpointStager):
    """Mock implementation of CheckpointStager for testing."""

    def __init__(self):
        self.staged_data = None
        self.closed = False

    def stage(self, state_dict: STATE_DICT, **kwargs) -> STATE_DICT:
        # Store kwargs for potential testing, but don't use them in this simple mock
        _ = kwargs  # Acknowledge kwargs parameter
        self.staged_data = state_dict.copy()
        return self.staged_data

    def close(self) -> None:
        self.closed = True


class TestCheckpointStager(unittest.TestCase):
    """Test the abstract CheckpointStager interface."""

    def test_abstract_interface(self):
        """Test that CheckpointStager cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            CheckpointStager()

    def test_mock_implementation(self):
        """Test that a concrete implementation works correctly."""
        stager = MockStager()
        test_dict = {"test": "value"}

        result = stager.stage(test_dict)
        self.assertEqual(result, test_dict)
        self.assertEqual(stager.staged_data, test_dict)

        stager.close()
        self.assertTrue(stager.closed)


class TestCheckpointStagerConfig(unittest.TestCase):
    def test_default_options(self):
        """Test that default options are set correctly."""
        options = CheckpointStagerConfig()
        self.assertTrue(options.use_pinned_memory)
        self.assertTrue(options.use_shared_memory)
        self.assertTrue(options.use_async_staging)
        self.assertTrue(options.use_cuda_non_blocking_copy)

    def test_custom_options(self):
        """Test that custom options are set correctly."""
        options = CheckpointStagerConfig(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_cuda_non_blocking_copy=False,
        )
        self.assertFalse(options.use_pinned_memory)
        self.assertFalse(options.use_shared_memory)
        self.assertFalse(options.use_async_staging)
        self.assertFalse(options.use_cuda_non_blocking_copy)

    def test_mixed_options(self):
        """Test various combinations of options."""
        options = CheckpointStagerConfig(
            use_pinned_memory=True,
            use_shared_memory=False,
            use_async_staging=True,
            use_cuda_non_blocking_copy=False,
        )
        self.assertTrue(options.use_pinned_memory)
        self.assertFalse(options.use_shared_memory)
        self.assertTrue(options.use_async_staging)
        self.assertFalse(options.use_cuda_non_blocking_copy)


class TestDefaultStager(unittest.TestCase):
    def setUp(self):
        # Create a test state dictionary with various data types
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
            "tensor": torch.randn(3, 4),
            "nested": {
                "inner_tensor": torch.ones(2, 2),
                "inner_value": 42
            }
        }

    def test_sync_staging(self):
        """Test synchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Stage the state dict
        staged_dict = stager.stage(self.state_dict)

        # Verify that a state dict is returned (not a Future)
        self.assertIsInstance(staged_dict, dict)

        # Verify the staged state dictionary
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)
        self.assertIn("tensor", staged_dict)
        self.assertIn("nested", staged_dict)

        # Clean up
        stager.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_async_staging(self):
        """Test asynchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=True)
        stager = DefaultStager(options)

        # Stage the state dict
        result = stager.stage(self.state_dict)

        # Verify that a Future is returned
        self.assertIsInstance(result, Future)

        # Wait for the Future to complete
        staged_dict = result.result()

        # Verify the staged state dictionary
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)

        # Clean up
        stager.close()

    def test_cuda_non_blocking_without_cuda(self):
        """Test that non-blocking copy fails when CUDA is not available."""
        if torch.cuda.is_available():
            self.skipTest("CUDA is available, cannot test CUDA unavailable scenario")

        options = CheckpointStagerConfig(use_cuda_non_blocking_copy=True)
        with self.assertRaises(AssertionError):
            DefaultStager(options)

    def test_different_option_combinations(self):
        """Test various combinations of staging options."""
        test_cases = [
            # All disabled
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=False,
                use_async_staging=False,
                use_cuda_non_blocking_copy=False,
            ),
            # Only pinned memory
            CheckpointStagerConfig(
                use_pinned_memory=True,
                use_shared_memory=False,
                use_async_staging=False,
                use_cuda_non_blocking_copy=False,
            ),
            # Only shared memory
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_cuda_non_blocking_copy=False,
            ),
        ]

        for options in test_cases:
            with self.subTest(options=options):
                stager = DefaultStager(options)

                # Test staging works with these options
                if options.use_async_staging and torch.cuda.is_available():
                    result = stager.stage(self.state_dict)
                    self.assertIsInstance(result, Future)
                    staged_dict = result.result()
                else:
                    staged_dict = stager.stage(self.state_dict)

                self.assertIsInstance(staged_dict, dict)
                self.assertIn("model", staged_dict)

                stager.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tensors_staging(self):
        """Test staging with CUDA tensors."""
        # Create state dict with CUDA tensors
        cuda_state_dict = {
            "cuda_tensor": torch.randn(3, 4).cuda(),
            "cpu_tensor": torch.randn(2, 3),
            "mixed_model": {
                "weight": torch.randn(5, 5).cuda(),
                "bias": torch.randn(5).cuda(),
            }
        }

        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        staged_dict = stager.stage(cuda_state_dict)

        # Verify tensors are staged (should be moved to CPU)
        self.assertIn("cuda_tensor", staged_dict)
        self.assertIn("cpu_tensor", staged_dict)
        self.assertIn("mixed_model", staged_dict)

        stager.close()

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Verify initial state
        self.assertIsNotNone(stager._state_dict_stager)

        # Close and verify cleanup
        stager.close()

        # Verify that the state_dict_stager's close method was called
        # (We can't directly test this without mocking, but we can ensure no exceptions)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_async_resource_cleanup(self):
        """Test that async resources are properly cleaned up."""
        options = CheckpointStagerConfig(use_async_staging=True)
        stager = DefaultStager(options)

        # Verify executor is created
        self.assertIsNotNone(stager._staging_executor)

        # Submit a task
        future = stager.stage(self.state_dict)
        result = future.result()
        self.assertIsInstance(result, dict)

        # Close and verify cleanup
        stager.close()

        # Verify executor is shut down (should not accept new tasks)
        self.assertTrue(stager._staging_executor._shutdown)

    def test_empty_state_dict(self):
        """Test staging with empty state dictionary."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        empty_dict = {}
        staged_dict = stager.stage(empty_dict)

        self.assertEqual(staged_dict, {})
        stager.close()

    def test_large_state_dict(self):
        """Test staging with a larger state dictionary."""
        # Create a larger model for testing
        large_model = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )

        large_state_dict = {
            "large_model": large_model.state_dict(),
            "large_tensor": torch.randn(1000, 1000),
            "metadata": {"layers": 5, "params": sum(p.numel() for p in large_model.parameters())}
        }

        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        staged_dict = stager.stage(large_state_dict)

        self.assertIn("large_model", staged_dict)
        self.assertIn("large_tensor", staged_dict)
        self.assertIn("metadata", staged_dict)

        stager.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_stream_synchronization(self):
        """Test that CUDA stream synchronization works correctly."""
        options = CheckpointStagerConfig(
            use_async_staging=True,
            use_cuda_non_blocking_copy=True
        )
        stager = DefaultStager(options)

        # Verify stream is created for async staging
        self.assertIsNotNone(stager._staging_stream)

        # Create CUDA tensors
        cuda_state_dict = {
            "tensor1": torch.randn(100, 100).cuda(),
            "tensor2": torch.randn(50, 50).cuda(),
        }

        future = stager.stage(cuda_state_dict)
        staged_dict = future.result()

        self.assertIn("tensor1", staged_dict)
        self.assertIn("tensor2", staged_dict)

        stager.close()

    def test_multiple_staging_operations(self):
        """Test multiple staging operations with the same stager."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Stage multiple different state dicts
        state_dicts = [
            {"model1": torch.nn.Linear(5, 3).state_dict()},
            {"model2": torch.nn.Conv2d(3, 16, 3).state_dict()},
            {"optimizer": {"lr": 0.001, "momentum": 0.9}},
        ]

        staged_results = []
        for state_dict in state_dicts:
            staged_dict = stager.stage(state_dict)
            staged_results.append(staged_dict)

        # Verify all staging operations succeeded
        self.assertEqual(len(staged_results), 3)
        for i, result in enumerate(staged_results):
            self.assertIsInstance(result, dict)
            # Verify the result contains the expected keys
            for key in state_dicts[i].keys():
                self.assertIn(key, result)

        stager.close()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_concurrent_async_staging(self):
        """Test multiple concurrent async staging operations."""
        options = CheckpointStagerConfig(use_async_staging=True)
        stager = DefaultStager(options)

        # Submit multiple staging operations
        futures = []
        for i in range(3):
            state_dict = {f"model_{i}": torch.nn.Linear(10, 5).state_dict()}
            future = stager.stage(state_dict)
            futures.append(future)

        # Wait for all to complete
        results = [f.result() for f in futures]

        # Verify all completed successfully
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertIn(f"model_{i}", result)

        stager.close()


if __name__ == "__main__":
    unittest.main()
