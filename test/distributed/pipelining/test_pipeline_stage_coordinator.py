# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Unit tests for PipelineStageCoordinator module.

This module tests the pipeline stage coordinator interface that provides
device-specific coordination for pipeline parallelism stages.
"""

from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining.pipeline_stage_coordinator import (
    _register_default_stage_coordinator,
    get_pipeline_stage_coordinator,
    PipelineStageCoordinator,
    PipelineStageCoordinatorRegistry,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestPipelineStageCoordinator(TestCase):
    """Test cases for PipelineStageCoordinator base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.group = Mock(spec=dist.ProcessGroup)
        self.coordinator = PipelineStageCoordinator(self.device, self.group)

    def test_init(self):
        """Test PipelineStageCoordinator initialization."""
        self.assertEqual(self.coordinator._device, self.device)
        self.assertEqual(self.coordinator._group, self.group)

    def test_create_stage_tensor_metadata(self):
        """Test creating tensor metadata from tensor."""
        tensor = torch.randn(2, 3, 4, dtype=torch.float32, device=self.device)
        metadata = self.coordinator.create_stage_tensor_metadata(tensor)

        self.assertEqual(metadata.shape, tensor.shape)
        self.assertEqual(metadata.dtype, tensor.dtype)
        self.assertEqual(metadata.device, torch.device("meta"))

    def test_create_stage_communication_buffer(self):
        """Test creating communication buffer from metadata."""
        metadata = torch.empty(2, 3, 4, dtype=torch.float32, device="meta")
        target_device = torch.device("cpu")

        buffer = self.coordinator.create_stage_communication_buffer(
            metadata, target_device
        )

        self.assertEqual(buffer.shape, metadata.shape)
        self.assertEqual(buffer.dtype, metadata.dtype)
        self.assertEqual(buffer.device, target_device)

    def test_infer_pipeline_stage_outputs_single_tensor(self):
        """Test inferring pipeline stage outputs for single tensor output."""

        # Create a simple module that outputs a single tensor
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        submod = SimpleModule()
        input_metadata = (torch.empty(3, 4, device="meta"),)

        outputs_meta = self.coordinator.infer_pipeline_stage_outputs(
            submod, input_metadata
        )

        self.assertIsInstance(outputs_meta, tuple)
        self.assertEqual(len(outputs_meta), 1)
        self.assertEqual(outputs_meta[0].shape, torch.Size([3, 2]))
        self.assertEqual(outputs_meta[0].device, torch.device("meta"))

    def test_infer_pipeline_stage_outputs_multiple_tensors(self):
        """Test inferring pipeline stage outputs for multiple tensor outputs."""

        # Create a module that outputs multiple tensors
        class MultiOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 2)
                self.linear2 = nn.Linear(4, 3)

            def forward(self, x):
                return self.linear1(x), self.linear2(x)

        submod = MultiOutputModule()
        input_metadata = (torch.empty(3, 4, device="meta"),)

        outputs_meta = self.coordinator.infer_pipeline_stage_outputs(
            submod, input_metadata
        )

        self.assertIsInstance(outputs_meta, tuple)
        self.assertEqual(len(outputs_meta), 2)
        self.assertEqual(outputs_meta[0].shape, torch.Size([3, 2]))
        self.assertEqual(outputs_meta[1].shape, torch.Size([3, 3]))

    def test_infer_pipeline_stage_outputs_with_kwargs(self):
        """Test inferring pipeline stage outputs with keyword arguments."""

        class ModuleWithKwargs(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x, scale=1.0):
                return self.linear(x) * scale

        submod = ModuleWithKwargs()
        input_metadata = (torch.empty(3, 4, device="meta"),)
        kwargs = {"scale": 2.0}

        outputs_meta = self.coordinator.infer_pipeline_stage_outputs(
            submod, input_metadata, kwargs
        )

        self.assertIsInstance(outputs_meta, tuple)
        self.assertEqual(len(outputs_meta), 1)
        self.assertEqual(outputs_meta[0].shape, torch.Size([3, 2]))


class TestPipelineStageCoordinatorRegistry(TestCase):
    """Test cases for PipelineStageCoordinatorRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = PipelineStageCoordinatorRegistry()
        self.device = torch.device("cpu")
        self.group = Mock(spec=dist.ProcessGroup)

    def test_init(self):
        """Test registry initialization."""
        self.assertEqual(len(self.registry._registry), 0)
        self.assertIsNone(self.registry._default_coordinator_creator)

    def test_create_coordinator_cpu(self):
        """Test creating coordinator for cpu."""

        def custom_creator(device, group):
            return PipelineStageCoordinator(device, group)

        device = torch.device("cpu")
        self.registry.register(device, custom_creator)

        coordinator = self.registry.create_coordinator(device, self.group)

        self.assertIsInstance(coordinator, PipelineStageCoordinator)
        self.assertEqual(coordinator._device, device)
        self.assertEqual(coordinator._group, self.group)

    def test_create_coordinator_not_implemented_error(self):
        """Test creating coordinator for unregistered device without default should raise error."""
        device = torch.device("cpu")

        with self.assertRaises(NotImplementedError):
            self.registry.create_coordinator(device, self.group)


class TestPipelineStageCoordinatorIntegration(TestCase):
    """Integration tests for pipeline stage coordinator."""

    def setUp(self):
        """Set up test fixtures with fake distributed environment."""
        # Initialize fake distributed environment
        self.store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=1, store=self.store)
        self.device = torch.device("cpu")
        self.group = dist.group.WORLD

    def tearDown(self):
        """Clean up distributed environment."""
        dist.destroy_process_group()

    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with coordinator."""
        # Register default stage coordinator
        _register_default_stage_coordinator()

        # Get coordinator
        coordinator = get_pipeline_stage_coordinator(self.device, self.group)

        # Create test tensor
        tensor = torch.randn(2, 3, 4)

        # Create metadata
        metadata = coordinator.create_stage_tensor_metadata(tensor)
        self.assertEqual(metadata.shape, tensor.shape)
        self.assertEqual(metadata.device, torch.device("meta"))

        # Create communication buffer
        buffer = coordinator.create_stage_communication_buffer(metadata, self.device)
        self.assertEqual(buffer.shape, tensor.shape)
        self.assertEqual(buffer.device, self.device)

        # Test inference
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        submod = TestModule()
        input_metadata = (metadata,)

        outputs_meta = coordinator.infer_pipeline_stage_outputs(submod, input_metadata)
        self.assertIsInstance(outputs_meta, tuple)
        self.assertEqual(len(outputs_meta), 1)


if __name__ == "__main__":
    run_tests()
