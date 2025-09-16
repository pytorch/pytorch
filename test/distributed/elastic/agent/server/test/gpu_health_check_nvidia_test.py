#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for NVIDIA-style GPU health check functionality.

This script demonstrates the usage of the GPU health check system
based on the NVIDIA resiliency extension implementation.
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from torch.distributed.elastic.utils.gpu_health_check import (
    GPUHealthCheck,
    PynvmlMixin,
    create_gpu_health_check,
    quick_gpu_health_check,
)
from torch.distributed.elastic.agent.server.health_check_server import (
    GPUHealthCheckServer,
    create_gpu_healthcheck_server,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class GPUHealthCheckTest(TestCase):
    """Test cases for GPU health check functionality."""

    def test_gpu_health_check_basic(self):
        """Test basic GPU health check functionality."""
        # Create a GPU health check instance
        gpu_check = create_gpu_health_check(
            device_index=None,  # Check all GPUs
            interval=30,
            on_failure=lambda: print("GPU health check failed!")
        )
        
        # Test basic properties
        self.assertIsInstance(gpu_check.enabled, bool)
        self.assertIsInstance(gpu_check.pynvml_available, bool)
        self.assertEqual(gpu_check.device_index, None)
        self.assertEqual(gpu_check.interval, 30)
        
        if gpu_check.enabled:
            # Perform synchronous health check
            result = gpu_check()
            self.assertIsInstance(result, bool)
            
            # Test quick health check
            quick_result = quick_gpu_health_check()
            self.assertIsInstance(quick_result, bool)


    def test_pynvml_mixin(self):
        """Test PynvmlMixin functionality."""
        mixin = PynvmlMixin()
        pynvml_available = mixin.check_pynvml_availability()
        self.assertIsInstance(pynvml_available, bool)
        
        if pynvml_available:
            # Test GB200 platform detection
            is_gb200 = mixin.is_gb200_platform()
            self.assertIsInstance(is_gb200, bool)
            
            if is_gb200:
                mapping = mixin.get_gb200_static_mapping()
                self.assertIsInstance(mapping, dict)
                # Check that mapping has expected structure
                for gpu_id, nic_name in mapping.items():
                    self.assertIsInstance(gpu_id, int)
                    self.assertIsInstance(nic_name, str)
            

    def test_gpu_health_check_server(self):
        """Test GPU health check server functionality."""
        # Mock alive callback
        def mock_alive_callback():
            return int(time.time())
        
        # Create GPU health check server
        server = create_gpu_healthcheck_server(
            alive_callback=mock_alive_callback,
            port=8080,
            timeout=30,
            gpu_device_index=None,  # Monitor all GPUs
            gpu_check_interval=30,
            enable_gpu_monitoring=True,
        )
        
        # Test server properties
        self.assertEqual(server._port, 8080)
        self.assertEqual(server._timeout, 30)
        self.assertEqual(server._gpu_device_index, None)
        self.assertEqual(server._gpu_check_interval, 30)
        self.assertTrue(server._gpu_monitoring_enabled)
        
        # Start the server
        server.start()
        
        # Get GPU health status
        gpu_health = server.get_gpu_health_status()
        self.assertIsInstance(gpu_health, dict)
        
        if "error" in gpu_health:
            self.assertIn("error", gpu_health)
        else:
            self.assertIn("enabled", gpu_health)
            self.assertIn("device_index", gpu_health)
            self.assertIn("is_healthy", gpu_health)
            self.assertIn("pynvml_available", gpu_health)
            self.assertIn("timestamp", gpu_health)
        
        # Get overall health summary
        summary = server.get_health_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("process_health", summary)
        self.assertIn("gpu_health", summary)
        self.assertIn("overall_healthy", summary)
        self.assertIn("timestamp", summary)
        
        # Stop the server
        server.stop()


    def test_async_functionality(self):
        """Test async functionality (without actually running async)."""
        gpu_check = create_gpu_health_check(
            device_index=0,  # Check specific GPU
            interval=10,
        )
        
        # Test that async methods exist
        self.assertTrue(hasattr(gpu_check, 'async_check'))
        self.assertTrue(hasattr(gpu_check, '_check_health'))
        
        # Test that async_check is callable
        self.assertTrue(callable(gpu_check.async_check))
        self.assertTrue(callable(gpu_check._check_health))


if __name__ == "__main__":
    run_tests()
