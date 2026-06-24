"""
Simple test to verify TrainingMonitor works correctly
"""

import os
import tempfile
import torch

from torch.utils.qol.trainingmonitor import TrainingMonitor, get_backend
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfNoCuda


class TestTrainingMonitor(TestCase):

    def test_basic_functionality(self):
        """Test basic TrainingMonitor functionality"""
        backend = get_backend()
        self.assertIsNotNone(backend)
        
        # Create dummy data
        data = list(range(10))
        
        # Securely manage the log file context using a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log_path = os.path.join(tmpdir, "test_log.csv")
            
            monitor = TrainingMonitor(data, desc="Testing", log_file=test_log_path)
            
            for i, item in enumerate(monitor):
                # Simulate some metrics
                monitor.log({
                    'loss': 1.0 / (i + 1),  # Decreasing loss
                    'accuracy': i / 10.0    # Increasing accuracy
                })
            
            # Verify the log file was actually created and populated
            self.assertTrue(os.path.exists(test_log_path), "Log file should be created")
            self.assertGreater(os.path.getsize(test_log_path), 0, "Log file should not be empty")

    @skipIfNoCuda
    def test_cuda_detection(self):
        """Test CUDA detection natively via the PyTorch harness"""
        self.assertTrue(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0)
        self.assertIsNotNone(device_name)


if __name__ == "__main__":
    run_tests()
