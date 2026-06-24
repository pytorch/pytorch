"""
Test suite for SmartSummary functionality
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path (kept if running standalone out-of-tree)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.qol.smartsummary import SmartSummary
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfNoCuda


class SimpleConvNet(nn.Module):
    """Simple CNN for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for testing nested structures"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ComplexNet(nn.Module):
    """More complex network with residual connections"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        self.fc = nn.Linear(64 * 16 * 16, 1000)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestSmartSummary(TestCase):

    def test_basic_summary(self):
        """Test basic SmartSummary functionality"""
        model = SimpleConvNet()
        summary = SmartSummary(model, input_size=(3, 32, 32), batch_size=2)
        summary.show(show_bottlenecks=False)
        
        self.assertGreater(summary.total_params, 0, "Total params should be > 0")
        self.assertGreater(summary.trainable_params, 0, "Trainable params should be > 0")
        self.assertGreater(len(summary.summary_data), 0, "Should have layer information")

    def test_without_input_size(self):
        """Test summary without forward pass"""
        model = SimpleConvNet()
        summary = SmartSummary(model)  # No input_size
        
        self.assertGreater(summary.total_params, 0, "Should count params even without forward pass")
        self.assertGreater(summary.trainable_params, 0)

    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        model = SimpleConvNet()
        summary = SmartSummary(model, input_size=(3, 32, 32))
        
        bottlenecks = summary.get_bottlenecks(top_n=3)
        self.assertGreater(len(bottlenecks), 0, "Should detect at least one bottleneck")
        self.assertTrue(all('score' in bn for bn in bottlenecks), "Each bottleneck should have a score")

    def test_complex_model(self):
        """Test with complex nested model"""
        model = ComplexNet()
        summary = SmartSummary(model, input_size=(3, 64, 64))
        summary.show()
        
        self.assertGreater(summary.total_params, 0, "Complex model should have parameters")

    def test_export_functionality(self):
        """Test export to dict and file securely using tempfile"""
        model = SimpleConvNet()
        summary = SmartSummary(model, input_size=(3, 32, 32))
        
        # Test to_dict
        summary_dict = summary.to_dict()
        self.assertIn("layers", summary_dict, "Dictionary should contain layers")
        self.assertIn("total_params", summary_dict, "Dictionary should contain total_params")
        self.assertIn("bottlenecks", summary_dict, "Dictionary should contain bottlenecks")
        
        # Safe handling of file export within PyTorch test sandbox
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            test_file = tmp.name
            
        try:
            summary.save_to_file(test_file)
            self.assertTrue(os.path.exists(test_file), "Summary file should be created")
            with open(test_file, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    @skipIfNoCuda
    def test_cuda_compatibility(self):
        """Test CUDA device compatibility"""
        model = SimpleConvNet().cuda()
        summary = SmartSummary(model, input_size=(3, 32, 32), device="cuda")
        self.assertGreater(summary.total_params, 0, "Should work on CUDA device")

    def test_gradient_tracking(self):
        """Test gradient variance tracking"""
        model = SimpleConvNet()
        summary = SmartSummary(
            model, 
            input_size=(3, 32, 32), 
            batch_size=2,
            track_gradients=True
        )
        
        self.assertGreater(len(summary.gradient_stats), 0, "Should have gradient statistics")
        for layer, stats in list(summary.gradient_stats.items())[:3]:
            self.assertIn('grad_variance', stats)
            self.assertIn('grad_mean', stats)
            self.assertIn('grad_max', stats)


if __name__ == "__main__":
    run_tests()
