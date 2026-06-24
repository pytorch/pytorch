"""
Unit tests for PyTorch utility functions
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.qol.utils import (
    lazy_flatten,
    get_flatten_size,
    loss_ncc,
    ncc_score,
    find_lr,
    LRFinder
)
from torch.testing._internal.common_utils import TestCase, run_tests


class TestQOLUtils(TestCase):

    def test_lazy_flatten(self):
        """Test lazy_flatten function"""
        # Test 1: 4D tensor (batch, channels, height, width)
        x = torch.randn(32, 16, 7, 7)
        x_flat = lazy_flatten(x)
        self.assertEqual(x_flat.shape, (32, 16 * 7 * 7))
        
        # Test 2: 3D tensor
        x = torch.randn(16, 10, 5)
        x_flat = lazy_flatten(x)
        self.assertEqual(x_flat.shape, (16, 50))
        
        # Test 3: Different start_dim
        x = torch.randn(8, 3, 224, 224)
        x_flat = lazy_flatten(x, start_dim=2)
        self.assertEqual(x_flat.shape, (8, 3, 224 * 224))
        
        # Test 4: Already flattened
        x = torch.randn(10, 100)
        x_flat = lazy_flatten(x)
        self.assertEqual(x_flat.shape, (10, 100))

    def test_get_flatten_size(self):
        """Test get_flatten_size function"""
        # Test 1: Conv output shape
        size = get_flatten_size((16, 7, 7))
        self.assertEqual(size, 784)
        
        # Test 2: Single dimension
        size = get_flatten_size((1024,))
        self.assertEqual(size, 1024)
        
        # Test 3: Large shape
        size = get_flatten_size((512, 14, 14))
        self.assertEqual(size, 100352)

    def test_loss_ncc(self):
        """Test NCC loss function"""
        # Test 1: Identical tensors (should give loss ~0)
        y_true = torch.randn(8, 1, 64, 64)
        y_pred = y_true.clone()
        loss = loss_ncc(y_true, y_pred)
        self.assertLess(loss.item(), 1e-4)
        
        # Test 2: Completely different tensors
        y_true = torch.randn(4, 1, 32, 32)
        y_pred = torch.randn(4, 1, 32, 32)
        loss = loss_ncc(y_true, y_pred)
        self.assertTrue(0 <= loss.item() <= 2)
        
        # Test 3: Gradient flow
        y_true = torch.randn(2, 1, 16, 16)
        y_pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        loss = loss_ncc(y_true, y_pred)
        loss.backward()
        self.assertIsNotNone(y_pred.grad)
        
        # Test 4: Multi-channel images
        y_true = torch.randn(4, 3, 64, 64)
        y_pred = torch.randn(4, 3, 64, 64)
        loss = loss_ncc(y_true, y_pred)
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss))
        
        # Test 5: Scaled version (same pattern, different scale)
        y_true = torch.randn(2, 1, 32, 32)
        y_pred = y_true * 2.0
        loss = loss_ncc(y_true, y_pred)
        self.assertLess(loss.item(), 0.1)

    def test_ncc_score(self):
        """Test NCC score function"""
        # Test 1: Identical tensors (should give score ~1)
        y_true = torch.randn(4, 1, 32, 32)
        y_pred = y_true.clone()
        score = ncc_score(y_true, y_pred)
        self.assertGreater(score.item(), 0.999)
        
        # Test 2: Score range
        y_true = torch.randn(2, 1, 16, 16)
        y_pred = torch.randn(2, 1, 16, 16)
        score = ncc_score(y_true, y_pred)
        self.assertTrue(-1 <= score.item() <= 1)

    def test_lr_finder(self):
        """Test LRFinder class"""
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        train_data = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 5, (100,))
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Test 1: Initialize LRFinder
        lr_finder = LRFinder(model, optimizer, criterion, device='cpu')
        self.assertIsNotNone(lr_finder)
        
        # Test 2: Run range test
        lr_finder.range_test(train_loader, start_lr=1e-5, end_lr=1.0, num_iter=20)
        self.assertGreater(len(lr_finder.lrs), 0)
        self.assertGreater(len(lr_finder.losses), 0)
        
        # Test 3: Get best LR
        best_lr = lr_finder.get_best_lr()
        self.assertIsNotNone(best_lr)
        self.assertGreater(best_lr, 0)

    def test_find_lr(self):
        """Test find_lr convenience function"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        train_data = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_lr = find_lr(
            model, optimizer, criterion, train_loader,
            device='cpu', num_iter=15, plot=False
        )
        self.assertIsNotNone(best_lr)
        self.assertGreater(best_lr, 0)

    def test_integration_conv_to_linear(self):
        """Integration test: Using lazy_flatten in a real model"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                
                flat_size = get_flatten_size((32, 8, 8))
                self.fc = nn.Linear(flat_size, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = lazy_flatten(x)
                return self.fc(x)
        
        model = TestModel()
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (4, 10))

    def test_ncc_medical_imaging(self):
        """Integration test: NCC for medical imaging scenario"""
        original = torch.randn(1, 1, 128, 128)
        transformed = original + torch.randn(1, 1, 128, 128) * 0.1
        
        loss = loss_ncc(original, transformed)
        score = ncc_score(original, transformed)
        
        self.assertTrue(0 <= loss.item() <= 2)
        self.assertTrue(-1 <= score.item() <= 1)


if __name__ == "__main__":
    run_tests()
