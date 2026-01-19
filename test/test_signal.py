"""Tests for torch.signal savgol."""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.signal.windows.savgol import savgol, savgol_coeffs


class TestSavgol(TestCase):
    def test_savgol_basic(self):
        """Test basic Savitzky-Golay filter functionality."""
        x = torch.randn(100) + torch.linspace(0, 10, 100)
        smoothed = savgol(x, window_length=11, polyorder=3)
        
        self.assertEqual(smoothed.shape, x.shape)
        self.assertEqual(smoothed.dtype, x.dtype)
    
    def test_savgol_coeffs_basic(self):
        """Test basic coefficient computation."""
        coeffs = savgol_coeffs(window_length=11, polyorder=3)
        
        self.assertEqual(coeffs.shape, (11,))
        self.assertEqual(coeffs.dtype, torch.float32)
    
    def test_savgol_different_window_lengths(self):
        """Test with different window lengths."""
        x = torch.randn(100)
        
        for window_length in [5, 7, 11, 15]:
            smoothed = savgol(x, window_length=window_length, polyorder=3)
            self.assertEqual(smoothed.shape, x.shape)
    
    def test_savgol_different_polyorders(self):
        """Test with different polynomial orders."""
        x = torch.randn(100)
        
        for polyorder in [1, 2, 3, 4]:
            smoothed = savgol(x, window_length=11, polyorder=polyorder)
            self.assertEqual(smoothed.shape, x.shape)
    
    def test_savgol_derivative(self):
        """Test derivative computation."""
        x = torch.linspace(0, 10, 100)
        # Test first derivative
        deriv1 = savgol(x, window_length=11, polyorder=3, deriv=1)
        self.assertEqual(deriv1.shape, x.shape)
    
    def test_savgol_modes(self):
        """Test different padding modes."""
        x = torch.randn(100)
        modes = ["mirror", "constant", "nearest", "interp", "reflect", "replicate"]
        
        for mode in modes:
            smoothed = savgol(x, window_length=11, polyorder=3, mode=mode)
            self.assertEqual(smoothed.shape, x.shape)
    
    def test_savgol_2d(self):
        """Test with 2D input along different axes."""
        x = torch.randn(50, 100)
        
        # Apply along axis 0
        smoothed_0 = savgol(x, window_length=11, polyorder=3, axis=0)
        self.assertEqual(smoothed_0.shape, x.shape)
        
        # Apply along axis 1
        smoothed_1 = savgol(x, window_length=11, polyorder=3, axis=1)
        self.assertEqual(smoothed_1.shape, x.shape)
    
    def test_savgol_dtypes(self):
        """Test with different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(100, dtype=dtype)
            smoothed = savgol(x, window_length=11, polyorder=3)
            self.assertEqual(smoothed.dtype, dtype)
    
    def test_savgol_cuda(self):
        """Test on CUDA if available."""
        if not torch.cuda.is_available():
            return
        
        x = torch.randn(100, device='cuda')
        smoothed = savgol(x, window_length=11, polyorder=3)
        
        self.assertEqual(smoothed.device, x.device)
        self.assertEqual(smoothed.shape, x.shape)
    
    def test_savgol_coeffs_device(self):
        """Test coefficient computation with device specification."""
        coeffs_cpu = savgol_coeffs(window_length=11, polyorder=3, device='cpu')
        self.assertEqual(coeffs_cpu.device.type, 'cpu')
        
        if torch.cuda.is_available():
            coeffs_cuda = savgol_coeffs(window_length=11, polyorder=3, device='cuda')
            self.assertEqual(coeffs_cuda.device.type, 'cuda')
    
    def test_savgol_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        x = torch.randn(100)
        
        # polyorder >= window_length should raise error
        with self.assertRaises(ValueError):
            savgol_coeffs(window_length=5, polyorder=5)
        
        # Invalid mode should raise error
        with self.assertRaises(ValueError):
            savgol(x, window_length=11, polyorder=3, mode='invalid')
        
        # window_length > signal length with interp mode should raise error
        with self.assertRaises(ValueError):
            savgol(x[:5], window_length=11, polyorder=3, mode='interp')


if __name__ == "__main__":
    run_tests()
