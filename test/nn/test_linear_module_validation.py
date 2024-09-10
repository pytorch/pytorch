import unittest
import torch
from torch.nn import Parameter
from torch.nn.modules.linear import Linear, Bilinear

class TestLinearValidation(unittest.TestCase):
    """Test that the Linear module correctly validates the inputs."""
    def test_invalid_in_features_type(self):
        """Test that TypeError is raised for invalid in_features type."""
        with self.assertRaises(TypeError):
            Linear(in_features=10.5, out_features=5)  # Invalid type float
    def test_invalid_out_features_type(self):
        """Test that TypeError is raised for invalid out_features type."""
        with self.assertRaises(TypeError):
            Linear(in_features=10, out_features="five")  # Invalid type str

class TestBilinearInitialization(unittest.TestCase):
    """Test that the Bilinear module correctly validates the inputs."""
    def test_invalid_in_features1_type(self):
        """Test that TypeError is raised for invalid in_features1 type."""
        with self.assertRaises(TypeError):
            Bilinear(in1_features=10.5, in2_features=5, out_features=3)  # Invalid type float
    def test_invalid_in_features2_type(self):
        """Test that TypeError is raised for invalid in_features2 type."""
        with self.assertRaises(TypeError):
            Bilinear(in1_features=10, in2_features="five", out_features=3)  # Invalid type str
    def test_invalid_out_features_type(self):
        """Test that TypeError is raised for invalid out_features type."""
        with self.assertRaises(TypeError):
            Bilinear(in1_features=10, in2_features=5, out_features="three")  # Invalid type str

if __name__ == "__main__":
    unittest.main()

