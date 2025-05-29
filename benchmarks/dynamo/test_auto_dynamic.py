#!/usr/bin/env python3

import unittest
import torch
from common import generate_varied_inputs


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)


class TestAutoDynamic(unittest.TestCase):
    def test_generate_varied_inputs(self):
        # Create a simple input tensor with batch size 8
        batch_size = 8
        x = torch.randn(batch_size, 10)
        example_inputs = (x,)
        
        # Generate 3 variations
        variations = generate_varied_inputs(example_inputs, batch_size, variation_factor=0.2, num_variations=3)
        
        # Should get 3 variations (including the original)
        self.assertEqual(len(variations), 3)
        
        # Each variation should be a tuple with one tensor
        for var in variations:
            self.assertEqual(len(var), 1)
            self.assertIsInstance(var[0], torch.Tensor)
            
        # All variations should have the second dimension unchanged (10)
        for var in variations:
            self.assertEqual(var[0].shape[1], 10)
            
        # The batch dimension should vary
        batch_sizes = [var[0].shape[0] for var in variations]
        self.assertGreater(len(set(batch_sizes)), 1, "Batch sizes should vary")
        
        # None should be exactly the same shape unless by chance
        # (though with random variation there is a small probability of this)
        shapes = [var[0].shape for var in variations]
        if len(shapes) == len(set(shapes)):
            # All shapes are different - this is expected
            pass
        else:
            # If we do have some identical shapes by chance, make sure it's not all of them
            self.assertLess(shapes.count(shapes[0]), len(shapes), 
                            "Not all variations should have the same shape")
    
    def test_complex_input_structure(self):
        """Test with a more complex input structure including dictionaries and lists."""
        batch_size = 8
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 5)
        z = torch.randint(0, 10, (batch_size,))
        
        # Create a complex nested structure
        example_inputs = {
            'input1': x, 
            'nested': [y, {'deep': z}]
        }
        
        # Generate variations
        variations = generate_varied_inputs(example_inputs, batch_size, variation_factor=0.2, num_variations=3)
        
        # Should get 3 variations
        self.assertEqual(len(variations), 3)
        
        # Check structure is preserved
        for var in variations:
            self.assertIn('input1', var)
            self.assertIsInstance(var['nested'], list)
            self.assertEqual(len(var['nested']), 2)
            self.assertIsInstance(var['nested'][1], dict)
            self.assertIn('deep', var['nested'][1])


if __name__ == "__main__":
    unittest.main()