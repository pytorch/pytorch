import unittest
import torch
from torch.overrides import get_overridable_functions


class TestOverridablesFunctionsFix(unittest.TestCase):
    """Test fix for issue #123245: dynamically added functions should not be overridable"""
    
    def test_dynamic_functions_not_overridable(self):
        """Test that dynamically added functions are not listed as overridable"""
        # Store original overridable functions
        original_funcs = set(get_overridable_functions()[torch.Tensor])
        
        # Add a dynamic function to torch.Tensor
        def custom_func(self):
            return "custom"
        
        torch.Tensor.custom_func = custom_func
        
        try:
            # Get overridable functions after adding dynamic function
            new_funcs = set(get_overridable_functions()[torch.Tensor])
            
            # The dynamic function should NOT be in the overridable list
            self.assertEqual(original_funcs, new_funcs, 
                           "Dynamic function should not appear in overridable functions")
            
            # Verify the function exists but is not overridable
            self.assertTrue(hasattr(torch.Tensor, 'custom_func'))
            self.assertNotIn(custom_func, new_funcs)
            
        finally:
            # Cleanup
            if hasattr(torch.Tensor, 'custom_func'):
                delattr(torch.Tensor, 'custom_func')


if __name__ == '__main__':
    unittest.main()
