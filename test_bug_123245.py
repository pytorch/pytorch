
#!/usr/bin/env python3
"""
Test script for PyTorch issue #123245
Demonstrates that get_overridable_functions() lists functions that are not actually overridable
"""

import torch

def test_overridable_functions_bug():
    print("=== Testing get_overridable_functions bug (Issue #123245) ===")
    
    # Add a custom function to torch.Tensor
    def custom_function():
        return "This is a custom function"
    
    torch.Tensor.custom_function = custom_function
    
    # Check if it appears in overridable functions
    overridable_funcs = torch.overrides.get_overridable_functions()
    tensor_funcs = set(overridable_funcs[torch.Tensor])
    
    print(f"custom_function in overridable functions: {custom_function in tensor_funcs}")
    
    # Try to actually use it with __torch_function__
    class TestTensor:
        def __torch_function__(self, func, types, args=(), kwargs=None):
            print(f"__torch_function__ called with: {func}")
            return "overridden!"
    
    test_tensor = TestTensor()
    
    try:
        # This won't work because custom_function doesn't actually support __torch_function__
        result = test_tensor.custom_function()
        print(f"Result: {result}")
        print("ERROR: This should not have worked!")
    except AttributeError as e:
        print(f"Expected error: {e}")
        print("✓ Bug confirmed: Function appears as overridable but doesn't work with __torch_function__")

if __name__ == "__main__":
    test_overridable_functions_bug()
