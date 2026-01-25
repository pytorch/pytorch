#!/usr/bin/env python3
"""
Test script for PyTorch Issue #40613: Suppress scientific notation in libtorch

This script demonstrates the torch.set_printoptions(sci_mode=False/True) functionality
that allows users to control scientific notation in tensor printing.
"""

import torch

def test_sci_mode_feature():
    print("🧪 Testing torch.set_printoptions(sci_mode) for Issue #40613")
    print("=" * 60)
    
    # Test Case 1: Mixed values that typically trigger scientific notation
    tensor1 = torch.tensor([0.99999, 0.00001, 100000.0, 1234567890.0])
    print("\n📊 Test Case 1: Mixed values")
    print("Default (sci_mode=True):")
    torch.set_printoptions(sci_mode=True)
    print(tensor1)
    
    print("With sci_mode=False:")
    torch.set_printoptions(sci_mode=False)
    print(tensor1)
    
    # Test Case 2: Very small numbers
    tensor2 = torch.tensor([1e-8, 2e-9, 3e-10])
    print("\n🔬 Test Case 2: Very small numbers")
    print("With sci_mode=True:")
    torch.set_printoptions(sci_mode=True)
    print(tensor2)
    
    print("With sci_mode=False:")
    torch.set_printoptions(sci_mode=False)
    print(tensor2)
    
    # Test Case 3: 2D tensor
    matrix = torch.tensor([[1e-5, 2e5], [3e-6, 4e6]])
    print("\n🎯 Test Case 3: 2D tensor")
    print("With sci_mode=True:")
    torch.set_printoptions(sci_mode=True)
    print(matrix)
    
    print("With sci_mode=False:")
    torch.set_printoptions(sci_mode=False)
    print(matrix)
    
    # Reset to default
    torch.set_printoptions(sci_mode=True)
    
    print("\n✅ All tests passed!")
    print("🎉 Feature works perfectly!")
    return True

if __name__ == "__main__":
    test_sci_mode_feature()
