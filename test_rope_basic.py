#!/usr/bin/env python3

"""
Basic test script for RoPE functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_rope_functional():
    """Test RoPE functional API"""
    print("Testing RoPE functional API...")
    
    # Test basic functionality
    seq_len, dim = 4, 8
    batch_size = 2
    
    # Test frequency generation
    freqs = F.rotary_position_embedding_freqs(seq_len, dim)
    print(f"Frequency matrix shape: {freqs.shape}")
    assert freqs.shape == (seq_len, dim // 2, 2), f"Expected {(seq_len, dim // 2, 2)}, got {freqs.shape}"
    
    # Test rotary embedding
    x = torch.randn(batch_size, seq_len, dim)
    x_rot = F.rotary_position_embedding(x, freqs)
    print(f"Input shape: {x.shape}, Output shape: {x_rot.shape}")
    assert x_rot.shape == x.shape, f"Shape mismatch: {x_rot.shape} vs {x.shape}"
    
    # Test with different base
    freqs_alt = F.rotary_position_embedding_freqs(seq_len, dim, base=20000.0)
    x_rot_alt = F.rotary_position_embedding(x, freqs_alt)
    assert not torch.allclose(x_rot, x_rot_alt), "Different bases should produce different results"
    
    print("✓ Functional API tests passed!")

def test_rope_module():
    """Test RoPE module"""
    print("Testing RoPE module...")
    
    # Test basic module
    dim = 64
    max_seq_len = 128
    rope = nn.RotaryPositionalEmbedding(dim, max_seq_len)
    
    # Test forward pass
    x = torch.randn(2, 50, dim)
    x_rot = rope(x)
    assert x_rot.shape == x.shape, f"Shape mismatch: {x_rot.shape} vs {x.shape}"
    
    # Test dynamic computation (no max_seq_len)
    rope_dynamic = nn.RotaryPositionalEmbedding(dim)
    x_rot_dynamic = rope_dynamic(x)
    assert x_rot_dynamic.shape == x.shape
    
    # Test that pre-computed and dynamic give same results for same input
    assert torch.allclose(x_rot, x_rot_dynamic, atol=1e-6)
    
    print("✓ Module tests passed!")

def test_rope_gradients():
    """Test gradients work correctly"""
    print("Testing RoPE gradients...")
    
    seq_len, dim = 8, 16
    x = torch.randn(2, seq_len, dim, requires_grad=True)
    
    # Test functional gradients
    freqs = F.rotary_position_embedding_freqs(seq_len, dim)
    x_rot = F.rotary_position_embedding(x, freqs)
    loss = x_rot.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow through RoPE"
    assert x.grad.shape == x.shape, "Gradient shape should match input"
    
    # Test module gradients
    x.grad = None  # Reset gradients
    rope = nn.RotaryPositionalEmbedding(dim)
    x_rot = rope(x)
    loss = x_rot.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow through RoPE module"
    
    print("✓ Gradient tests passed!")

def test_rope_edge_cases():
    """Test edge cases and error conditions"""
    print("Testing RoPE edge cases...")
    
    # Test odd dimension error
    try:
        F.rotary_position_embedding_freqs(4, 7)  # Odd dimension
        assert False, "Should raise error for odd dimension"
    except ValueError:
        pass
    
    # Test dimension mismatch
    freqs = F.rotary_position_embedding_freqs(4, 8)
    x = torch.randn(2, 4, 6)  # Wrong dimension
    try:
        F.rotary_position_embedding(x, freqs)
        assert False, "Should raise error for dimension mismatch"
    except ValueError:
        pass
    
    # Test sequence length mismatch
    freqs = F.rotary_position_embedding_freqs(4, 8)
    x = torch.randn(2, 6, 8)  # Longer sequence
    try:
        F.rotary_position_embedding(x, freqs)
        assert False, "Should raise error for sequence length mismatch"
    except ValueError:
        pass
    
    print("✓ Edge case tests passed!")

if __name__ == "__main__":
    print("Running basic RoPE tests...")
    test_rope_functional()
    test_rope_module() 
    test_rope_gradients()
    test_rope_edge_cases()
    print("\n🎉 All tests passed!")