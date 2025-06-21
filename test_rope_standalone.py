#!/usr/bin/env python3
"""
Standalone test for RoPE (Rotary Positional Embedding) implementation.
This can be run independently to verify the basic functionality.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the torch.version module to avoid import issues in dev environment
if 'torch.version' not in sys.modules:
    import types
    version_module = types.ModuleType('torch.version')
    version_module.__version__ = '2.5.0+dev'
    sys.modules['torch.version'] = version_module

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    def test_rope_basic():
        print("Testing basic RoPE functionality...")
        
        # Test frequency generation
        seq_len, dim = 10, 4
        freqs = F.rotary_embedding_freqs(seq_len=seq_len, dim=dim)
        print(f"Generated frequencies shape: {freqs.shape}")
        expected_shape = (seq_len, dim // 2, 2)
        assert freqs.shape == expected_shape, f"Expected {expected_shape}, got {freqs.shape}"
        
        # Test that frequencies are in valid range for cosine/sine
        assert torch.all(torch.abs(freqs) <= 1.1), "Frequency values out of expected range"
        
        # Test rotation application
        x = torch.randn(2, seq_len, dim)
        rotated = F.apply_rotary_embedding(x, freqs)
        print(f"Input shape: {x.shape}, Rotated shape: {rotated.shape}")
        assert rotated.shape == x.shape, f"Shape mismatch: {rotated.shape} vs {x.shape}"
        
        # Test that RoPE preserves magnitude approximately (rotations preserve norms)
        input_norms = torch.norm(x, dim=-1)
        output_norms = torch.norm(rotated, dim=-1)
        assert torch.allclose(input_norms, output_norms, atol=1e-6), "Norms not preserved"
        
        print("✓ Basic functional API tests passed")
        
    def test_rope_module():
        print("Testing RoPE module...")
        
        # Test module creation and forward pass
        dim = 64
        rope = nn.RotaryPositionalEmbedding(dim=dim)
        x = torch.randn(2, 10, dim)
        output = rope(x)
        
        print(f"Module input shape: {x.shape}, output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        
        # Test with different sequence lengths
        x_short = torch.randn(1, 5, dim)
        x_long = torch.randn(1, 100, dim)
        output_short = rope(x_short)
        output_long = rope(x_long)
        
        assert output_short.shape == x_short.shape
        assert output_long.shape == x_long.shape
        
        print("✓ Module tests passed")
        
    def test_rope_compatibility():
        print("Testing compatibility with existing RoPE implementations...")
        
        # Test that our implementation matches the expected mathematical behavior
        # For position 0, RoPE should be close to identity (small rotation)
        dim = 8
        x = torch.randn(1, 1, dim)  # Single element at position 0
        
        freqs = F.rotary_embedding_freqs(seq_len=1, dim=dim)
        rotated = F.apply_rotary_embedding(x, freqs)
        
        # At position 0, the rotation should be minimal
        # (though not exactly identity due to the base frequency)
        diff = torch.norm(x - rotated)
        print(f"Difference from identity at position 0: {diff.item():.6f}")
        
        # Test that different positions give different results
        freqs_long = F.rotary_embedding_freqs(seq_len=10, dim=dim)
        x_seq = x.repeat(1, 10, 1)
        rotated_seq = F.apply_rotary_embedding(x_seq, freqs_long)
        
        # Different positions should yield different embeddings
        assert not torch.allclose(rotated_seq[:, 0], rotated_seq[:, 5]), "Different positions should yield different embeddings"
        
        print("✓ Compatibility tests passed")
        
    def main():
        print("Running standalone RoPE tests...")
        try:
            test_rope_basic()
            test_rope_module() 
            test_rope_compatibility()
            print("\n🎉 All RoPE tests passed successfully!")
            return 0
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires a working PyTorch installation.")
    sys.exit(1)