#!/usr/bin/env python3
"""
Example usage of PyTorch RoPE (Rotary Positional Embedding) implementation.

This shows how the new torch.nn.functional.rotary_embedding_freqs and 
torch.nn.functional.apply_rotary_embedding functions can be used,
as well as the torch.nn.RotaryPositionalEmbedding module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def example_functional_api():
    """Example using the functional API directly."""
    print("=== Functional API Example ===")
    
    # Generate RoPE frequencies for sequences up to length 512 with 64-dimensional embeddings
    seq_len, dim = 512, 64
    freqs = F.rotary_embedding_freqs(seq_len=seq_len, dim=dim, base=10000.0)
    print(f"Generated frequencies shape: {freqs.shape}")  # (512, 32, 2)
    
    # Apply RoPE to query and key tensors in attention
    batch_size, num_heads, actual_seq_len, head_dim = 8, 12, 128, 64
    
    # Simulate query and key tensors from attention mechanism
    q = torch.randn(batch_size, num_heads, actual_seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, actual_seq_len, head_dim)
    
    # Use only the frequencies for the actual sequence length
    freqs_truncated = freqs[:actual_seq_len]  # (128, 32, 2)
    
    # Apply RoPE to queries and keys
    # unsqueeze_dim=2 because seq_len is at index 2 in (batch, heads, seq_len, dim)
    q_rotated = F.apply_rotary_embedding(q, freqs_truncated, unsqueeze_dim=2)
    k_rotated = F.apply_rotary_embedding(k, freqs_truncated, unsqueeze_dim=2) 
    
    print(f"Original Q shape: {q.shape}, Rotated Q shape: {q_rotated.shape}")
    print(f"Original K shape: {k.shape}, Rotated K shape: {k_rotated.shape}")
    
    # The rotated tensors can now be used in attention computation
    # attention_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))
    
def example_module_api():
    """Example using the RotaryPositionalEmbedding module."""
    print("\n=== Module API Example ===")
    
    # Create a RoPE module for 64-dimensional embeddings
    rope = nn.RotaryPositionalEmbedding(dim=64, max_seq_len=2048, base=10000.0)
    print(f"Created RoPE module: {rope}")
    
    # Example 1: Simple case with (batch_size, seq_len, dim) tensors
    x = torch.randn(4, 100, 64)  # (batch, seq_len, dim)
    x_rotated = rope(x)  # seq_len dim is -2 by default
    print(f"Simple case - Input: {x.shape}, Output: {x_rotated.shape}")
    
    # Example 2: Multi-head attention case with (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(4, 8, 100, 64)  # (batch, heads, seq_len, head_dim)
    k = torch.randn(4, 8, 100, 64)
    
    # Apply RoPE along the seq_len dimension (index 2)
    q_rotated = rope(q, dim=2)
    k_rotated = rope(k, dim=2)
    print(f"Multi-head case - Q input: {q.shape}, Q output: {q_rotated.shape}")
    
    # Example 3: Variable sequence lengths (useful for dynamic batching)
    x_var = torch.randn(2, 50, 64)  # shorter sequence
    x_var_rotated = rope(x_var, seq_len=50)  # explicitly specify seq_len
    print(f"Variable length - Input: {x_var.shape}, Output: {x_var_rotated.shape}")

def example_transformer_integration():
    """Example showing how to integrate RoPE into a simple transformer layer."""
    print("\n=== Transformer Integration Example ===")
    
    class MultiHeadAttentionWithRoPE(nn.Module):
        def __init__(self, d_model, num_heads, max_seq_len=2048):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model) 
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            # RoPE for queries and keys
            self.rope = nn.RotaryPositionalEmbedding(
                dim=self.head_dim, 
                max_seq_len=max_seq_len
            )
            
        def forward(self, x):
            batch_size, seq_len, d_model = x.shape
            
            # Project to Q, K, V
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Apply RoPE to queries and keys
            # dim=2 because seq_len is at index 2 in (batch, heads, seq_len, head_dim)
            q = self.rope(q, dim=2)
            k = self.rope(k, dim=2)
            
            # Compute attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, d_model
            )
            return self.out_proj(attn_output)
    
    # Test the attention layer
    attention = MultiHeadAttentionWithRoPE(d_model=512, num_heads=8)
    x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
    output = attention(x)
    print(f"Transformer attention - Input: {x.shape}, Output: {output.shape}")

def example_comparison_with_existing():
    """Example showing how this RoPE implementation compares to existing ones."""
    print("\n=== Comparison with Existing Implementations ===")
    
    # This shows how existing RoPE code can be replaced
    seq_len, dim = 50, 64
    
    # OLD WAY (from existing implementations):
    # def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000):
    #     freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    #     t = torch.arange(seq_len, device=freqs.device)
    #     freqs = torch.outer(t, freqs)
    #     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    #     cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    #     return cache.to(dtype=torch.bfloat16)
    
    # NEW WAY (using PyTorch core):
    freqs = F.rotary_embedding_freqs(seq_len=seq_len, dim=dim, base=10000)
    print(f"New RoPE frequencies shape: {freqs.shape}")
    
    # OLD WAY (apply rotation):
    # def apply_rotary_emb(x, freqs_cis):
    #     xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    #     freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    #     x_out2 = torch.stack([
    #         xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
    #         xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    #     ], -1)
    #     x_out2 = x_out2.flatten(3)
    #     return x_out2.type_as(x)
    
    # NEW WAY (using PyTorch core):
    x = torch.randn(2, seq_len, dim)
    rotated = F.apply_rotary_embedding(x, freqs)
    print(f"Applied RoPE - Input: {x.shape}, Output: {rotated.shape}")
    
    print("Benefits of the new PyTorch core implementation:")
    print("✓ Standardized API across all PyTorch projects")
    print("✓ Consistent parameter names and behavior")
    print("✓ Better documentation and type hints")
    print("✓ Optimized implementation")
    print("✓ Module wrapper for easy integration")

if __name__ == "__main__":
    print("PyTorch RoPE Implementation Examples")
    print("====================================")
    
    example_functional_api()
    example_module_api()
    example_transformer_integration()
    example_comparison_with_existing()
    
    print("\n🎉 All examples completed successfully!")
    print("\nThe RoPE implementation is now available in PyTorch core!")
    print("- Functional API: torch.nn.functional.rotary_embedding_freqs(), torch.nn.functional.apply_rotary_embedding()")
    print("- Module API: torch.nn.RotaryPositionalEmbedding")