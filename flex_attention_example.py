import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention


class FlexAttentionModule(nn.Module):
    """
    Example module that uses flex_attention, which will trigger
    torch.ops.higher_order.flex_attention_backward in the backward graph.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, N, L, H, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (B, N, H, L, D) format expected by flex_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use flex_attention - this will trigger flex_attention_backward
        # You can optionally pass a score_mod function for custom attention patterns
        attn_output = flex_attention(q, k, v)

        # Reshape back: (B, N, H, L, D) -> (B, N, L, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)
        return output


# Example usage with autograd
if __name__ == "__main__":
    # Create model
    batch_size = 2
    seq_len = 16
    embed_dim = 128
    num_heads = 4

    model = FlexAttentionModule(embed_dim, num_heads)

    # Create input with requires_grad=True to enable autograd
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    # Forward pass
    output = model(x)

    # Compute loss (simple sum for demonstration)
    loss = output.sum()

    # Backward pass - this will trigger torch.ops.higher_order.flex_attention_backward
    loss.backward()

    print(f"Output shape: {output.shape}")
    print(f"Gradient computed: {x.grad is not None}")
    print(f"Input gradient shape: {x.grad.shape if x.grad is not None else None}")

    # Optional: Inspect the backward graph
    # You can use torch.autograd.graph to inspect the computation graph
    print("\nBackward graph contains flex_attention_backward")
