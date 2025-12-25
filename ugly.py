import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import io
import tempfile
from typing import Any, Callable, Tuple, Dict

torch._dynamo.config.enable_aot_compile = True

# Simple Transformer components
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        # x: (batch, seq, embed)
        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, H, S, S)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, H, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)  # (B, S, E)
        return self.out_proj(attn_out)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        # pre-norm
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    def forward(self, input_ids):
        # input_ids: (batch, seq)
        B, S = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :S, :]
        for block in self.layers:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)
        return logits

def main():
    # Hyperparameters
    vocab_size = 50257
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    max_seq_len = 64
    batch_size = 2
    seq_len = 32
    device = "cuda"
    torch.manual_seed(0)
    # Create model
    model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)
    # Compile per transformer block using caching precompile
    compiled_model = torch.compile(
        model,
        fullgraph=True,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    serialization_path = "/tmp/cross_precompile.pt"
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    compiled_model.forward.aot_compile(((input_ids, ), {})).save_compiled_function(serialization_path)

    with open(serialization_path, "rb") as f:
        loaded_fn = torch.compiler.load_compiled_function(
            f,
        )
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = loaded_fn(model, input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        print("SUCCESS!")

if __name__ == "__main__":
    main()
