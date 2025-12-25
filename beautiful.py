import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple Transformer components (same as ugly.py)
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
    example_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Phase 1: Dynamo - trace the model and capture the FX graph
    print("Phase 1: Running Precompile.dynamo()...")
    dynamo_output = torch.Precompile.dynamo(model, example_input)
    print(f"  - Captured graph with {len(list(dynamo_output.graph_module.graph.nodes))} nodes")

    # Phase 2: AOT Autograd - trace forward+backward (inference mode for now)
    print("Phase 2: Running Precompile.aot_autograd()...")
    aot_output = torch.Precompile.aot_autograd(dynamo_output, trace_joint=False)
    print(f"  - Joint graph has {len(list(aot_output.joint_graph.graph.nodes))} nodes")

    # Phase 3: Inductor - compile with the Inductor backend
    print("Phase 3: Running Precompile.inductor()...")
    inductor_output = torch.Precompile.inductor(aot_output)
    print(f"  - Compiled module type: {type(inductor_output.compiled_module).__name__}")

    # Phase 4: Precompile - bundle all artifacts
    print("Phase 4: Running Precompile.precompile()...")
    precompiled_artifact = torch.Precompile.precompile(inductor_output)
    print(f"  - System info: {precompiled_artifact.system_info}")

    # Phase 5: Save - serialize to disk
    serialization_path = "/tmp/beautiful_precompile.pt"
    print(f"Phase 5: Saving to {serialization_path}...")
    torch.Precompile.save(
        serialization_path,
        precompiled_artifact,
        dynamo_output.bytecode,
        dynamo_output.guards,
    )
    print("  - Saved successfully")

    # Phase 6: Load - deserialize from disk
    print(f"Phase 6: Loading from {serialization_path}...")
    loaded_artifact = torch.Precompile.load(serialization_path)
    print(f"  - Loaded artifact type: {type(loaded_artifact).__name__}")

    # Run the loaded artifact
    # The compiled function expects all inputs (params + user inputs) that were
    # captured during dynamo tracing. Use dynamo_output.example_inputs for the
    # same inputs, or pass new user inputs with the existing params.
    print("Running loaded artifact...")

    # For a new input, we need to extract params from dynamo_output.example_inputs
    # and replace the user input portion. The dynamo_output.example_inputs contains
    # [param1, param2, ..., paramN, user_input] in traced order.
    # For now, we use the same inputs to verify correctness.
    with torch.no_grad():
        traced_inputs = [
            inp.detach() if inp.requires_grad else inp
            for inp in dynamo_output.example_inputs
        ]
        logits = loaded_artifact(*traced_inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
    print(f"  - Output shape: {logits.shape}")

    # Verify correctness against eager model with the same input
    # The last element of traced_inputs is the user input
    with torch.no_grad():
        eager_logits = model(example_input)
    if torch.allclose(logits, eager_logits, atol=1e-4):
        print("SUCCESS! Loaded artifact matches eager model output.")
    else:
        max_diff = (logits - eager_logits).abs().max().item()
        print(f"WARNING: Max difference from eager: {max_diff}")


if __name__ == "__main__":
    main()
