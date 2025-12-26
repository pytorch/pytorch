import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import traceback as fx_traceback
from torch.fx.passes.regional_inductor import regional_inductor


# Simple Transformer components with regional compilation annotations
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

        # Mark the attention computation for regional inductor compilation
        with fx_traceback.annotate({"compile_with_inductor": "attention_region"}):
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

        # Mark the MLP computation for regional inductor compilation
        with fx_traceback.annotate({"compile_with_inductor": "mlp_region"}):
            mlp_out = self.mlp(self.ln2(x))

        x = x + mlp_out
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

    # =========================================================================
    # Composable Precompilation with Regional Inductor
    # =========================================================================
    # Using torch.Precompile API with regional_inductor as the compiler.
    # This compiles ONLY marked regions (attention, MLP) with Inductor while
    # leaving unmarked ops to run via eager/interpreter.
    # =========================================================================

    # Phase 1: Dynamo - trace the model and capture the FX graph
    print("Phase 1: Running Precompile.dynamo()...")
    gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
    print(f"  - Captured graph with {len(list(gm.graph.nodes))} nodes")

    # Count marked nodes
    marked_nodes = []
    for node in gm.graph.nodes:
        if node.op not in ('placeholder', 'output'):
            custom = node.meta.get('custom', None)
            if custom and 'compile_with_inductor' in custom:
                marked_nodes.append((node.name, custom['compile_with_inductor']))
    print(f"  - Found {len(marked_nodes)} nodes marked for regional compilation")

    # Phase 2: AOT Autograd with regional_inductor as the compiler
    # This creates forward/backward graphs and compiles marked regions
    print("Phase 2: Running Precompile.aot_autograd(compiler=regional_inductor)...")
    compiled_fn, guards = torch.Precompile.aot_autograd(
        gm, guards, compiler=regional_inductor
    )
    print("  - Compiled with regional inductor")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\nTraining with regional compilation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(3):
        # Forward pass - uses the compiled function
        logits = compiled_fn(example_inputs)

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = example_input[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print("\nTraining complete!")

    # =========================================================================
    # Verify correctness
    # =========================================================================
    print("\nVerifying correctness (fresh model)...")
    torch.manual_seed(42)
    fresh_model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)
    fresh_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Compile with regional inductor using Precompile API
    fresh_gm, _, fresh_guards, fresh_example_inputs = torch.Precompile.dynamo(fresh_model, fresh_input)
    fresh_compiled, _ = torch.Precompile.aot_autograd(fresh_gm, fresh_guards, compiler=regional_inductor)

    # Compare outputs
    fresh_model.eval()
    with torch.no_grad():
        compiled_out = fresh_compiled(fresh_example_inputs)
        eager_out = fresh_model(fresh_input)

    if torch.allclose(compiled_out, eager_out, atol=1e-4):
        print("SUCCESS! Regional compiled model matches eager model output.")
    else:
        max_diff = (compiled_out - eager_out).abs().max().item()
        print(f"Max difference from eager: {max_diff}")


if __name__ == "__main__":
    main()
