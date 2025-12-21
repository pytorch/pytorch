import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx.traceback as fx_traceback

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
        with fx_traceback.annotate({"compile_with_inductor": 0}):
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
    torch.set_default_device(device)

    # Create model
    model = Transformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)
    model.eval()  # export-friendly; switch to train() if you want training graphs

    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs=True):
        # Inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Compiler toolkit imports
        from contextlib import ExitStack
        from torch._functorch.aot_autograd import (
            aot_export_joint_with_descriptors,
            aot_compile_joint_with_descriptors,
        )
        from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable

        # Export and compile the joint graph
        stack = ExitStack()
        with stack:
            jd = aot_export_joint_with_descriptors(
                stack,
                model,
                (input_ids,),
            )

            from torch.fx.passes.regional_inductor import regional_inductor

            def fwd_compile(gm: torch.fx.GraphModule, example_inputs):
                gm = regional_inductor(gm, example_inputs)
                from torch._inductor.output_code import MockFXGraphCacheOutput
                return MockFXGraphCacheOutput(gm)

            def bwd_compile(gm: torch.fx.GraphModule, example_inputs):
                gm = regional_inductor(gm, example_inputs)
                from torch._inductor.output_code import MockFXGraphCacheOutput
                return MockFXGraphCacheOutput(gm)

            compiled_wrapper = aot_compile_joint_with_descriptors(
                jd,
                fw_compiler=fwd_compile,
                bw_compiler=bwd_compile,
            )

        # Save compiled artifacts
        serialization_path = "/tmp/cross_precompile_1.pt"
        with open(serialization_path, "wb") as f:
            f.write(BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(compiled_wrapper))

    # Load compiled callable
    with open(serialization_path, "rb") as f:
        loaded_fn = BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(f.read())

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Run loaded compiled callable: it expects (params, buffers, inputs)
    (logits,) = loaded_fn(*model.parameters(), *model.buffers(), input_ids)

    # Compute loss and backward for demonstration (note: compiled callable returns forward; backward graph is bundled)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    print("Successfully cross compiled with compiler toolkit")

if __name__ == "__main__":
    main()
