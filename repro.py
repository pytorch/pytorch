import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore


torch._dynamo.config.enable_aot_compile = True


# Simple Transformer components
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
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
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)  # (B, H, S, S)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # (B, H, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)  # (B, S, E)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
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
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        device_mesh=None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.device_mesh = device_mesh

    def forward(self, input_ids):
        # input_ids: (batch, seq)
        input_ids = input_ids.redistribute(self.device_mesh, [Replicate()])
        input_ids = input_ids.to_local()
        B, S = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :S, :]
        for block in self.layers:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab)
        return logits


def main():
    # Hyperparameters
    vocab_size = 32
    embed_dim = 4
    num_heads = 2
    num_layers = 1
    max_seq_len = 8
    batch_size = 1
    seq_len = 8
    device = "cuda"
    torch.manual_seed(0)
    # Create model

    from torch._subclasses.fake_tensor import FakeTensorMode

    if not dist.is_initialized():
        fake_store = FakeStore()
        dist.init_process_group(backend="fake", store=fake_store, rank=0, world_size=1)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Device mesh for DTensor
    device_mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp",),
    )
    model = Transformer(
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        device_mesh=device_mesh,
    ).to(device)
    # Compile per transformer block using caching precompile
    compiled_model = torch.compile(
        model,
        fullgraph=True,
    )

    with FakeTensorMode(allow_non_fake_inputs=True):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        input_ids_dt = DTensor.from_local(input_ids, device_mesh, [Shard(0)])

        compiled_model(input_ids_dt)

    print("BOOP")


if __name__ == "__main__":
    main()
