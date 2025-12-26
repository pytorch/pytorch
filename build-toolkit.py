import torch
import torch.distributed as dist
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore


torch._dynamo.config.enable_aot_compile = True


def dtensorify_module(
    module: nn.Module,
    device_mesh,
    *,
    param_placements=None,
    buffer_placements=None,
) -> None:
    """
    Replace every Parameter/Buffer on `module` (recursively) with a DTensor version.
    Default is Replicate, which matches the common torchtitan-style baseline for DP.
    """
    if param_placements is None:
        param_placements = [Replicate()]
    if buffer_placements is None:
        buffer_placements = [Replicate()]

    # Parameters on this module only
    for name, p in list(module.named_parameters(recurse=False)):
        if p is None or isinstance(p, DTensor):
            continue
        dt = DTensor.from_local(p.data, device_mesh, param_placements)
        new_p = nn.Parameter(dt, requires_grad=p.requires_grad)
        setattr(module, name, new_p)

    # Buffers on this module only
    for name, b in list(module.named_buffers(recurse=False)):
        if b is None or isinstance(b, DTensor):
            continue
        dt = DTensor.from_local(b, device_mesh, buffer_placements)
        module._buffers[name] = dt

    for child in module.children():
        dtensorify_module(
            child,
            device_mesh,
            param_placements=param_placements,
            buffer_placements=buffer_placements,
        )


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
        with fx_traceback.annotate({"compile_with_inductor": 0}):
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

    def forward(self, input_ids: DTensor):
        # Keep everything in DTensor land
        input_ids = input_ids.redistribute(self.device_mesh, [Replicate()])

        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.shape[1], :]

        def _run_block(block, hidden):
            return block(hidden)

        for block in self.layers:
            x = torch.utils.checkpoint.checkpoint(
                _run_block, block, x, use_reentrant=False
            )

        x = self.ln_f(x)
        logits = self.head(x)  # DTensor
        return logits


def main():
    # Distributed setup using a fake process group to run single-rank
    if not dist.is_initialized():
        fake_store = FakeStore()
        dist.init_process_group(backend="fake", store=fake_store, rank=0, world_size=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Hyperparameters
    vocab_size = 50257
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    max_seq_len = 64
    batch_size = 2
    seq_len = 32
    torch.manual_seed(0)
    torch.set_default_device(device)

    # Device mesh for DTensor
    device_mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp",),
    )

    # Create model
    model = Transformer(
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        device_mesh=device_mesh,
    ).to(device)

    # Torchtitan-style: dtensorify params and buffers before capture/export
    dtensorify_module(
        model,
        device_mesh,
        param_placements=[Replicate()],
        buffer_placements=[Replicate()],
    )

    model.eval()

    # Capture with Dynamo first, then export/compile with AOTAutograd using the captured GraphModule
    from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
    from torch._functorch.aot_autograd import (
        aot_compile_joint_with_descriptors,
        aot_export_joint_with_descriptors,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode, torch._dynamo.config.patch(install_free_tensors=True):
        local_input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        input_ids_dt = DTensor.from_local(local_input_ids, device_mesh, [Shard(0)])

        from torch._dynamo.functional_export import _dynamo_graph_capture_for_export

        gm = _dynamo_graph_capture_for_export(model)((input_ids_dt,))
        fake_mode = gm.meta["fake_mode"]

    import contextlib

    with contextlib.ExitStack() as stack:
        if fake_mode is not None:
            stack.enter_context(fake_mode)

        from contextlib import ExitStack

        with ExitStack() as export_stack:
            jd = aot_export_joint_with_descriptors(
                export_stack,
                gm,
                (input_ids_dt,),
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

        serialization_path = "/tmp/cross_precompile_1.pt"
        with open(serialization_path, "wb") as f:
            f.write(
                BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
                    compiled_wrapper
                )
            )

    # Load compiled callable
    with open(serialization_path, "rb") as f:
        loaded_fn = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                f.read()
            )
        )

    # Create DTensor inputs for run
    local_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    input_ids_dt = DTensor.from_local(local_input_ids, device_mesh, [Shard(0)])

    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Run loaded compiled callable: it expects (params, buffers, inputs)
    (logits_dt,) = loaded_fn(*model.parameters(), *model.buffers(), input_ids_dt)

    # Loss on local tensors for convenience
    logits = logits_dt.to_local() if isinstance(logits_dt, DTensor) else logits_dt
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    print(
        "Successfully cross compiled with compiler toolkit using DTensor params and DTensor input_ids"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
