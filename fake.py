"""
fake.py - Example of FakeTensorMode + FakeProcessGroup + DTensor + Transformer
with aot_export_joint_with_descriptors for cross-compilation.

Run with: python fake.py
No torchrun needed - everything is faked!

Pattern:
1. Create model on meta device (no storage)
2. DTensorify with meta tensors (OUTSIDE FakeTensorMode)
3. Enter FakeTensorMode and call to_empty() to materialize as fake tensors
4. Capture with dynamo_graph_capture_for_export
5. Export with aot_export_joint_with_descriptors
6. Compile with aot_compile_joint_with_descriptors
7. Serialize and deserialize
8. Run inference with real tensors
"""

import contextlib
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as c10d
import torch.nn as nn
import torch.nn.functional as F

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore


# =============================================================================
# Transformer Model
# =============================================================================


@dataclass
class ModelArgs:
    vocab_size: int = 1000
    max_seq_len: int = 128
    n_layers: int = 2
    dim: int = 64
    n_heads: int = 4


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = nn.Sequential(
            nn.Linear(args.dim, 4 * args.dim),
            nn.GELU(),
            nn.Linear(4 * args.dim, args.dim),
        )
        self.attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.pos_emb = nn.Embedding(args.max_seq_len, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # Register position indices as buffer so it gets DTensorified
        self.register_buffer(
            "positions", torch.arange(args.max_seq_len), persistent=False
        )

    def forward(self, tokens):
        bsz, seq_len = tokens.shape
        # Use registered buffer (DTensorified) instead of torch.arange
        pos_indices = self.positions[:seq_len]
        h = self.tok_emb(tokens) + self.pos_emb(pos_indices)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output(h)


# =============================================================================
# Helpers
# =============================================================================


def dtensorify_module(
    module: nn.Module,
    device_mesh,
    *,
    param_placements=None,
    buffer_placements=None,
) -> None:
    """Convert module parameters and buffers to DTensors."""
    if param_placements is None:
        param_placements = [Replicate()]
    if buffer_placements is None:
        buffer_placements = [Replicate()]

    for name, p in list(module.named_parameters(recurse=False)):
        if p is None or isinstance(p, DTensor):
            continue
        dt = DTensor.from_local(p.data, device_mesh, param_placements)
        new_p = nn.Parameter(dt, requires_grad=p.requires_grad)
        setattr(module, name, new_p)

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


def _restore_state_dict(model, gm):
    """Restore state dict from model to graph module."""
    # Copy over parameter/buffer attributes from original model to gm
    params_and_buffers = dict(model.named_parameters())
    params_and_buffers.update(dict(model.named_buffers()))
    for name, val in params_and_buffers.items():
        # Replace dots with underscores for attribute names
        attr_name = name.replace(".", "_")
        if hasattr(gm, attr_name):
            setattr(gm, attr_name, val)


# =============================================================================
# Main
# =============================================================================
def main():
    # Cleanup any previous process group
    if c10d.is_initialized():
        c10d.destroy_process_group()

    # Config
    world_size = 1
    rank = 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize fake process group
    fake_store = FakeStore()
    c10d.init_process_group(backend="fake", store=fake_store, rank=rank, world_size=world_size)

    try:
        # Model config
        args = ModelArgs(vocab_size=1000, dim=256, n_heads=8, n_layers=2, max_seq_len=32)
        batch_size = 2
        seq_len = 16
        torch.manual_seed(0)
        torch.set_default_device(device)

        # Create device mesh
        device_mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp",),
        )

        # 1. Create model on META device (no storage allocated)
        print("Creating model on meta device...")
        with torch.device("meta"):
            model = Transformer(args)

        # 2. DTensorify with META tensors (OUTSIDE FakeTensorMode!)
        print("DTensorifying model...")
        dtensorify_module(
            model,
            device_mesh,
            param_placements=[Replicate()],
            buffer_placements=[Replicate()],
        )

        # 3. Enter FakeTensorMode and materialize with to_empty()
        print("Entering FakeTensorMode and materializing model...")
        outer_fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with outer_fake_mode:
            # Convert meta tensors -> fake tensors on target device
            model.to_empty(device=device)

            # Create fake input
            local_input_ids = torch.randint(
                0, args.vocab_size, (batch_size, seq_len), device=device
            )
            input_ids_dt = DTensor.from_local(local_input_ids, device_mesh, [Shard(0)])

        # 4. Capture with dynamo_graph_capture_for_export
        print("Capturing with dynamo_graph_capture_for_export...")
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        gm = dynamo_graph_capture_for_export(model)(input_ids_dt)

        # Restore state dict
        _restore_state_dict(model, gm)

        fake_mode = gm.meta["fake_mode"]

        # 5. Export and compile with aot_export_joint_with_descriptors
        print("Exporting with aot_export_joint_with_descriptors...")
        from torch._functorch.aot_autograd import (
            aot_compile_joint_with_descriptors,
            aot_export_joint_with_descriptors,
        )
        from torch.fx.passes.regional_inductor import regional_inductor
        from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
        from torch._inductor.output_code import RegionalOutputCode

        with contextlib.ExitStack() as stack:
            if fake_mode is not None:
                stack.enter_context(fake_mode)

            with contextlib.ExitStack() as export_stack:
                jd = aot_export_joint_with_descriptors(
                    export_stack,
                    gm,
                    (input_ids_dt,),
                )

                def fwd_compile(gm: torch.fx.GraphModule, example_inputs):
                    print("  Compiling forward graph...")
                    gm = regional_inductor(gm, example_inputs)
                    return RegionalOutputCode(gm)

                def bwd_compile(gm: torch.fx.GraphModule, example_inputs):
                    print("  Compiling backward graph...")
                    gm = regional_inductor(gm, example_inputs)
                    return RegionalOutputCode(gm)

                print("Compiling with aot_compile_joint_with_descriptors...")
                compiled_wrapper = aot_compile_joint_with_descriptors(
                    jd,
                    fw_compiler=fwd_compile,
                    bw_compiler=bwd_compile,
                    serializable=True,
                )

            # 6. Serialize
            print("Serializing compiled artifacts...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                serialization_path = f.name
                f.write(
                    BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
                        compiled_wrapper
                    )
                )

        # 7. Deserialize
        print(f"Deserializing from {serialization_path}...")
        with open(serialization_path, "rb") as f:
            loaded_fn = BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                f.read()
            )

        # 8. Run inference with real tensors
        print("Running inference with real tensors...")
        local_input_ids = torch.randint(
            0, args.vocab_size, (batch_size, seq_len), device=device
        )
        input_ids_dt = DTensor.from_local(local_input_ids, device_mesh, [Shard(0)])
        targets = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)

        # Create fresh model with real weights
        model = Transformer(args)
        model.to_empty(device=device)
        # Fix: reinitialize the positions buffer with actual indices
        model.positions = torch.arange(args.max_seq_len, device=device)

        # Run compiled function
        (logits_dt,) = loaded_fn(
            *model.parameters(), *model.buffers(), input_ids_dt
        )

        logits = logits_dt.to_local() if isinstance(logits_dt, DTensor) else logits_dt
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        print(f"Loss: {loss.item()}")

        # Backward pass
        loss.backward()
        print("Backward pass completed!")

        # Cleanup
        os.unlink(serialization_path)
        print(f"Cleaned up {serialization_path}")

        print("\nSuccess! Cross-compilation with aot_export_joint_with_descriptors works!")

    finally:
        c10d.destroy_process_group()


if __name__ == "__main__":
    main()
