import torch

import contextlib
from torch._guards import tracing, TracingContext

import torch
import torch.nn as nn

from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)

from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
import torch.distributed as dist


def graph_capture_and_aot_export_joint_with_descriptors(model, inputs):
    assert isinstance(inputs, tuple)
    with torch._dynamo.config.patch(install_free_tensors=True), torch.fx.traceback.preserve_node_meta():
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
        fake_mode = gm.meta.get("fake_mode", None)

    with tracing(TracingContext(fake_mode)):
        return aot_export_joint_with_descriptors_alone(gm, inputs), fake_mode

def aot_export_joint_with_descriptors_alone(model, inputs):
    assert isinstance(inputs, tuple)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            inputs,
        )
        return joint_with_descriptors

def get_custom_metadata(gm):
    def helper(gm):
        custom_metadata = []
        for node in gm.graph.nodes:
            if hasattr(node, "meta") and node.meta.get("custom", None):
                custom_metadata.append((node.op, node.name, node.meta["custom"]))
            if node.op == "get_attr" and isinstance(
                getattr(gm, node.target), torch.fx.GraphModule
            ):
                custom_metadata.append(helper(getattr(gm, node.target)))
        return custom_metadata

    return "\n".join(str(x) for x in helper(gm))

def run():
    def _squared(score, b, h, m, n):
        return score

    def _get_block_causal_mask_mod(seq_idx):
        """
        Returns a document-aware causal mask function.
        This is similar to _get_block_causal_mask_mod in torchtitan.

        Args:
            seq_idx: Tensor of shape [batch_size, seqlen] containing sequence/document IDs

        Returns:
            A mask_mod function that captures seq_idx in its closure
        """
        def block_causal_mask(b, h, q_idx, kv_idx):
            """
            Document-aware causal mask (only 4 parameters as required)
            Args:
                b: batch index
                h: head index
                q_idx: query position index
                kv_idx: key/value position index
            Returns:
                Boolean mask: tokens can only attend within same document and to previous positions
            """
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    flex_attn_compiled = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    a = 12
    b = 64
    batch_size = 2
    seqlen = a * b

    # Create seq_idx tensor - maps each position to a document/sequence ID
    # Example: Split sequence into 2 documents for each batch
    # First half (0:384) belongs to document 0, second half (384:768) to document 1
    seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device="cuda")
    seq_idx[:, seqlen//2:] = 1  # Second half belongs to document 1

    # Get the mask_mod function with seq_idx captured in closure
    mask_mod = _get_block_causal_mask_mod(seq_idx)

    # Create block_mask with the mask_mod function (which only takes 4 args)
    # Note: We don't compile create_block_mask itself, just flex_attention
    block_mask = create_block_mask(mask_mod, None, None, seqlen, seqlen)

    class FlexAttentionModule(nn.Module):
        """Flex attention submodule similar to the sdpa in Llama3 Attention"""
        def forward(self, xq, xk, xv):
            """
            Args:
                xq: Query tensor (bs, n_heads, seqlen, head_dim)
                xk: Key tensor (bs, n_heads, seqlen, head_dim)
                xv: Value tensor (bs, n_heads, seqlen, head_dim)
            Returns:
                Output tensor (bs, n_heads, seqlen, head_dim)
            """
            with torch.fx.traceback.annotate({"compile_with_inductor": "flex_attention"}):
                output = flex_attn_compiled(
                    xq, xk, xv, block_mask=block_mask, score_mod=_squared
                )
            return output

    world_size = 8
    device_type = "cuda"
    store = FakeStore()
    dist.init_process_group(
        backend="fake", rank=0, world_size=world_size, store=store
    )

    # Set up 2-D mesh [dp, tp] similar to the example
    dp_degree = 2
    tp_degree = max(1, world_size // dp_degree)

    # Create 2-D mesh with [dp, tp] dimensions
    mesh_2d = init_device_mesh(
        device_type,
        mesh_shape=(dp_degree, tp_degree),
        mesh_dim_names=["dp", "tp"],
    )

    # Model configuration
    n_heads = 4
    head_dim = 64

    # Create input tensors in the shape expected by FlexAttentionModule
    # Shape: (bs, n_heads, seqlen, head_dim)
    xq = torch.randn(batch_size, n_heads, seqlen, head_dim, requires_grad=True, device="cuda")
    xk = torch.randn(batch_size, n_heads, seqlen, head_dim, requires_grad=True, device="cuda")
    xv = torch.randn(batch_size, n_heads, seqlen, head_dim, requires_grad=True, device="cuda")

    inputs = (xq, xk, xv)

    # Distribute the tensors across the tp dimension with Replicate placement
    dtensor_xq = distribute_tensor(xq, mesh_2d["tp"], placements=[Replicate()])
    dtensor_xk = distribute_tensor(xk, mesh_2d["tp"], placements=[Replicate()])
    dtensor_xv = distribute_tensor(xv, mesh_2d["tp"], placements=[Replicate()])

    print(f"Input xq tensor type: {type(dtensor_xq)}")
    print(f"Input xq tensor shape: {dtensor_xq.shape}")
    print(f"Device mesh: {mesh_2d}")

    # Create the FlexAttentionModule directly (no Attention wrapper)
    model = FlexAttentionModule().to("cuda")

    # Use regular input for now (can switch to dtensor inputs later)
    # inputs = (dtensor_xq, dtensor_xk, dtensor_xv)

    graph, mode = graph_capture_and_aot_export_joint_with_descriptors(model, inputs)

    print(graph.graph_module.print_readable(print_output=True))

    print("\nCustom Metadata:")
    print(get_custom_metadata(graph.graph_module))

    # Print module structure
    print("\nModule structure:")
    for name, module in model.named_modules():
        print(f"  {name}: {type(module).__name__}")

run()
