"""The ``fsdp`` ops that the engine defines in Python; ``fsdp::chunk_cat_mixed_dtype``
and ``fsdp::_all_gather_copy_out_`` are C++ ops (torch/csrc/distributed/fsdp).
The modules that call them import this one, and Inductor lowers them
(torch/_inductor/lowering.py)."""

import torch


lib = torch.library.Library("fsdp", "FRAGMENT")

lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        Tensor all_gather_output,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt rank
    ) -> (Tensor, Tensor)
    """
)


# These ops only call other ATen ops, so their impls register once to a
# composite-explicit alias key instead of once per backend: that covers every
# backend (Meta included) and, unlike CompositeImplicitAutograd, keeps the ops
# from being decomposed when traced, so compile still sees them.
# all_gather_copy_in fills its output by mutating views of it, so it uses the
# NonFunctional variant, which leaves out functional backends (XLA, Lazy). Its
# Meta impl, which skips the copy, takes precedence on meta tensors.
@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "all_gather_copy_in", "CompositeExplicitAutogradNonFunctional")
def all_gather_copy_in_default(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


lib.define(
    "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
)


@torch.library.impl(lib, "split_with_sizes_copy", "CompositeExplicitAutograd")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: list[int],
    dim: int = 0,
    *,
    out: list[torch.Tensor],
) -> None:
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )


# Unused within FSDP since the reduce-scatter copy-in moved to
# fsdp::chunk_cat_mixed_dtype; kept to avoid breaking potential external callers.
lib.define(
    "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
)


@torch.library.impl(lib, "chunk_cat", "CompositeExplicitAutograd")
def chunk_cat(
    tensors: list[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


lib.define(
    "record_grad_output_stream(Tensor buffer, int stream_id, int device_index, int device_type) -> ()"
)


@torch.library.impl(lib, "record_grad_output_stream", "CompositeExplicitAutograd")
def _record_grad_output_stream_default(
    buffer: torch.Tensor,
    stream_id: int,
    device_index: int,
    device_type: int,
) -> None:
    # No-op default. Where free is host-side bookkeeping (e.g. CUDA) the block
    # stays mapped, so no read-before-free edge is needed. A backend whose free
    # releases device memory registers its own impl. See BucketRuntime.reduce_grads.
    return


lib.define("copy_(Tensor(a!) tensor, Tensor data) -> ()")


# Like the ops above, this registers once to CompositeExplicitAutograd so that it covers every backend without listing
# them. Not CompositeImplicitAutograd: that would decompose it into aten.copy_,
# skipping copy__functionalize below and Inductor's fsdp.copy_ lowering.
@torch.library.impl(lib, "copy_", "CompositeExplicitAutograd")
def copy_(tensor, data):
    tensor.copy_(data)


@torch.library.impl(lib, "copy_", "Functionalize")
def copy__functionalize(tensor, data):
    torch._sync(tensor)
    torch._sync(data)
    tensor_inner = torch._from_functional_tensor(tensor)
    data_inner = torch._from_functional_tensor(data)
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        torch.ops.fsdp.copy_.default(tensor_inner, data_inner)


torch.fx.node.has_side_effect(torch.ops.fsdp.copy_.default)
