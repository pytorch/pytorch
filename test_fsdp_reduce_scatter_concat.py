# python -m torch.distributed.run --standalone --nproc_per_node=4 test_fsdp_reduce_scatter_concat.py

import torch
import os
import logging
torch_log = logging.getLogger("torch")
import logging
log = logging.getLogger(__name__)
from torch.distributed._composable.fsdp import _fsdp_collectives
from typing import Optional, List, cast
import math


if __name__ == "__main__":
    import torch.distributed as dist
    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    # world_size = int(os.getenv("WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    def fn_orig(full_default: "bf16[4800][1]cuda:0", convert_element_type_417: "bf16[4800][1]cuda:0", view_95: "bf16[4800][1]cuda:0"):
        slice_scatter_default_1: "bf16[4800][1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, convert_element_type_417, 0, 0, 4800);  convert_element_type_417 = None
        reduce_scatter_tensor_1: "bf16[1200][1]cuda:0" = torch.ops._c10d_functional.reduce_scatter_tensor.default(slice_scatter_default_1, 'avg', 4, '0');  slice_scatter_default_1 = None
        wait_tensor_163: "bf16[1200][1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        convert_element_type_420: "f32[1200][1]cuda:0" = torch.ops.prims.convert_element_type.default(wait_tensor_163, torch.float32);  wait_tensor_163 = None
        
        slice_scatter_default_2: "bf16[4800][1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, view_95, 0, 0, 4800);  view_95 = None
        reduce_scatter_tensor_2: "bf16[1200][1]cuda:0" = torch.ops._c10d_functional.reduce_scatter_tensor.default(slice_scatter_default_2, 'avg', 4, '0');  slice_scatter_default_2 = None
        wait_tensor_164: "bf16[1200][1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        convert_element_type_425: "f32[1200][1]cuda:0" = torch.ops.prims.convert_element_type.default(wait_tensor_164, torch.float32);  wait_tensor_164 = None
        
        return [convert_element_type_420, convert_element_type_425]

    def fn_concat(full_default: "bf16[4800][1]cuda:0", convert_element_type_417: "bf16[4800][1]cuda:0", view_95: "bf16[4800][1]cuda:0"):
        shard_world_size = 4
        shard_rank = 0
        shard_dim = 0

        slice_scatter_default_1: "bf16[4800][1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, convert_element_type_417, 0, 0, 4800);  convert_element_type_417 = None
        slice_scatter_default_2: "bf16[4800][1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, view_95, 0, 0, 4800);  view_95 = None
        
        unsharded_grads = [slice_scatter_default_1, slice_scatter_default_2]
        # TODO: check all tensors in unsharded_grads have the same dtype
        reduce_dtype = torch.bfloat16  # should get this from the original RS input's dtype
        # Only float32 and bfloat16 are supported for now.
        # To support fp16, please see FSDP2 `_get_gradient_divide_factors`.
        assert reduce_dtype in (torch.float32, torch.bfloat16), f"reduce_dtype {reduce_dtype} is not supported"
        world_size = 4

        def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
            padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
            return cast(torch.Size, torch.Size([padded_dim0]) + tensor_size[1:])

        padded_unsharded_sizes = tuple(
            _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
        )
        reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
        reduce_scatter_input = torch.empty(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )

        def foreach_reduce_scatter_copy_in(
            unsharded_grads: List[torch.Tensor],
            reduce_scatter_input: torch.Tensor,
            world_size: int,
        ) -> None:
            reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
            torch.ops.fsdp.chunk_cat(
                unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
            )

        foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
        reduce_scatter_reduce_op = 'avg'  # should get this from the original RS op
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(reduce_scatter_input, reduce_scatter_reduce_op, world_size, '0')
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor)

        def _chunk_with_empty(
            tensor: torch.Tensor, num_chunks: int, dim: int
        ) -> List[torch.Tensor]:
            chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
            while len(chunks) < num_chunks:
                chunks.append(chunks[0].new_empty(0))
            return chunks

        def _to_dtype_if_needed(
            tensor: torch.Tensor, dtype: Optional[torch.dtype]
        ) -> torch.Tensor:
            if dtype is not None and tensor.dtype != dtype:
                return tensor.to(dtype)
            return tensor

        # orig_dtype = torch.float32  # should get this from the immediately following convert_element_type op (expect all grads to be converted to the same dtype in the original graph)
        # reduce_output = _to_dtype_if_needed(wait_tensor, orig_dtype)
        reduce_output = wait_tensor
        # View out and accumulate sharded gradients
        new_sharded_grads = []
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, unsharded_grad in zip(
            padded_unsharded_sizes, unsharded_grads
        ):
            chunks = _chunk_with_empty(unsharded_grad, shard_world_size, dim=shard_dim)
            sharded_param = chunks[shard_rank]
            sharded_size = sharded_param.size()
            contiguous_sharded_stride = torch._prims_common.make_contiguous_strides_for(sharded_size)
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=sharded_size,
                stride=contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            new_sharded_grads.append(new_sharded_grad)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        return [x.to(torch.float32) for x in new_sharded_grads]

    torch.manual_seed(1337 + local_rank)
    full_default = torch.randn(4800, device="cuda", dtype=torch.bfloat16)
    convert_element_type_417 = torch.randn(4800, device="cuda", dtype=torch.bfloat16)
    view_95 = torch.randn(4800, device="cuda", dtype=torch.bfloat16)
    out_orig = fn_orig(full_default, convert_element_type_417, view_95)
    out_flatten_concat = fn_concat(full_default, convert_element_type_417, view_95)
    assert out_orig[0].shape == out_flatten_concat[0].shape
    assert out_orig[0].dtype == out_flatten_concat[0].dtype
    assert torch.allclose(out_orig[0], out_flatten_concat[0], atol=1e-6, rtol=1e-6)
    assert out_orig[1].shape == out_flatten_concat[1].shape
    assert out_orig[1].dtype == out_flatten_concat[1].dtype
    assert torch.allclose(out_orig[1], out_flatten_concat[1], atol=1e-6, rtol=1e-6)
    