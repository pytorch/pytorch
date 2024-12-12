# python -m torch.distributed.run --standalone --nproc_per_node=4 test_fsdp_all_gather_concat.py

import torch
import os
import logging
torch_log = logging.getLogger("torch")
import logging
log = logging.getLogger(__name__)
from torch.distributed.fsdp._fully_shard import _fsdp_collectives


if __name__ == "__main__":
    import torch.distributed as dist
    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    # world_size = int(os.getenv("WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    def fn_orig(arg55_1: "f32[96, 16][16, 1]cuda:0", arg56_1: "f32[96][1]cuda:0"):
        convert_element_type: "bf16[96, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        all_gather_into_tensor: "bf16[384, 16][16, 1]cuda:0" = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type, 4, '0');  convert_element_type = None
        wait_tensor: "bf16[384, 16][16, 1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        slice_1: "bf16[384, 16][16, 1]cuda:0" = torch.ops.aten.slice.Tensor(wait_tensor, 0, 0, 384);  wait_tensor = None
        convert_element_type_1: "bf16[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        all_gather_into_tensor_1: "bf16[384][1]cuda:0" = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 4, '0');  convert_element_type_1 = None
        wait_tensor_1: "bf16[384][1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        slice_2: "bf16[384][1]cuda:0" = torch.ops.aten.slice.Tensor(wait_tensor_1, 0, 0, 384);  wait_tensor_1 = None
        return (slice_1, slice_2)

    def fn_concat(arg55_1: "f32[96, 16][16, 1]cuda:0", arg56_1: "f32[96][1]cuda:0"):
        group_size = 4
        rank = 0
        group_name = '0'
        dtype = torch.bfloat16
        device = torch.device("cuda", local_rank)
        convert_element_type: "bf16[96, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_1: "bf16[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        param_all_gather_inputs_orig = [convert_element_type, convert_element_type_1]
        param_all_gather_inputs = [t.view(-1) for t in param_all_gather_inputs_orig]
        inp_split_sizes = [t.numel() for t in param_all_gather_inputs]
        param_all_gather_outputs = [
            torch.empty(torch.Size([t.numel() * group_size]), dtype=t.dtype, device=device)
            for t in param_all_gather_inputs
        ]
        param_all_gather_outputs_shape_orig = [(t.shape[0] * group_size,) + t.shape[1:] for t in param_all_gather_inputs_orig]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            param_all_gather_inputs,
            inp_split_sizes,
            all_gather_input_numel,
            group_size,
            rank,
            dtype,
            device,
        )
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor_out.default(all_gather_input, group_size, group_name, out=all_gather_output)
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor)
        all_gather_output = all_gather_output.view(group_size, -1)
        out = [t.view(group_size, -1) for t in param_all_gather_outputs]
        print(f"all_gather_output.shape: {all_gather_output.shape}")
        print(f"inp_split_sizes: {inp_split_sizes}")
        for o in out:
            print(f"o.shape: {o.shape}")
        torch.ops.fsdp.split_with_sizes_copy(
            all_gather_output, inp_split_sizes, dim=1, out=out
        )
        return tuple([t.view(orig_shape) for t, orig_shape in zip(out, param_all_gather_outputs_shape_orig)])

    torch.manual_seed(1337 + local_rank)
    arg55_1 = torch.rand(96, 16, device="cuda", dtype=torch.float32)
    arg56_1 = torch.rand(96, device="cuda", dtype=torch.float32)
    # out_orig = fn_orig(arg55_1, arg56_1)
    # out_concat = fn_concat(arg55_1, arg56_1)
    # assert out_orig[0].shape == out_concat[0].shape
    # assert torch.allclose(out_orig[0], out_concat[0], atol=1e-6, rtol=1e-6)
    # assert out_orig[1].shape == out_concat[1].shape
    # assert torch.allclose(out_orig[1], out_concat[1], atol=1e-6, rtol=1e-6)

    def fn_all_gather_copy_in(arg55_1: "f32[96, 16][16, 1]cuda:0", arg56_1: "f32[96][1]cuda:0"):
        group_size = 4
        rank = 0
        group_name = '0'
        dtype = torch.bfloat16
        device = torch.device("cuda", local_rank)
        convert_element_type: "bf16[96, 16][16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_1: "bf16[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        param_all_gather_inputs_orig = [convert_element_type, convert_element_type_1]
        param_all_gather_inputs = [t.view(-1) for t in param_all_gather_inputs_orig]
        inp_split_sizes = [t.numel() for t in param_all_gather_inputs]
        param_all_gather_outputs = [
            torch.empty(torch.Size([t.numel() * group_size]), dtype=t.dtype, device=device)
            for t in param_all_gather_inputs
        ]
        param_all_gather_outputs_shape_orig = [(t.shape[0] * group_size,) + t.shape[1:] for t in param_all_gather_inputs_orig]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            param_all_gather_inputs,
            inp_split_sizes,
            all_gather_input_numel,
            group_size,
            rank,
            dtype,
            device,
        )
        return all_gather_input, all_gather_output

    # out_fn_all_gather_copy_in_compiled = torch.compile(fn_all_gather_copy_in, backend="inductor", fullgraph=True)(arg55_1, arg56_1)

    def fn_split_with_sizes_copy(all_gather_output, inp_split_sizes, out):
        torch.ops.fsdp.split_with_sizes_copy(
            all_gather_output, inp_split_sizes, dim=1, out=out
        )
        return out

    all_gather_output = torch.rand(4, 1632, device="cuda", dtype=torch.float32)
    inp_split_sizes = [1536, 96]
    out = [
        torch.empty([4, 1536], device="cuda", dtype=torch.float32),
        torch.empty([4, 96], device="cuda", dtype=torch.float32),
    ]
    out_fn_split_with_sizes_copy_compiled = torch.compile(fn_split_with_sizes_copy, backend="inductor", fullgraph=True)(all_gather_output, inp_split_sizes, out)
