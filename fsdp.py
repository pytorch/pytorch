"""
torchrun --standalone --nproc_per_node=2 fsdp.py
"""
import os

import torch
import torch._dynamo
from torch._dynamo import compiled_autograd
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import contextlib

def init():
    torch.manual_seed(0)
    fsdp_kwargs = {
        "use_orig_params": True,
        "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
    }
    model = nn.Sequential(
        nn.Linear(3, 3, device="cuda"), nn.ReLU(), nn.Linear(3, 3, device="cuda")
    )
    model = FSDP(
        model,
        **fsdp_kwargs,
    )
    # TODO: Add `model = torch.compile(model)` here if desired
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optim


def printing_eager(gm, inputs):
    gm.graph.print_tabular()
    return gm.forward

gpu_id = int(os.environ["LOCAL_RANK"])

def run(model, optim):
    torch.manual_seed(42)
    losses = []
    inp = torch.randn((2, 3), device="cuda")

    for _ in range(4):
        optim.zero_grad(set_to_none=True)
        inp = torch.randn((2, 3), device="cuda")
        torch.storage.resize_count_and_loc = {}
        if gpu_id == 0:
            print("FORWARD")
        out = model(inp)
        if gpu_id == 0:
            print("END FORWARD")
        # torch.storage.resize_count_and_loc = {}
        loss = out.sum()
        losses.append(loss)
        torch.storage.resize_count_and_loc = {}
        if gpu_id == 0:
            print("BACKWARD")
        loss.backward()
        if gpu_id == 0:
            print("END BACKWARD")
        optim.step()
    return losses


def main(compiled):
    model, optim = init()

    def compiler_fn(gm):
        print("Compiling autograd?")
        return torch.compile(gm, backend="eager", fullgraph=True, dynamic=False)

    compile_bwd = True
    ctx = compiled_autograd.enable(compiler_fn) if compile_bwd else contextlib.nullcontext()

    if compiled:
        with ctx:
            print("RUNNING COMPILE")
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            if gpu_id == 0:
                params = {
                    **dict(model.named_parameters(remove_duplicate=False)),
                    **dict(model.named_buffers(remove_duplicate=False)),
                }
                for name, param in params.items():
                    print(f"Pre compile. {name} {param.size()}")
            model = torch._dynamo.optimize("eager", nopython=True, dynamic=False)(model)
            res = run(model, optim)
    else:
        res = run(model, optim)
    return res




if __name__ == "__main__":
    import time

    dist.init_process_group(backend="nccl")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    # eager = main(compiled=False)
    # print("EAGER:", eager)
    # exit(0)
    print("--------------------")
    # time.sleep(5)
    print("--------------------")
    compiled = main(compiled=True)
    print("COMPILED:", compiled)
