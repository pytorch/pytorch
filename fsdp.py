"""
torchrun --standalone --nproc_per_node=2 test_basic.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch._dynamo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
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

def run(model, optim):
    losses = []
    torch.manual_seed(dist.get_rank() + 1)
    inp = torch.randn((2, 3), device="cuda")
    # torch._dynamo.optimize(printing_eager)(model)(inp)
    explain = torch._dynamo.explain(model, inp)
    for g in explain.graphs:
        g.graph.print_tabular()
    for i, gb in enumerate(explain.break_reasons):
        print(f"{i}. {gb}")
    sorted_dict = dict(sorted(torch._dynamo.exc.unimpl_and_count.items(), key=lambda x: x[1], reverse=True))

    i = 0
    for k, v in sorted_dict.items():
        print(f"{i}. {k} - {v}")
        i += 1
    # for _ in range(3):
    #     optim.zero_grad(set_to_none=True)
    #     inp = torch.randn((2, 3), device="cuda")
    #     out = model(inp)
    #     loss = out.sum()
    #     losses.append(loss)
    #     loss.backward()
    #     optim.step()
    return losses

def main():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    model, optim = init()
    run(model, optim)

if __name__ == "__main__":
    main()
