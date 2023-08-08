"""
torchrun --standalone --nproc_per_node=2 test_basic.py
"""
import os

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def init():
    torch.manual_seed(0)
    fsdp_kwargs = {
        "use_orig_params": True,
        "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
    }
    model = nn.Linear(3, 3, device="cuda")
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
    torch.manual_seed(42)
    losses = []
    inp = torch.randn((2, 3), device="cuda")

    # explain = torch._dynamo.explain(model, inp)
    # for g in explain.graphs:
    #     g.graph.print_tabular()
    # for i, gb in enumerate(explain.break_reasons):
    #     print(f"{i}. {gb}")
    # sorted_dict = dict(sorted(torch._dynamo.exc.unimpl_and_count.items(), key=lambda x: x[1], reverse=True))
    # i = 0
    # for k, v in sorted_dict.items():
    #     print(f"{i}. {k} - {v}")
    #     i += 1
    # print("Compiling")
    # model = torch._dynamo.optimize("eager")(model)

    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        inp = torch.randn((2, 3), device="cuda")
        out = model(inp)
        loss = out.sum()
        losses.append(loss)
        loss.backward()
        optim.step()
    return losses


def main(compiled):
    model, optim = init()
    if compiled:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        model = torch._dynamo.optimize("eager", nopython=True)(model)
    return run(model, optim)


gpu_id = int(os.environ["LOCAL_RANK"])


def tracefunc(frame, event, arg, indent=[0]):
    if gpu_id == 0:
        if event == "call":
            indent[0] += 2
            print("-" * indent[0] + "> call function", frame.f_code.co_name)
        elif event == "return":
            print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
            indent[0] -= 2
        return tracefunc


import sys

# sys.setprofile(tracefunc)

if __name__ == "__main__":
    import time

    dist.init_process_group(backend="nccl")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    eager = main(compiled=False)
    print("EAGER:", eager)
    time.sleep(2)
    compiled = main(compiled=True)
    print("COMPILED:", compiled)
    # for i in range(0, len(eager)):
    # distance = torch.norm(eager[i] - compiled[i], p=2)  # p=2 specifies the Euclidean norm
    # output_file = f"output_{os.getpid()}.txt"
    # with open(output_file, 'a') as f:
    # f.write(f"DIST AT {i} : {str(distance.item())}")
    # print("DIST AT", i, distance.item())
    # breakpoint()
