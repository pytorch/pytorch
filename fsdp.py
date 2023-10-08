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
import logging
import sys
import traceback

torch_log = logging.getLogger("torch")

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    torch_log.error("Uncaught exception\n%s", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

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

    for _ in range(1):
        optim.zero_grad(set_to_none=True)
        inp = torch.randn((2, 3), device="cuda")
        torch.storage.resize_count_and_loc = {}
        torch_log.warning("FORWARD")
        for p in model.parameters():
            torch_log.warning(f"PRE STATE: {p.shape}")
        out = model(inp)
        torch_log.warning("END FORWARD")
        # torch.storage.resize_count_and_loc = {}
        loss = out.sum()
        losses.append(loss)
        torch.storage.resize_count_and_loc = {}
        for p in model.parameters():
            torch_log.warning(f"POST STATE: {p.shape}")
        torch_log.warning("BACKWARD")
        loss.backward()
        torch_log.warning("END BACKWARD")
        optim.step()
    return losses


def main(compiled_fwd, compiled_bwd, aot_eager):
    model, optim = init()

    def compiler_fn(gm):
        torch_log.warning("Compiling autograd?")
        return torch.compile(gm, backend="eager", fullgraph=True, dynamic=False)

    ctx = compiled_autograd.enable(compiler_fn) if compiled_bwd else contextlib.nullcontext()

    with ctx:
        if compiled_fwd:
                torch_log.warning("RUNNING COMPILE")
                torch._dynamo.config.capture_dynamic_output_shape_ops = True
                torch._dynamo.config.capture_scalar_outputs = True
                model = torch._dynamo.optimize("aot_eager" if aot_eager else "eager", nopython=True, dynamic=False)(model)
                res = run(model, optim)
        else:
            res = run(model, optim)
    return res


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--compiled-fwd', action='store_true')
    parser.add_argument('--compiled-bwd', action='store_true')
    parser.add_argument('--aot-eager', action='store_true')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    res = main(compiled_fwd=args.compiled_fwd, compiled_bwd=args.compiled_bwd, aot_eager=args.aot_eager)
    torch_log.warning("res: %s", res)
