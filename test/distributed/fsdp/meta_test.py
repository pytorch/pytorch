# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import always_wrap_policy as default


class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.lin1 = nn.Linear(2, 2, bias=False, device=device)
        self.lin2 = nn.Linear(2, 2, bias=False, device=device)

    def forward(self, x):
        return self.lin2(self.lin1(x))


def run_test_with_meta():
    model = MyModel(device="meta")
    assert next(model.lin1.parameters()).is_meta
    assert next(model.lin2.parameters()).is_meta

    def init_fn(module):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        try:
            with torch.no_grad():
                module.reset_parameters()
        except BaseException:
            pass

    # Eventually the API will be as simple as FSDP(
    #    model,
    #    fsdp_auto_wrap_policy=auto_wrap_policy
    # )
    # where we will use materialize_module() API to initialize the module
    fsdp_model = FSDP(
        model,
        fsdp_auto_wrap_policy=default,
        param_init_fns=init_fn,
    )
    print(fsdp_model)

    # Test to make sure it is the same model parameters as regular FSDP
    # approach.
    regular = MyModel(device="cuda")
    regular.apply(init_fn)
    fsdp_regular = FSDP(regular, fsdp_auto_wrap_policy=default)

    for m1, m2 in zip(fsdp_model.modules(), fsdp_regular.modules()):
        p1 = list(m1.parameters())
        p2 = list(m2.parameters())
        for x, y in zip(p1, p2):
            assert torch.allclose(x, y), f"{x} {y} "

    print(f"Initialized FSDP model")

def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)

    run_test_with_meta()



if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())
