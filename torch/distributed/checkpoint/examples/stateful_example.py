# Owner(s): ["oncall: distributed"]


import os
import shutil
from enum import auto, Enum
from functools import partial

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed.checkpoint as DCP
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")

def _make_stateful(model, optim):
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)

def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def run(world_size, device="cuda"):

    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)

    device_mesh = init_device_mesh(device, (world_size,))
    model = FSDP(
        dummy_model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )

    _train(model, optim, train_steps=2)

    DCP.save(
        state_dict={"model": model, "optimizer": optim},
        storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
    )

    # presumably do something else

    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)
    DCP.load(
        state_dict={"model": dist_model, "optimizer": dist_optim},
        storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
    )
    _train(model, optim, train_steps=2)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
