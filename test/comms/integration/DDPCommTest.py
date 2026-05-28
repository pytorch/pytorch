#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torch.comms
import torch.nn as nn
from torch.comms.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPCommTest(unittest.TestCase):
    def test_training(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

        comm = torch.comms.new_comm(backend, device, name="comms_test_ddp")

        try:
            device_mesh = init_device_mesh(
                mesh_dim_comms=(comm,),
                mesh_dim_names=("main",),
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                comm.finalize()
                return
            raise

        pg = device_mesh.get_group("main")

        torch.manual_seed(42)
        dim0 = 4
        nlayer = 2
        model = nn.Sequential(
            *[nn.Linear(dim0, dim0, bias=False, device=device) for _ in range(nlayer)]
        )

        model = DDP(model, process_group=pg)

        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        # Create input on CPU for reproducibility across ranks, then move
        inp = torch.randn((4, dim0)).to(device)

        prev_loss = None
        for i in range(10):
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss_val = loss.item()
            if prev_loss is not None and i < 5:
                # Verify loss is changing (training is progressing)
                assert loss_val != prev_loss, (
                    f"loss did not change at step {i}: {loss_val}"
                )
            prev_loss = loss_val

        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
