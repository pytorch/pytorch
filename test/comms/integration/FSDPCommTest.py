#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os
import unittest

import torch
import torch.comms
import torch.nn as nn
from torch.comms.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard


class FSDPCommTest(unittest.TestCase):
    def test_training(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

        comm = torch.comms.new_comm(backend, device, name="comms_test_name")

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

        torch.manual_seed(42)
        dim0 = 4
        nlayer = 2
        model = nn.Sequential(
            *[
                torch.nn.Linear(dim0, dim0, bias=False, device=device)
                for _ in range(nlayer)
            ]
        )
        ref_model = copy.deepcopy(model)
        # set_gradient_divide_factor to focus reduceOp as sum,
        # reduceOp.AVG is not supported yet.
        for layer in model:
            fully_shard(layer, mesh=device_mesh)
            if isinstance(layer, FSDPModule):
                layer.set_gradient_divide_factor(1.0)
        fully_shard(model, mesh=device_mesh)

        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=0.05)
        inp = torch.randn((4, dim0), device=device)

        for _ in range(10):
            loss = model(inp).sum()
            ref_loss = ref_model(inp).sum()
            assert torch.allclose(loss, ref_loss, atol=1e-7, rtol=1e-5)

            loss.backward()
            ref_loss.backward()
            optim.step()
            ref_optim.step()
            optim.zero_grad()
            ref_optim.zero_grad()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
