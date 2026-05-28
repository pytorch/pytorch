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
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)


class MLPModule(nn.Module):
    def __init__(self, device, bias: bool = True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class TPCommTest(unittest.TestCase):
    def _compare_params(
        self,
        local_module,
        dist_module,
        rank0_only,
        skip_rowwise_bias=False,
        compare_grad=False,
    ):
        replicate = [Replicate()]
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            param = param.grad if compare_grad else param
            dist_param = dist_param.grad if compare_grad else dist_param
            if (
                (not rank0_only)
                or (self.rank == 0)
                or (
                    name not in ["net2.bias"]
                    and not skip_rowwise_bias
                    or name not in ["bias", "net2.bias"]
                )
            ):
                local_param_full_tensor = param
                tp_param_full_tensor = dist_param.redistribute(
                    device_mesh=dist_param.device_mesh, placements=replicate
                ).to_local()

                self.assertTrue(
                    torch.equal(local_param_full_tensor, tp_param_full_tensor)
                )

    @unittest.skipIf(
        torch.accelerator.device_count() < 2, "Skipping non GPU situations for now"
    )
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

        rank0_only = False
        rowwise = False
        inp_size = [12, 10]
        model = MLPModule(device)
        torch.manual_seed(0)

        model_tp = copy.deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            },
        )
        LR = 8e-4  # Use the learning rate from torchtitan
        local_optim = torch.optim.SGD(model.parameters(), lr=LR)
        dist_optim = torch.optim.SGD(model_tp.parameters(), lr=LR)
        self._compare_params(model, model_tp, rank0_only)
        for _ in range(30):
            inp = torch.rand(*inp_size, device=device)
            output = model(inp)
            tp_output = model_tp(inp)
            tp_output = (
                tp_output.redistribute(tp_output.device_mesh, [Replicate()]).to_local()
                if isinstance(tp_output, DTensor)
                else tp_output
            )
            # there's slight difference between losses, so we check the difference
            loss_diff = output.sum() - tp_output.sum()
            self.assertLess(abs(loss_diff.item()), 1e-5)

            output.sum().backward()
            tp_output.sum().backward()

            # check backward and ensure gradients are same
            self._compare_params(model, model_tp, rank0_only, rowwise, True)

            local_optim.step()
            dist_optim.step()
            self._compare_params(model, model_tp, rank0_only, rowwise)

            local_optim.zero_grad()
            dist_optim.zero_grad()

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
