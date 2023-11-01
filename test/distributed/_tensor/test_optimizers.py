# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch

import torch.nn as nn

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)


# shard function to do full sharding on all parameters of a module
def shard_fn(name, module, device_mesh):
    if isinstance(module, nn.Linear):
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            # make sure partial sum get cleared after backward()
            dist_param.register_hook(
                lambda grad: grad.redistribute(placements=[Shard(0)])
            )
            module.register_parameter(name, dist_param)


# prepare input
def input_fn(inputs, device_mesh):
    # split the input tensor to be sharded input
    dist_inp = distribute_tensor(inputs[0], device_mesh, [Shard(0)])
    return dist_inp


# prepare output to be local torch.Tensor
def output_fn(outputs, device_mesh):
    assert isinstance(outputs, DTensor)
    return outputs.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()


class TestDTensorOptimizer(DTensorTestBase):
    def _assert_optimizer(
        self,
        mesh,
        model,
        optim,
        dist_model,
        dist_optim,
        inputs,
    ):
        # run forward/backward/optim for original model
        optim.zero_grad()
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        optim.step()

        # run forward/backward/optim for distributed model
        dist_optim.zero_grad()
        dist_out = dist_model(inputs)
        # dist_out = dist_out.redistribute(placements=[Replicate()] * mesh.ndim)
        dist_loss = dist_out.sum()
        dist_loss.backward()
        dist_optim.step()

        # check that the optimizer update parameters with same numerics
        for p1, p2 in zip(model.parameters(), dist_model.parameters()):
            # turn p2 to full replication for comparison
            p2 = p2.redistribute(placements=[Replicate()] * mesh.ndim)
            p2 = p2.to_local()
            self.assertEqual(p1, p2)

    @with_comms
    def test_adam_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # TODO: add fused_adam support
        adam_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "foreach": True},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": True},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True, "foreach": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "foreach": True,
            },
        ]

        for config in adam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)
