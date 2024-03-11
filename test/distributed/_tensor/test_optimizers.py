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
from torch.testing._internal.common_utils import run_tests

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
def input_fn(mod, inputs, device_mesh):
    # split the input tensor to be sharded input
    dist_inp = distribute_tensor(inputs[0], device_mesh, [Shard(0)])
    return dist_inp


# prepare output to be local torch.Tensor
def output_fn(mod, outputs, device_mesh):
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
        for iter_idx in range(2):
            # run forward/backward/optim for original model
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            out = model(inputs)
            loss = out.sum()
            loss.backward()
            optim.step()

            # run forward/backward/optim for distributed model
            dist_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            dist_out = dist_model(inputs)
            dist_loss = dist_out.sum()
            dist_loss.backward()
            dist_optim.step()

            # check that the optimizer update parameters with same numerics
            for p1, p2 in zip(model.parameters(), dist_model.parameters()):
                p2 = p2.full_tensor()
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

    @with_comms
    def test_adamw_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # TODO: add fused_adamw support
        adamw_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "foreach": True,
            },
        ]

        for config in adamw_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.AdamW(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.AdamW(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_sgd_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        sgd_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.05},
            {"lr": 0.1, "momentum": 0.05, "foreach": True},
            {"lr": 0.1, "momentum": 0.06, "dampening": 0.07, "foreach": True},
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
            },
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
                "foreach": True,
            },
        ]

        for config in sgd_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.SGD(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.SGD(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adagrad_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        adagrad_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "lr_decay": 0.05},
            {"lr": 0.1, "lr_decay": 0.02, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
                "foreach": True,
            },
        ]

        for config in adagrad_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adagrad(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adagrad(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_RMSprop_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        RMSprop_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "alpha": 0.85},
            {"lr": 0.1, "alpha": 0.88, "eps": 1e-6},
            {"lr": 0.1, "alpha": 0.88, "eps": 1e-6, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
                "foreach": True,
            },
        ]

        for config in RMSprop_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.RMSprop(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.RMSprop(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adadelta_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        adadelta_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "rho": 0.85},
            {"lr": 0.1, "rho": 0.88, "eps": 1e-5},
            {"lr": 0.1, "rho": 0.88, "eps": 1e-6, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
                "maximize": True,
            },
        ]

        for config in adadelta_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adadelta(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adadelta(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_nadam_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        nadam_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
                "foreach": True,
            },
        ]

        for config in nadam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.NAdam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.NAdam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_radam_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        radam_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
                "foreach": True,
            },
        ]

        for config in radam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.RAdam(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.RAdam(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adamax_1d_sharding(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        adamax_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "betas": (0.6, 0.66)},
            {"lr": 0.1, "betas": (0.6, 0.66), "eps": 1e-6},
            {"lr": 0.1, "betas": (0.6, 0.66), "eps": 1e-6, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": True,
                "maximize": True,
            },
        ]

        for config in adamax_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adamax(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adamax(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)


if __name__ == "__main__":
    run_tests()
