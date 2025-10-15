# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
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
        *,
        rtol: float = 1.3e-6,
        atol: float = 1e-5,
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
                # Default 'rtol' and 'atol' for attr:`~torch.float32` are ``1.3e-6`` and ``1e-5``
                self.assertEqual(p1, p2, atol=atol, rtol=rtol)

    def test_optimizer_foreach_supported_types_include_DTensor(self):
        from torch.optim.optimizer import _foreach_supported_types

        self.assertTrue(DTensor in _foreach_supported_types)

    @with_comms
    def test_adam_1d_sharding(self):
        mesh = self.build_device_mesh()

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adam_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adam_float_lr_configs = [
            {"lr": 0.1, "fused": True},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True, "fused": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adam optimizer
        fused_adam_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adam_float_lr_configs
        ]
        fused_adam_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adam_float_lr_configs
            ]
        )
        adam_configs = [
            *adam_float_lr_configs,
            *fused_adam_float_lr_configs,
            *fused_adam_tensor_lr_configs,
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
        mesh = self.build_device_mesh()

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adamw_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adamw_float_lr_configs = [
            {"lr": 0.1, "weight_decay": 0.05, "fused": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "fused": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adamW optimizer
        fused_adamw_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adamw_float_lr_configs
        ]
        fused_adamw_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adamw_float_lr_configs
            ]
        )
        adamw_configs = [
            *adamw_float_lr_configs,
            *fused_adamw_float_lr_configs,
            *fused_adamw_tensor_lr_configs,
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
        mesh = self.build_device_mesh()

        sgd_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "momentum": 0.05, "foreach": False},
            {"lr": 0.1, "momentum": 0.05},
            {"lr": 0.1, "momentum": 0.06, "dampening": 0.07},
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
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
        mesh = self.build_device_mesh()

        adagrad_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "lr_decay": 0.05, "foreach": False},
            {"lr": 0.1, "lr_decay": 0.02, "weight_decay": 0.05, "foreach": False},
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
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
        mesh = self.build_device_mesh()

        RMSprop_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "alpha": 0.85, "foreach": False},
            {"lr": 0.1, "alpha": 0.88, "eps": 1e-6, "foreach": False},
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
                "foreach": False,
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
        mesh = self.build_device_mesh()

        adadelta_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "rho": 0.85, "foreach": False},
            {"lr": 0.1, "rho": 0.88, "eps": 1e-5, "foreach": False},
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
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
        mesh = self.build_device_mesh()

        nadam_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
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
        mesh = self.build_device_mesh()

        radam_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
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
        mesh = self.build_device_mesh()

        adamax_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "betas": (0.6, 0.66), "foreach": False},
            {"lr": 0.1, "betas": (0.6, 0.66), "eps": 1e-6, "foreach": False},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
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

    @with_comms
    def test_asgd_1d_sharding(self):
        mesh = self.build_device_mesh()

        asgd_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "foreach": False},
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "t0": 1e5, "foreach": False},
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": False,
            },
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
            },
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
                "maximize": True,
            },
        ]

        for config in asgd_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.ASGD(mod.parameters(), **config)

            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.ASGD(dist_mod.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)

            # TODO: We want to keep a unit test for ASGD optimizer for the time being, but we need to look into why
            # when using ASGD we need higher atol and rtol when comparing model parameters.
            # Default 'rtol' and 'atol' for attr:`~torch.float32` are ``1.3e-6`` and ``1e-5``
            # Pointer here: https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L65
            self._assert_optimizer(
                mesh, mod, opt, dist_mod, dist_opt, inp, atol=1.3e-5, rtol=1e-4
            )

    @with_comms
    def test_admaw_fused_across_meshes(self):
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(
            self.device_type, mesh_shape, mesh_dim_names=("x", "y")
        )
        mesh_flatten = mesh_2d[("x", "y")]._flatten(mesh_dim_name="mesh_flatten")

        # lr as a Tensor is not supported for capturable=False and foreach=True
        adamw_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        fused_adamw_float_lr_configs = [
            {"lr": 0.1, "weight_decay": 0.05, "fused": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "fused": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # lr could be a Tensor or a float when fused=True for adamW optimizer
        fused_adamw_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adamw_float_lr_configs
        ]
        fused_adamw_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adamw_float_lr_configs
            ]
        )
        adamw_configs = [
            *adamw_float_lr_configs,
            *fused_adamw_float_lr_configs,
            *fused_adamw_tensor_lr_configs,
        ]

        # shard function to do full sharding on all parameters of a module
        def _shard_fn_2d(name, module, device_mesh):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, [Replicate(), Shard(0)])
                    )
                    # make sure partial sum get cleared after backward()
                    dist_param.register_hook(
                        lambda grad: grad.redistribute(
                            placements=[Replicate(), Shard(0)]
                        )
                    )
                    module.register_parameter(name, dist_param)

        # prepare input
        def _input_fn_2d(mod, inputs, device_mesh):
            # split the input tensor to be sharded input on a 2d mesh
            dist_inp = DTensor.from_local(
                inputs[0], device_mesh, [Replicate(), Shard(0)], run_check=False
            )
            return dist_inp

        for config in adamw_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.AdamW(mod.parameters(), **config)

            mod_copy = deepcopy(mod)
            # MLPModule.net1 is sharded on the flatten mesh
            distribute_module(
                mod_copy.net1, mesh_flatten, shard_fn, input_fn, output_fn
            )
            # MLPModule.net2 is sharded on the 2d mesh
            distribute_module(
                mod_copy.net2, mesh_2d, _shard_fn_2d, _input_fn_2d, output_fn
            )
            dist_opt = torch.optim.AdamW(mod_copy.parameters(), **config)

            # use ones to make sure the single machine model have the same input
            # on different ranks
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(None, mod, opt, mod_copy, dist_opt, inp)


if __name__ == "__main__":
    run_tests()
