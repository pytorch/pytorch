# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed.fsdp import DataParallelMeshDimNames, fully_shard
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class TestFullyShardDTensor(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    def _run_train_parity(
        self, model, ref_model, dp_pg, mesh=None, num_iters=5, mlp_dim=16
    ):
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        torch.manual_seed(42 + dp_pg.rank() + 1)
        for i in range(num_iters):
            inp = torch.randn((2, mlp_dim), device=device_type)
            ref_optim.zero_grad(set_to_none=(i % 2 == 0))
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            ref_optim.step()

            optim.zero_grad(set_to_none=(i % 2 == 0))
            if mesh is not None:
                inp = DTensor.from_local(
                    inp, mesh, [Replicate()] * mesh.ndim, run_check=False
                )
            loss = model(inp).sum()
            loss.backward()
            optim.step()

            self.assertEqual(ref_loss, loss)

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters()
        ):
            p2_full = p2.full_tensor() if isinstance(p2, DTensor) else p2
            self.assertEqual(p1, p2_full, msg=f"Param mismatch: {n1} vs {n2}")

    @skip_if_lt_x_gpu(2)
    def test_fsdp_dtensor_1d_train_parity(self):
        """FSDP with 1D mesh, all params DTensors with Replicate()."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        dp_pg = mesh.get_group()

        torch.manual_seed(42)
        model = MLP(16, device=device_type)
        ref_model = copy.deepcopy(model)

        distribute_module(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dim_names=DataParallelMeshDimNames(shard="fsdp"),
        )

        replicate(
            ref_model,
            device_ids=[self.rank] if device_type.type != "cpu" else None,
            process_group=dp_pg,
        )

        self._run_train_parity(model, ref_model, dp_pg, mesh=mesh)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_dtensor_train_parity(self):
        """HSDP with 2D mesh, all params DTensors with (Replicate, Replicate)."""
        dp_size = 2
        shard_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, shard_size),
            mesh_dim_names=("ddp", "fsdp"),
        )

        torch.manual_seed(42)
        model = MLP(16, device=device_type)
        ref_model = copy.deepcopy(model)

        distribute_module(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dim_names=DataParallelMeshDimNames(shard="fsdp", replicate="ddp"),
        )

        replicate(
            ref_model,
            device_ids=[self.rank] if device_type.type != "cpu" else None,
            process_group=dist.group.WORLD,
        )

        self._run_train_parity(model, ref_model, dist.group.WORLD, mesh=mesh)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_multi_shard_dtensor_train_parity(self):
        """FSDP with multi-shard dims flattened into one, all params DTensors."""
        dp0_size = 2
        dp1_size = self.world_size // dp0_size
        mesh = init_device_mesh(
            device_type.type,
            (dp0_size, dp1_size),
            mesh_dim_names=("dp0", "dp1"),
        )

        torch.manual_seed(42)
        model = MLP(16, device=device_type)
        ref_model = copy.deepcopy(model)

        distribute_module(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dim_names=DataParallelMeshDimNames(shard=("dp0", "dp1")),
        )

        replicate(
            ref_model,
            device_ids=[self.rank] if device_type.type != "cpu" else None,
            process_group=dist.group.WORLD,
        )

        self._run_train_parity(model, ref_model, dist.group.WORLD, mesh=mesh)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_dtensor_sharded_params(self):
        """Verify sharded param mesh and placements for FSDP+TP on 2D mesh."""
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )

        mlp_dim = 16
        model = MLP(mlp_dim, device=device_type)

        def partition_fn(name, module, device_mesh):
            if not isinstance(module, nn.Linear):
                return
            for param_name, param in list(module.named_parameters(recurse=False)):
                if param_name == "weight":
                    if "in_proj" in name:
                        placements = [Replicate(), Shard(0)]
                    else:
                        placements = [Replicate(), Shard(1)]
                else:
                    placements = [Replicate(), Replicate()]
                dist_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, placements),
                    requires_grad=param.requires_grad,
                )
                module.register_parameter(param_name, dist_param)

        distribute_module(model, mesh, partition_fn)

        def shard_fn(param):
            if any(isinstance(p, Shard) and p.dim == 0 for p in param.placements):
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dim_names=DataParallelMeshDimNames(shard="fsdp"),
        )

        for name, param in model.named_parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device_mesh, mesh)
            self.assertEqual(len(param.placements), 2)
            if "in_proj.weight" in name:
                # FSDP shards dim 1 (avoiding TP dim 0), TP shards dim 0
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 1)
                self.assertIsInstance(param.placements[1], Shard)
                self.assertEqual(param.placements[1].dim, 0)
            elif "out_proj.weight" in name:
                # FSDP shards dim 0 (default), TP shards dim 1
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 0)
                self.assertIsInstance(param.placements[1], Shard)
                self.assertEqual(param.placements[1].dim, 1)
            elif "bias" in name:
                # FSDP shards dim 0 (default), TP replicates
                self.assertIsInstance(param.placements[0], Shard)
                self.assertEqual(param.placements[0].dim, 0)
                self.assertIsInstance(param.placements[1], Replicate)

    @skip_if_lt_x_gpu(2)
    def test_sharded_param_correctness_1d(self):
        """Verify sharded param mesh and placements for FSDP on 1D mesh."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )

        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dim_names=DataParallelMeshDimNames(shard="fsdp"),
        )

        for param in model.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device_mesh, mesh)
            self.assertEqual(len(param.placements), 1)
            self.assertIsInstance(param.placements[0], Shard)

    @skip_if_lt_x_gpu(2)
    def test_validation_non_replicate_dp_placement(self):
        """Error when a param has non-Replicate placement on the DP shard dim."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = nn.Linear(16, 32, device=device_type)
        # Distribute weight with Shard(0) on the FSDP dim
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.data, mesh, [Shard(0)]),
            requires_grad=True,
        )
        model.bias = nn.Parameter(
            distribute_tensor(model.bias.data, mesh, [Replicate()]),
            requires_grad=True,
        )
        with self.assertRaisesRegex(ValueError, "Expected Replicate"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dim_names=DataParallelMeshDimNames(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_invalid_dim_names(self):
        """Error when dp_mesh_dim_names references nonexistent mesh dim names."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(ValueError, "not found in mesh.mesh_dim_names"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dim_names=DataParallelMeshDimNames(shard="nonexistent"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_at_least_one_required(self):
        """Error when neither shard nor replicate is set."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(ValueError, "At least one of shard or replicate"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dim_names=DataParallelMeshDimNames(),
            )


if __name__ == "__main__":
    run_tests()
