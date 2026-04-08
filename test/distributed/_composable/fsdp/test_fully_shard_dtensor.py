# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed._composable.replicate_with_fsdp import (
    replicate as replicate_with_fsdp,
)
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
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


def _tp_partition_fn(name, module, device_mesh):
    """Partition Linear weights across the last mesh dim for TP."""
    if not isinstance(module, nn.Linear):
        return
    num_non_tp_dims = device_mesh.ndim - 1
    replicate_prefix = [Replicate()] * num_non_tp_dims
    for param_name, param in list(module.named_parameters(recurse=False)):
        if param_name == "weight":
            if "in_proj" in name:
                placements = replicate_prefix + [Shard(0)]
            else:
                placements = replicate_prefix + [Shard(1)]
        else:
            placements = replicate_prefix + [Replicate()]
        dist_param = nn.Parameter(
            distribute_tensor(param, device_mesh, placements),
            requires_grad=param.requires_grad,
        )
        module.register_parameter(param_name, dist_param)


def _tp_shard_fn(param):
    """FSDP shard placement that avoids the existing TP shard dim."""
    if any(isinstance(p, Shard) and p.dim == 0 for p in param.placements):
        return Shard(1)
    return Shard(0)


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
                # Use Replicate on all dims: each rank computes on the full
                # input (like DDP). DTensor handles the TP-sharded params.
                inp = DTensor.from_local(
                    inp, mesh, [Replicate()] * mesh.ndim, run_check=False
                )
            loss = model(inp).sum()
            loss.backward()
            optim.step()

            loss_cmp = loss.full_tensor() if isinstance(loss, DTensor) else loss
            self.assertEqual(ref_loss, loss_cmp)

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters(), strict=True
        ):
            p2_full = p2.full_tensor() if isinstance(p2, DTensor) else p2
            self.assertEqual(p1, p2_full, msg=f"Param mismatch: {n1} vs {n2}")

    @skip_if_lt_x_gpu(2)
    def test_dtensor_train_parity(self):
        """Train parity for FSDP/HSDP/DDP with DTensors on SPMD meshes."""
        ws = self.world_size
        world_mesh = init_device_mesh(
            device_type.type, (ws,), mesh_dim_names=("world",)
        )
        # (sizes, names, dp_dims, use_tp, reshard, dp_pg_source, use_rep_fsdp)
        cases = [
            # 1D: FSDP
            (
                (ws,),
                ("fsdp",),
                DataParallelMeshDims(shard="fsdp"),
                False,
                True,
                None,
                False,
            ),
            # 1D: FSDP with reshard_after_forward=False
            (
                (ws,),
                ("fsdp0",),
                DataParallelMeshDims(shard="fsdp0"),
                False,
                False,
                None,
                False,
            ),
            # 1D: DDP-only
            (
                (ws,),
                ("ddp",),
                DataParallelMeshDims(replicate="ddp"),
                False,
                True,
                None,
                False,
            ),
            # 1D: replicate_with_fsdp
            (
                (ws,),
                ("ddp0",),
                DataParallelMeshDims(replicate="ddp0"),
                False,
                True,
                None,
                True,
            ),
        ]
        if ws >= 4:
            cases.extend(
                [
                    # HSDP 2D
                    (
                        (2, ws // 2),
                        ("rep", "shard"),
                        DataParallelMeshDims(shard="shard", replicate="rep"),
                        False,
                        True,
                        "world",
                        False,
                    ),
                    # Multi-shard FSDP
                    (
                        (2, ws // 2),
                        ("dp0", "dp1"),
                        DataParallelMeshDims(shard=("dp0", "dp1")),
                        False,
                        True,
                        "world",
                        False,
                    ),
                    # FSDP+TP
                    (
                        (2, ws // 2),
                        ("fsdp1", "tp"),
                        DataParallelMeshDims(shard="fsdp1"),
                        True,
                        True,
                        "fsdp1",
                        False,
                    ),
                    # FSDP+TP with reshard_after_forward=False
                    (
                        (2, ws // 2),
                        ("fsdp2", "tp0"),
                        DataParallelMeshDims(shard="fsdp2"),
                        True,
                        False,
                        "fsdp2",
                        False,
                    ),
                    # HSDP+TP 3D
                    (
                        (1, ws // 2, 2),
                        ("rep0", "fsdp3", "tp1"),
                        DataParallelMeshDims(shard="fsdp3", replicate="rep0"),
                        True,
                        True,
                        "fsdp3",
                        False,
                    ),
                    # Multi-dim replicate
                    (
                        (1, ws // 2, 2),
                        ("ddp1", "ddp2", "fsdp4"),
                        DataParallelMeshDims(shard="fsdp4", replicate=("ddp1", "ddp2")),
                        False,
                        True,
                        "world",
                        False,
                    ),
                ]
            )
        mlp_dim = 16
        for sizes, names, dp_dims, use_tp, reshard, dp_pg_src, use_rep in cases:
            with self.subTest(
                names=names, use_tp=use_tp, reshard=reshard, use_rep=use_rep
            ):
                mesh = world_mesh._unflatten(0, sizes, names)

                torch.manual_seed(42)
                model = MLP(mlp_dim, device=device_type)
                ref_model = copy.deepcopy(model)

                partition_fn = _tp_partition_fn if use_tp else None
                distribute_module(model, mesh, partition_fn)

                if use_rep:
                    replicate_with_fsdp(model, mesh=mesh, dp_mesh_dims=dp_dims)
                else:
                    shard_fn = _tp_shard_fn if use_tp else None
                    fully_shard(
                        model,
                        mesh=mesh,
                        reshard_after_forward=reshard,
                        shard_placement_fn=shard_fn,
                        dp_mesh_dims=dp_dims,
                    )

                if dp_pg_src is None:
                    dp_pg = mesh.get_group()
                elif dp_pg_src == "world":
                    dp_pg = dist.group.WORLD
                else:
                    dp_pg = mesh[dp_pg_src].get_group()

                replicate(
                    ref_model,
                    device_ids=[self.rank] if device_type.type != "cpu" else None,
                    process_group=dp_pg,
                )

                self._run_train_parity(
                    model, ref_model, dp_pg, mesh=mesh, mlp_dim=mlp_dim
                )
            dist.barrier()

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
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        for param in model.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.device_mesh, mesh)
            self.assertEqual(len(param.placements), 1)
            self.assertIsInstance(param.placements[0], Shard)

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
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
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
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_invalid_dim_names(self):
        """Error when dp_mesh_dims references nonexistent mesh dim names."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(ValueError, "not found in mesh.mesh_dim_names"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="nonexistent"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_mesh_mismatch(self):
        """Error when param DTensor mesh differs from the mesh passed to fully_shard."""
        mesh1 = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        mesh2 = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = nn.Linear(16, 32, device=device_type)
        # Distribute params on mesh1 but pass mesh2 to fully_shard
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.data, mesh1, [Replicate()]),
            requires_grad=True,
        )
        model.bias = nn.Parameter(
            distribute_tensor(model.bias.data, mesh1, [Replicate()]),
            requires_grad=True,
        )
        with self.assertRaisesRegex(ValueError, "same mesh"):
            fully_shard(
                model,
                mesh=mesh2,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    def test_validation_at_least_one_required(self):
        """Error when neither shard nor replicate is set."""
        with self.assertRaisesRegex(ValueError, "At least one of shard or replicate"):
            DataParallelMeshDims()

    @skip_if_lt_x_gpu(2)
    def test_validation_spmd_mesh_non_dtensor_params(self):
        """Error when dp_mesh_dims is provided but params are not DTensors."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        # Do NOT call distribute_module -- params are plain tensors
        with self.assertRaisesRegex(ValueError, "must be DTensors"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(2)
    def test_validation_reshard_after_forward_int_spmd(self):
        """Error when reshard_after_forward is int with SPMD mesh."""
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        model = MLP(16, device=device_type)
        distribute_module(model, mesh)
        with self.assertRaisesRegex(
            NotImplementedError, "reshard_after_forward as int"
        ):
            fully_shard(
                model,
                mesh=mesh,
                reshard_after_forward=2,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )


if __name__ == "__main__":
    run_tests()
