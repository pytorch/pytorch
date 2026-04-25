# Owner(s): ["oncall: distributed"]

import copy
import unittest
from contextlib import ExitStack

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
from torch.distributed.spmd_types import (
    assert_type as spmd_assert_type,
    get_local_type,
    get_partition_spec,
    has_local_type,
    I,
    is_available as spmd_types_available,
    MeshAxis,
    P,
    R,
    S,
    set_current_mesh,
    set_local_type,
    typecheck,
    V,
)
from spmd_types.types import PartitionSpec  # pyrefly: ignore
from torch.distributed.tensor import (
    DTensor,
    init_device_mesh,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


def _annotate_params_with_spmd_types(model, mesh):
    """Annotate all model parameters with spmd_types I (Invariant) on all dims."""
    for param in model.parameters():
        local_type = {}
        for name in mesh.mesh_dim_names:
            axis = MeshAxis.of(mesh.get_group(name))
            local_type[axis] = I
        set_local_type(param, local_type)


@unittest.skipUnless(spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdTypes(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    def setUp(self):
        super().setUp()
        from torch.distributed.spmd_types import _reset

        _reset()

    def _run_train_parity(self, model, ref_model, dp_pg, mesh, num_iters=5, mlp_dim=16):
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))

        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)

        def check_spmd_types_hook(module, args):
            for name, param in module.named_parameters(recurse=False):
                self.assertNotIsInstance(
                    param.data,
                    DTensor,
                    f"{name} should be a plain tensor during forward, not DTensor",
                )
                self.assertTrue(
                    has_local_type(param),
                    f"Missing spmd_types on {name} during forward",
                )

        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(check_spmd_types_hook)

        tc_stack = ExitStack()

        def enter_typecheck(module, args):
            tc_stack.enter_context(typecheck(strict_mode="strict"))
            spmd_assert_type(args[0], {fsdp_axis: S(0)})

        def exit_typecheck(module, args, output):
            tc_stack.close()

        model.register_forward_pre_hook(enter_typecheck, prepend=False)
        model.register_forward_hook(exit_typecheck)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        with set_current_mesh(mesh):
            for i in range(num_iters):
                inp = torch.randn((2, mlp_dim), device=device_type)
                ref_optim.zero_grad()
                ref_loss = ref_model(inp).sum()
                ref_loss.backward()
                ref_optim.step()

                optim.zero_grad()
                output = model(inp)
                local_type = get_local_type(output)
                self.assertIn(fsdp_axis, local_type)
                self.assertEqual(local_type[fsdp_axis], V)
                spmd_assert_type(output, {fsdp_axis: S(0)})
                self.assertEqual(
                    get_partition_spec(output), PartitionSpec(fsdp_axis, None)
                )

                loss = output.sum()
                loss.backward()
                optim.step()

                self.assertEqual(ref_loss, loss)

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters(), strict=True
        ):
            if isinstance(p2, DTensor):
                p2_full = p2.full_tensor()
            elif has_local_type(p2):
                gathered = [torch.empty_like(p2) for _ in range(dp_pg.size())]
                dist.all_gather(gathered, p2, group=dp_pg)
                p2_full = torch.cat(gathered, dim=0)
            else:
                p2_full = p2
            self.assertEqual(p1, p2_full, msg=f"Param mismatch: {n1} vs {n2}")

    @skip_if_lt_x_gpu(2)
    def test_fsdp_1d_spmd_types_train_parity(self):
        """Train parity and sharded param correctness with 1D FSDP."""
        mlp_dim = 16
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))

        torch.manual_seed(42)
        model = MLP(mlp_dim, device=device_type)
        ref_model = copy.deepcopy(model)

        _annotate_params_with_spmd_types(model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        for param in model.parameters():
            self.assertNotIsInstance(param, DTensor)
            local_type = get_local_type(param)
            self.assertIn(fsdp_axis, local_type)
            self.assertIs(local_type[fsdp_axis], V)
            spec = get_partition_spec(param)
            self.assertIsNotNone(spec)
            self.assertIn(fsdp_axis, spec.axes_with_partition_spec())

        replicate(
            ref_model,
            device_ids=[self.rank] if device_type.type != "cpu" else None,
        )
        self._run_train_parity(
            model, ref_model, dist.group.WORLD, mesh=mesh, mlp_dim=mlp_dim
        )

    def _make_fsdp_tp_model(self):
        """Create an FSDP+TP model with spmd_types annotations."""
        from spmd_types import all_reduce, convert

        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = MeshAxis.of(mesh.get_group("tp"))
        tp_pg = mesh.get_group("tp")

        tp_plan = {
            "in_proj.weight": S(0),
            "out_proj.weight": S(1),
        }

        class SpmdTypesTPMLP(nn.Module):
            def __init__(self, dim, tp_size, device=None):
                super().__init__()
                self.in_proj = nn.Linear(
                    dim, dim * 4 // tp_size, bias=False, device=device
                )
                self.out_proj = nn.Linear(
                    dim * 4 // tp_size, dim, bias=False, device=device
                )

            def forward(self, x):
                x = convert(x, tp_pg, src=I, dst=R)
                z = self.in_proj(x)
                z = torch.nn.functional.relu(z)
                z = self.out_proj(z)
                z = all_reduce(z, tp_pg, dst=I)
                z = torch.nn.functional.relu(z)
                return z

        torch.manual_seed(42)
        model = SpmdTypesTPMLP(16, tp_size, device=device_type)
        for fqn, param in model.named_parameters():
            set_local_type(param, {fsdp_axis: I, tp_axis: tp_plan[fqn]})

        def shard_fn(param):
            lt = get_local_type(param)
            tp_type = lt.get(tp_axis)
            if isinstance(tp_type, S) and tp_type.dim == 0:
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )
        return model, mesh, tp_plan

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_sharded_param_types(self):
        """Verify sharded param types and partition specs for FSDP+TP."""
        model, mesh, tp_plan = self._make_fsdp_tp_model()
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = MeshAxis.of(mesh.get_group("tp"))

        expected_specs = {
            "in_proj.weight": PartitionSpec(tp_axis, fsdp_axis),
            "out_proj.weight": PartitionSpec(fsdp_axis, tp_axis),
        }
        for name, param in model.named_parameters():
            self.assertNotIsInstance(param, DTensor)
            self.assertTrue(has_local_type(param))
            local_type = get_local_type(param)
            self.assertIs(local_type[fsdp_axis], V)
            self.assertIn(tp_axis, local_type)
            self.assertEqual(get_partition_spec(param), expected_specs[name])

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_optimizer_state_spmd_types(self):
        """Optimizer state should be plain tensors with spmd_types after step."""
        model, mesh, tp_plan = self._make_fsdp_tp_model()
        fsdp_axis = MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = MeshAxis.of(mesh.get_group("tp"))
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 16), device=device_type)
        spmd_assert_type(inp, {fsdp_axis: S(0), tp_axis: I})
        with set_current_mesh(mesh), typecheck(strict_mode="permissive", local=False):
            output = model(inp)
            output_type = get_local_type(output)
            self.assertIs(output_type[fsdp_axis], V)
            self.assertIs(output_type[tp_axis], I)
            self.assertEqual(
                get_partition_spec(output), PartitionSpec(fsdp_axis, None)
            )
            output.sum().backward()
        with set_current_mesh(mesh), typecheck(strict_mode="permissive"):
            optim.step()

        for name, param in model.named_parameters():
            state = optim.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertIn(key, state)
                self.assertNotIsInstance(state[key], DTensor)
                self.assertEqual(state[key].shape, param.shape)


if __name__ == "__main__":
    run_tests()
