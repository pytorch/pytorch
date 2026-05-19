# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import replicate
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard


if dist._is_spmd_types_available():
    import spmd_types as spmd
    import spmd_types._checker
    import spmd_types._type_attr

from torch.distributed.pipelining.schedules import ScheduleGPipe
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import DTensor, init_device_mesh, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class TestMLP(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 4, bias=False, device=device)
        self.out_proj = nn.Linear(dim * 4, dim, bias=False, device=device)

    def forward(self, x):
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        return z


def _tp_init(model, tp_pg):
    """Shard weights for column/row parallel TP and patch forward."""
    from spmd_types import all_reduce, convert

    tp_rank = tp_pg.rank()
    tp_size = tp_pg.size()

    with torch.no_grad():
        model.in_proj.weight = nn.Parameter(
            model.in_proj.weight.chunk(tp_size, dim=0)[tp_rank].contiguous()
        )
        model.out_proj.weight = nn.Parameter(
            model.out_proj.weight.chunk(tp_size, dim=1)[tp_rank].contiguous()
        )

    def _tp_forward(self, x):
        x = convert(x, tp_pg, src=spmd.I, dst=spmd.R)
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = all_reduce(z, tp_pg, dst=spmd.I)
        z = F.relu(z)
        return z

    import types

    model.forward = types.MethodType(_tp_forward, model)


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdTypes(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    def setUp(self):
        super().setUp()
        from spmd_types._mesh_axis import _reset

        _reset()

    def _check_dtensor_placements(self, model, expected_placements_by_fqn):
        """Check that sharded params are DTensors with expected placements."""
        for fqn, param in model.named_parameters():
            self.assertIsInstance(
                param, DTensor, f"{fqn} should be DTensor after fully_shard"
            )
            if fqn in expected_placements_by_fqn:
                self.assertEqual(
                    param.placements,
                    expected_placements_by_fqn[fqn],
                    f"{fqn} placement mismatch",
                )

    def _register_param_check_hooks(self, model, expected_types):
        """Register pre-hooks on Linear modules to check spmd_types at compute time.

        expected_types maps param FQN (e.g. "in_proj.weight") to the expected
        local type dict and optional PartitionSpec. Each value is a tuple of
        (local_type_dict, partition_spec_or_None).
        """
        from spmd_types.runtime import get_partition_spec

        def make_hook(module_fqn):
            def check_params_at_compute(module, args):
                for name, param in module.named_parameters(recurse=False):
                    fqn = f"{module_fqn}.{name}" if module_fqn else name
                    self.assertNotIsInstance(
                        param.data,
                        DTensor,
                        f"{fqn} should be plain tensor at compute time",
                    )
                    self.assertTrue(
                        spmd._checker.has_local_type(param),
                        f"{fqn} should have spmd_types annotation at compute time",
                    )
                    if fqn in expected_types:
                        expected_lt, expected_ps = expected_types[fqn]
                        self.assertEqual(
                            dict(spmd.get_local_type(param)),
                            expected_lt,
                            f"{fqn} local type mismatch",
                        )
                        actual_ps = get_partition_spec(param)
                        self.assertEqual(
                            actual_ps,
                            expected_ps,
                            f"{fqn} partition spec mismatch",
                        )

            return check_params_at_compute

        for fqn, m in model.named_modules():
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(make_hook(fqn))

    def _run_fwd_bwd(self, model, ref_model, inp, fsdp_axis, input_type):
        """Run forward + backward and check output numerics, output type, and grad annotations."""
        ref_out = ref_model(inp)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        with spmd._checker.typecheck(strict_mode="strict", local=False):
            spmd.assert_type(inp, input_type)
            out = model(inp)
        self.assertEqual(spmd.get_local_type(out)[fsdp_axis], spmd.V)
        loss = out.sum()
        loss.backward()

        self.assertEqual(ref_loss, loss)

        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), model.named_parameters(), strict=True
        ):
            self.assertIsNotNone(p2.grad, f"{n2} grad should not be None")
            self.assertIsInstance(p2.grad, DTensor, f"{n2} grad should be DTensor")
            self.assertEqual(
                p1.grad, p2.grad.full_tensor(), msg=f"Grad mismatch: {n1} vs {n2}"
            )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_1d(self):
        """FSDP alone: params initialized as Replicated on the FSDP mesh."""
        mlp_dim = 16
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        fsdp_axis = spmd.MeshAxis.of(mesh.get_group("fsdp"))

        torch.manual_seed(42)
        model = TestMLP(mlp_dim, device=device_type)
        ref_model = TestMLP(mlp_dim, device=device_type)
        ref_model.load_state_dict(model.state_dict())

        for param in model.parameters():
            spmd._type_attr.set_local_type(param, {fsdp_axis: spmd.R})

        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )
        self._check_dtensor_placements(
            model,
            {
                "in_proj.weight": (Shard(0),),
                "out_proj.weight": (Shard(0),),
            },
        )
        replicate(ref_model, device_ids=[self.rank])
        expected_types = {
            "in_proj.weight": ({fsdp_axis: spmd.R}, None),
            "out_proj.weight": ({fsdp_axis: spmd.R}, None),
        }
        self._register_param_check_hooks(model, expected_types)

        input_type = {fsdp_axis: spmd.S(0)}
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device=device_type)
        with spmd.set_current_mesh(mesh):
            self._run_fwd_bwd(model, ref_model, inp, fsdp_axis, input_type)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_joint_mesh(self):
        """FSDP + TP: params initialized on the joint mesh with Invariant on FSDP, sharded on TP."""
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        fsdp_axis = spmd.MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        tp_pg = mesh.get_group("tp")
        dp_pg = mesh.get_group("fsdp")

        tp_plan = {
            "in_proj.weight": spmd.S(0),
            "out_proj.weight": spmd.S(1),
        }

        torch.manual_seed(42)
        ref_model = TestMLP(16, device=device_type)
        model = TestMLP(16, device=device_type)
        model.load_state_dict(ref_model.state_dict())
        _tp_init(model, tp_pg)

        for fqn, param in model.named_parameters():
            spmd._type_attr.set_local_type(
                param, {fsdp_axis: spmd.R, tp_axis: tp_plan[fqn]}
            )

        def shard_fn(param):
            lt = spmd.get_local_type(param)
            tp_type = lt.get(tp_axis)
            if isinstance(tp_type, spmd.S) and tp_type.dim == 0:
                return Shard(1)
            return Shard(0)

        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=shard_fn,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
        )

        self._check_dtensor_placements(
            model,
            {
                "in_proj.weight": (Shard(1), Shard(0)),
                "out_proj.weight": (Shard(0), Shard(1)),
            },
        )

        from spmd_types.types import PartitionSpec

        replicate(ref_model, process_group=dp_pg)
        expected_types = {
            "in_proj.weight": (
                {fsdp_axis: spmd.R, tp_axis: spmd.V},
                PartitionSpec(tp_axis, None),
            ),
            "out_proj.weight": (
                {fsdp_axis: spmd.R, tp_axis: spmd.V},
                PartitionSpec(None, tp_axis),
            ),
        }
        self._register_param_check_hooks(model, expected_types)

        input_type = {fsdp_axis: spmd.S(0), tp_axis: spmd.I}
        torch.manual_seed(42 + dp_pg.rank() + 1)
        inp = torch.randn((2, 16), device=device_type)
        with spmd.set_current_mesh(mesh):
            self._run_fwd_bwd(model, ref_model, inp, fsdp_axis, input_type)


class TwoLayerMLP(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.layer0 = TestMLP(dim, device=device)
        self.layer1 = TestMLP(dim, device=device)

    def forward(self, x):
        return self.layer1(self.layer0(x))


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestPipelineSpmdTypes(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    def setUp(self):
        super().setUp()
        from spmd_types._mesh_axis import _reset

        _reset()

    @skip_if_lt_x_gpu(4)
    def test_pp_tp(self):
        """PP + TP: verify TP annotations survive through pipeline stages."""
        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh = init_device_mesh(
            device_type.type,
            (pp_size, tp_size),
            mesh_dim_names=("pp", "tp"),
        )
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        tp_pg = mesh.get_group("tp")
        pp_group = mesh["pp"].get_group()
        pp_rank = dist.get_rank(pp_group)

        mlp_dim = 16
        torch.manual_seed(42)
        model = TwoLayerMLP(mlp_dim, device=device_type)
        stage_module = model.layer0 if pp_rank == 0 else model.layer1
        _tp_init(stage_module, tp_pg)

        tp_plan = {
            "in_proj.weight": spmd.S(0),
            "out_proj.weight": spmd.S(1),
        }
        for fqn, param in stage_module.named_parameters():
            spmd._type_attr.set_local_type(param, {tp_axis: tp_plan[fqn]})

        # Track whether TP annotations are alive during compute
        annotations_seen = []

        def check_tp_hook(module, args):
            for name, param in module.named_parameters(recurse=False):
                self.assertNotIsInstance(param.data, DTensor)
                self.assertTrue(
                    spmd._checker.has_local_type(param),
                    f"{name} should have spmd_types annotation in compute",
                )
                lt = spmd.get_local_type(param)
                self.assertIn(tp_axis, lt, f"{name} missing TP axis annotation")
                annotations_seen.append(name)

        for m in stage_module.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(check_tp_hook)

        stage = PipelineStage(
            submodule=stage_module,
            stage_index=pp_rank,
            num_stages=pp_size,
            device=device_type,
            group=pp_group,
        )

        chunks = 2
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=chunks,
            loss_fn=lambda out, tgt: out.sum(),
        )

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, mlp_dim), device=device_type)
        target = torch.randn((4, mlp_dim), device=device_type)

        with spmd.set_current_mesh(mesh):
            if pp_rank == 0:
                schedule.step(inp)
            else:
                schedule.step(target=target)

        self.assertGreater(
            len(annotations_seen), 0,
            "TP annotations should have been checked during compute",
        )


if __name__ == "__main__":
    run_tests()
