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
    from spmd_types.checker import typecheck

from torch.distributed.tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests


c10d_functional = torch.ops.c10d_functional
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


class TestScale(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, device=device))

    def forward(self, x):
        return x * self.weight


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
                    self.assertNotEqual(
                        spmd.get_local_type(param),
                        {},
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
            if any(m.named_parameters(recurse=False)):
                m.register_forward_pre_hook(make_hook(fqn))

    def _run_fwd_bwd(self, model, ref_model, inp, fsdp_axis, input_type):
        """Run forward + backward and check output numerics, output type, and grad annotations."""
        ref_out = ref_model(inp)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        with typecheck(strict_mode="strict", local=False):
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
        """FSDP-only params start as R@FSDP plain tensors.

        fully_shard should store them as Shard(0) DTensors, restore R@FSDP for
        compute, and reduce-scatter grads back to sharded DTensor grads.
        """
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
            spmd.assert_type(param, {fsdp_axis: spmd.R})

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
        """FSDP+TP params start as R@FSDP and S(dim)@TP.

        fully_shard should add FSDP sharding while preserving TP sharding, then
        restore R@FSDP/V@TP plus PartitionSpec for compute.
        """
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
            spmd.assert_type(param, {fsdp_axis: spmd.R, tp_axis: tp_plan[fqn]})

        def shard_fn(param):
            from spmd_types.runtime import get_partition_spec
            from spmd_types.types import partition_spec_get_shard

            tp_shard = partition_spec_get_shard(get_partition_spec(param), tp_axis)
            if tp_shard is not None and tp_shard.dim == 0:
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

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_local_v_param_requires_partition_spec(self):
        """Local-only V@TP params are ambiguous for FSDP.

        Without PartitionSpec shard info, FSDP cannot choose a DTensor
        Shard(dim), so fully_shard should reject the parameter at init time.
        """
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh = init_device_mesh(
            device_type.type,
            (dp_size, tp_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        fsdp_axis = spmd.MeshAxis.of(mesh.get_group("fsdp"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))

        model = TestScale(16, device=device_type)
        spmd.assert_type(model.weight, {fsdp_axis: spmd.R, tp_axis: spmd.V})

        with self.assertRaisesRegex(ValueError, "no PartitionSpec shard info"):
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_unsharded_param(self):
        """TP-replicated params handle I@TP and R@TP grads differently.

        I@TP keeps grads local; R@TP infers Partial@TP grads and all-reduces
        them back to Replicate@TP storage before accumulation.
        """
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

        for tp_param_type, tp_input_type in (
            (spmd.I, spmd.I),
            (spmd.R, spmd.S(0)),
        ):
            # Initialize a TP-replicated parameter and annotate its SPMD type.
            torch.manual_seed(42)
            ref_model = TestScale(16, device=device_type)
            model = TestScale(16, device=device_type)
            model.load_state_dict(ref_model.state_dict())
            spmd.assert_type(
                model.weight,
                {fsdp_axis: spmd.R, tp_axis: tp_param_type},
            )

            # fully_shard should add FSDP sharding while preserving TP replication.
            fully_shard(
                model,
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            )
            self._check_dtensor_placements(model, {"weight": (Shard(0), Replicate())})
            self._register_param_check_hooks(
                model,
                {
                    "weight": (
                        {fsdp_axis: spmd.R, tp_axis: tp_param_type},
                        None,
                    )
                },
            )

            # Build the reference grad; only R@TP needs an explicit TP all-reduce.
            replicate(ref_model, process_group=dp_pg)
            if tp_input_type is spmd.I:
                torch.manual_seed(1000 + dp_pg.rank())
            else:
                torch.manual_seed(1000 + dp_pg.rank() * tp_size + tp_pg.rank())
            inp = torch.randn((2, 16), device=device_type)

            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            if tp_param_type is spmd.R:
                dist.all_reduce(ref_model.weight.grad, group=tp_pg)

            # Run typed FSDP forward/backward with matching input annotations.
            if tp_input_type is spmd.I:
                input_type = {fsdp_axis: spmd.S(0), tp_axis: spmd.I}
                input_partition_spec = None
            else:
                input_type = {fsdp_axis: spmd.V, tp_axis: spmd.V}
                input_partition_spec = spmd.PartitionSpec((fsdp_axis, tp_axis), None)
            with (
                spmd.set_current_mesh(mesh),
                typecheck(strict_mode="strict", local=False),
            ):
                spmd.assert_type(inp, input_type, partition_spec=input_partition_spec)
                loss = model(inp).sum()

            # count collectives: should only AR if R@TP start.
            with CommDebugMode() as comm_mode:
                loss.backward()
            expected_all_reduce_count = 1 if tp_param_type is spmd.R else 0
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_reduce],
                expected_all_reduce_count,
            )

            # The stored sharded DTensor grad should match the full reference grad.
            param = dict(model.named_parameters())["weight"]
            self.assertIsInstance(param.grad, DTensor)
            self.assertEqual(ref_model.weight.grad, param.grad.full_tensor())


if __name__ == "__main__":
    run_tests()
