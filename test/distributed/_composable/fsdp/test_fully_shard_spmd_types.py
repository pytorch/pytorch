# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck

c10d_ops = torch.ops.c10d
c10d_functional = torch.ops.c10d_functional


class SpmdLinear(nn.Module):
    def __init__(self, mesh, seq_parallel: bool):
        super().__init__()
        self.mesh = mesh
        self.tp_pg = mesh.get_group("tp")
        self.seq_parallel = seq_parallel
        self.unsharded_weight = nn.Parameter(torch.randn(16, 16))
        self.sharded_weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        """Simulate the TP collectives around a sharded projection.

        The global computation is x = x @ A; x = x @ B; return x.sum().
        With sequence parallelism, the activation is all-gathered before the
        sharded projection. Without it, the output is all-gathered before loss.
        """
        x = x @ self.unsharded_weight
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(1) if self.seq_parallel else spmd.I,
            dst=spmd.R,
            backward_options={"op_dtype": torch.float32},
        )
        x = x @ self.sharded_weight
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(2),
            dst=spmd.I,
            backward_options={"op_dtype": torch.float32},
        )
        return x.sum()


class HsdpLinear(nn.Module):
    def __init__(self, dp_axis):
        super().__init__()
        self.dp_axis = dp_axis
        self.weight = nn.Parameter(torch.randn(16, 16))
        self.compute_local_type = None

    def forward(self, x):
        self.compute_local_type = dict(spmd.get_local_type(self.weight))
        return (x @ self.weight).sum()


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdTypes(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
        cls.mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        cls.hsdp_mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def test_restores_param_spmd_type_for_compute(self):
        """FSDP restores user SPMD metadata on params for compute.

        The user only annotates TP sharding; FSDP should restore the TP
        metadata and add DP:R before the module sees its parameter or
        MP-cast input.
        """
        with spmd.set_current_mesh(self.mesh):
            for seq_parallel in (False, True):
                with self.subTest(seq_parallel=seq_parallel):
                    model = SpmdLinear(self.mesh, seq_parallel)
                    spmd.assert_type(
                        model.unsharded_weight,
                        {"tp": spmd.R if seq_parallel else spmd.I},
                    )
                    spmd.assert_type(model.sharded_weight, {"tp": spmd.S(1)})
                    fully_shard(
                        model,
                        mesh=self.mesh,
                        dp_mesh_dims=DataParallelMeshDims(shard="dp"),
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=torch.bfloat16,
                            reduce_dtype=torch.float32,
                            cast_forward_inputs=True,
                        ),
                    )
                    inp = torch.randn(4, 8, 16)
                    with typecheck(strict_mode="strict", local=False):
                        spmd.assert_type(
                            inp,
                            {
                                "dp": spmd.S(0),
                                "tp": spmd.S(1) if seq_parallel else spmd.I,
                            },
                        )
                        loss = model(inp)
                        spmd.assert_type(
                            loss,
                            {"dp": spmd.P, "tp": spmd.I},
                        )

                    # run BWD to count collectives
                    with CommDebugMode() as comm_mode:
                        loss.backward()
                    comm_counts = comm_mode.get_comm_counts()

                    if seq_parallel:
                        # DP reduce-scatter for both weight grads;
                        # TP reduce-scatter for intermediate grad;
                        # TP all-reduce for unsharded weight grad.
                        self.assertEqual(comm_counts[c10d_ops._reduce_scatter_base_], 2)
                    else:
                        # DP reduce-scatter;
                        # TP all-reduce for intermediate grad.
                        self.assertEqual(comm_counts[c10d_ops._reduce_scatter_base_], 1)
                    self.assertEqual(
                        comm_counts[c10d_ops.allreduce_]
                        + comm_counts[c10d_functional.all_reduce],
                        1,
                    )

    def test_local_v_param_requires_partition_spec(self):
        """Local-only V@TP params are ambiguous for FSDP.

        Without PartitionSpec shard info, FSDP cannot choose the matching
        DTensor Shard(dim), so it rejects the parameter at init time.
        """
        model = SpmdLinear(self.mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.mesh):
            spmd.assert_type(model.unsharded_weight, {"tp": spmd.I})
            spmd.assert_type(model.sharded_weight, {"tp": spmd.V})
            with self.assertRaises(ValueError) as cm:
                fully_shard(
                    model,
                    mesh=self.mesh,
                    dp_mesh_dims=DataParallelMeshDims(shard="dp"),
                )
        self.assertExpectedInline(
            str(cm.exception),
            """Parameter 'sharded_weight' has V type on mesh dim 'tp' but no PartitionSpec shard info. Use assert_type with S(dim) or pass a PartitionSpec.""",
        )

    def test_hsdp_uses_logical_dp_axis_for_spmd_type(self):
        dp_axis = spmd.MeshAxis.of(
            self.hsdp_mesh[("dp_replicate", "dp_shard")]._flatten("dp").get_group()
        )
        model = HsdpLinear(dp_axis)

        with spmd.set_current_mesh({"dp": dp_axis}):
            spmd.assert_type(model.weight, {"dp": spmd.R})

        fully_shard(
            model,
            mesh=self.hsdp_mesh,
            dp_mesh_dims=DataParallelMeshDims(
                shard="dp_shard",
                replicate="dp_replicate",
            ),
        )
        model(torch.randn(4, 16))
        self.assertEqual(model.compute_local_type, {dp_axis: spmd.R})


if __name__ == "__main__":
    run_tests()
