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
from torch.distributed.tensor import init_device_mesh, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck


class SpmdLinear(nn.Module):
    def __init__(self, mesh, seq_parallel: bool):
        super().__init__()
        self.mesh = mesh
        self.tp_pg = mesh.get_group("tp")
        self.seq_parallel = seq_parallel
        self.unsharded_weight = nn.Parameter(torch.randn(16, 16))
        self.sharded_weight = nn.Parameter(torch.randn(16, 16))
        self.compute_param_types = None

    def forward(self, x):
        """Simulate the TP collectives around a sharded projection.

        The global computation is x = x @ A; x = x @ B; return x.sum().
        With sequence parallelism, the activation is all-gathered before the
        sharded projection. Without it, the output is all-gathered before loss.
        """
        self.compute_param_types = (
            dict(spmd.get_local_type(self.unsharded_weight)),
            dict(spmd.get_local_type(self.sharded_weight)),
        )
        x = x @ self.unsharded_weight
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(1) if self.seq_parallel else spmd.I,
            dst=spmd.R,
            backward_options={"op_dtype": torch.float32},
        )
        x = x @ self.sharded_weight.t()
        x = spmd.redistribute(
            x,
            self.tp_pg,
            src=spmd.S(2),
            dst=spmd.I,
            backward_options={"op_dtype": torch.float32},
        )
        return x.sum()


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdTypes(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=0, world_size=16
        )
        cls.type_mesh = init_device_mesh(
            "cpu", (4, 2, 2), mesh_dim_names=("dp", "cp", "tp")
        )
        cls.fsdp_mesh = init_device_mesh(
            "cpu", (2, 2, 2, 2), mesh_dim_names=("dpr", "dps", "cp", "tp")
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def test_restores_param_spmd_type_for_compute(self):
        """FSDP restores user SPMD metadata on params for compute.

        FSDP should preserve the compute-mesh annotations when applied with
        a different storage-mesh view. fully_shard() itself does not need a
        set_current_mesh() scope when all parameter axes are already annotated.
        """
        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        cp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("cp"))
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))
        for seq_parallel in (False, True):
            with self.subTest(seq_parallel=seq_parallel):
                # init model, annotate params
                # # TODO(pianpwk): support optionally-annotated for DP/CP axes.
                model = SpmdLinear(self.type_mesh, seq_parallel)
                spmd.assert_type(
                    model.unsharded_weight,
                    {
                        dp_axis: spmd.R,
                        cp_axis: spmd.R,
                        tp_axis: spmd.R if seq_parallel else spmd.I,
                    },
                )
                spmd.assert_type(
                    model.sharded_weight,
                    {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.S(0)},
                )

                # FSDP turns params into DTensor, in this case should contain StridedShard @ dps
                fully_shard(
                    model,
                    mesh=self.fsdp_mesh,
                    dp_mesh_dims=DataParallelMeshDims(
                        shard=("dps", "cp"),
                        replicate="dpr",
                    ),
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        cast_forward_inputs=True,
                    ),
                )
                self.assertEqual(
                    model.sharded_weight._spec.placements,
                    (
                        Replicate(),
                        _StridedShard(0, split_factor=self.type_mesh["tp"].size()),
                        Shard(0),
                    ),
                )

                # annotate model inputs as V + PartitionSpec
                inp = torch.randn(4, 8, 16)
                input_type = {
                    dp_axis: spmd.V,
                    cp_axis: spmd.V,
                    tp_axis: spmd.V if seq_parallel else spmd.I,
                }
                input_partition_spec = spmd.PartitionSpec(
                    dp_axis,
                    (cp_axis, tp_axis) if seq_parallel else cp_axis,
                    None,
                )

                # check loss output type, check compute-time param annotations are restored.
                with (
                    spmd.set_current_mesh(self.type_mesh),
                    typecheck(strict_mode="strict", local=False),
                ):
                    spmd.assert_type(
                        inp,
                        input_type,
                        partition_spec=input_partition_spec,
                    )
                    loss = model(inp)
                    spmd.assert_type(
                        loss,
                        {dp_axis: spmd.P, cp_axis: spmd.P, tp_axis: spmd.I},
                    )
                self.assertEqual(
                    model.compute_param_types,
                    (
                        {
                            dp_axis: spmd.R,
                            cp_axis: spmd.R,
                            tp_axis: spmd.R if seq_parallel else spmd.I,
                        },
                        {dp_axis: spmd.R, cp_axis: spmd.R, tp_axis: spmd.V},
                    ),
                )

                with CommDebugMode() as comm_mode:
                    loss.backward()
                comm_counts = comm_mode.get_comm_counts()
                if seq_parallel:
                    self.assertEqual(
                        comm_counts[torch.ops.c10d._reduce_scatter_base_], 2
                    )
                else:
                    self.assertEqual(
                        comm_counts[torch.ops.c10d._reduce_scatter_base_], 1
                    )
                self.assertEqual(
                    comm_counts[torch.ops.c10d.allreduce_]
                    + comm_counts[torch.ops.c10d_functional.all_reduce],
                    2,
                )

    def test_local_v_param_requires_partition_spec(self):
        """Local-only V@TP params are ambiguous for FSDP.

        Without PartitionSpec shard info, FSDP cannot choose the matching
        DTensor Shard(dim), so it rejects the parameter at init time.
        """
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
            spmd.assert_type(
                model.sharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.V},
            )
        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            """Parameter 'sharded_weight' has V type on mesh dim 'tp' but no PartitionSpec shard info. Use assert_type with S(dim) or pass a PartitionSpec.""",
        )

    def test_partial_param_annotations_not_supported(self):
        """TODO(pianpwk) support this by inferring from current_mesh."""
        model = SpmdLinear(self.type_mesh, seq_parallel=False)
        dp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(self.type_mesh.get_group("tp"))

        spmd.assert_type(
            model.unsharded_weight,
            {dp_axis: spmd.R, tp_axis: spmd.I},
        )
        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'unsharded_weight' has spmd_types annotations that are "
            "not compatible with the full SPMD mesh passed to fully_shard. "
            "FSDP requires fully annotated parameters spanning the same mesh "
            "as the one passed to fully_shard. Got local_type={mesh_dp: R, "
            "mesh_tp: I}, partition_spec=None, and spmd_mesh=DeviceMesh((dpr=2, "
            "dps=2, cp=2, tp=2), 'cpu', stride=(8, 4, 2, 1)).",
        )

    def test_spmd_params_require_dp_mesh_dims(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
        with self.assertRaises(ValueError) as cm:
            fully_shard(model, mesh=self.type_mesh["dp"])
        self.assertExpectedInline(
            str(cm.exception),
            "spmd_types parameters require a named SPMD mesh "
            "(pass dp_mesh_dims to fully_shard)",
        )

    def test_fsdp_dp_axes_must_be_r(self):
        model = SpmdLinear(self.type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.I, "cp": spmd.R, "tp": spmd.I},
            )
        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.fsdp_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dps", "cp"),
                    replicate="dpr",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Expected spmd.R on FSDP DP mesh dim 'dpr' for parameter "
            "'unsharded_weight' but got PerMeshAxisLocalSpmdType.I. FSDP "
            "requires DP parameters to be R since it handles the DP gradient "
            "reduction.",
        )


if __name__ == "__main__":
    run_tests()
