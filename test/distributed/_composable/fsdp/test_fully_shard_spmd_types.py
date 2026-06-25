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
from torch.distributed.fsdp._fully_shard._fsdp_common import ShardPlacementResult
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
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


class DenseSparseParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_weight = nn.Parameter(torch.randn(16, 16))
        self.sparse_weight = nn.Parameter(torch.randn(16, 16))
        self.compute_param_types = None

    def forward(self):
        self.compute_param_types = (
            dict(spmd.get_local_type(self.dense_weight)),
            dict(spmd.get_local_type(self.sparse_weight)),
        )
        return self.dense_weight.sum(), self.sparse_weight.sum()


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
        cls.dense_type_mesh = init_device_mesh(
            "cpu", (4, 2, 2), mesh_dim_names=("dp", "cp", "tp")
        )
        cls.dense_storage_mesh = init_device_mesh(
            "cpu",
            (2, 2, 2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
        )
        cls.sparse_mesh = init_device_mesh(
            "cpu", (2, 2, 4), mesh_dim_names=("dp_replicate", "efsdp", "ep")
        )
        cls.dp_axis = spmd.MeshAxis.of(cls.dense_type_mesh.get_group("dp"))
        cls.cp_axis = spmd.MeshAxis.of(cls.dense_type_mesh.get_group("cp"))
        cls.tp_axis = spmd.MeshAxis.of(cls.dense_type_mesh.get_group("tp"))
        cls.dp_replicate_axis = spmd.MeshAxis.of(
            cls.dense_storage_mesh.get_group("dp_replicate")
        )
        cls.efsdp_axis = spmd.MeshAxis.of(cls.sparse_mesh.get_group("efsdp"))
        cls.ep_axis = spmd.MeshAxis.of(cls.sparse_mesh.get_group("ep"))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def test_restores_param_spmd_type_for_compute(self):
        """FSDP restores user SPMD metadata on params for compute.

        FSDP should preserve the compute-mesh annotations when applied with
        a different storage-mesh view.
        """
        for seq_parallel in (False, True):
            with self.subTest(seq_parallel=seq_parallel):
                model = SpmdLinear(self.dense_type_mesh, seq_parallel)
                spmd.assert_type(
                    model.unsharded_weight,
                    {
                        self.dp_axis: spmd.R,
                        self.cp_axis: spmd.R,
                        self.tp_axis: spmd.R if seq_parallel else spmd.I,
                    },
                )
                spmd.assert_type(
                    model.sharded_weight,
                    {
                        self.dp_axis: spmd.R,
                        self.cp_axis: spmd.R,
                        self.tp_axis: spmd.S(0),
                    },
                )

                # FSDP turns params into DTensor, in this case should contain StridedShard @ dp_shard
                with spmd.set_current_mesh(self.dense_type_mesh):
                    fully_shard(
                        model,
                        mesh=self.dense_storage_mesh,
                        dp_mesh_dims=DataParallelMeshDims(
                            shard=("dp_shard", "cp"),
                            replicate="dp_replicate",
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
                        _StridedShard(
                            0, split_factor=self.dense_type_mesh["tp"].size()
                        ),
                        Shard(0),
                    ),
                )

                # annotate model inputs as V + PartitionSpec
                inp = torch.randn(4, 8, 16)
                input_type = {
                    self.dp_axis: spmd.V,
                    self.cp_axis: spmd.V,
                    self.tp_axis: spmd.V if seq_parallel else spmd.I,
                }
                input_partition_spec = spmd.PartitionSpec(
                    self.dp_axis,
                    (self.cp_axis, self.tp_axis) if seq_parallel else self.cp_axis,
                    None,
                )

                # check loss output type, check compute-time param annotations are restored.
                with (
                    spmd.set_current_mesh(self.dense_type_mesh),
                    typecheck(strict_mode="strict", local=False),
                ):
                    spmd.assert_type(
                        inp,
                        input_type,
                        partition_spec=input_partition_spec,
                    )
                    loss = model(inp)
                    self.assertEqual(
                        dict(spmd.get_local_type(loss)),
                        {
                            self.dp_axis: spmd.P,
                            self.cp_axis: spmd.P,
                            self.tp_axis: spmd.I,
                        },
                    )
                self.assertEqual(
                    model.compute_param_types,
                    (
                        {
                            self.dp_axis: spmd.R,
                            self.cp_axis: spmd.R,
                            self.tp_axis: spmd.R if seq_parallel else spmd.I,
                        },
                        {
                            self.dp_axis: spmd.R,
                            self.cp_axis: spmd.R,
                            self.tp_axis: spmd.V,
                        },
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

    def test_full_param_annotations_do_not_require_init_compute_mesh(self):
        """Full per-axis annotations are enough without an init compute mesh."""
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        spmd.assert_type(
            model.unsharded_weight,
            {self.dp_axis: spmd.R, self.cp_axis: spmd.R, self.tp_axis: spmd.I},
        )
        spmd.assert_type(
            model.sharded_weight,
            {
                self.dp_axis: spmd.R,
                self.cp_axis: spmd.R,
                self.tp_axis: spmd.S(0),
            },
        )
        fully_shard(
            model,
            mesh=self.dense_storage_mesh,
            dp_mesh_dims=DataParallelMeshDims(
                shard=("dp_shard", "cp"),
                replicate="dp_replicate",
            ),
        )

        inp = torch.randn(4, 8, 16)
        with (
            spmd.set_current_mesh(self.dense_type_mesh),
            typecheck(strict_mode="strict", local=False),
        ):
            spmd.assert_type(
                inp,
                {self.dp_axis: spmd.V, self.cp_axis: spmd.V, self.tp_axis: spmd.I},
                partition_spec=spmd.PartitionSpec(self.dp_axis, self.cp_axis, None),
            )
            loss = model(inp)
            self.assertEqual(
                dict(spmd.get_local_type(loss)),
                {self.dp_axis: spmd.P, self.cp_axis: spmd.P, self.tp_axis: spmd.I},
            )

    def test_local_v_param_requires_partition_spec(self):
        """Local-only V@TP params are ambiguous for FSDP.

        Without PartitionSpec shard info, FSDP cannot choose the matching
        DTensor Shard(dim), so it rejects the parameter at init time.
        """
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.dense_type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
            spmd.assert_type(
                model.sharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.V},
            )
        with (
            self.assertRaises(ValueError) as cm,
            spmd.set_current_mesh(self.dense_type_mesh),
        ):
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            """Cannot convert plain Varying to a DTensor placement. Use S(dim) to specify which tensor dimension is sharded.""",
        )

    def test_partial_param_annotations_infer_fsdp_axes_at_init(self):
        """Use init-time current_mesh to infer omitted FSDP axes as R.

        Params annotate only TP; FSDP fills DP/CP axes during fully_shard().
        """
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        spmd.assert_type(
            model.unsharded_weight,
            {self.tp_axis: spmd.I},
        )
        spmd.assert_type(
            model.sharded_weight,
            {self.tp_axis: spmd.S(0)},
        )
        with spmd.set_current_mesh(self.dense_type_mesh):
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )

        inp = torch.randn(4, 8, 16)
        with (
            spmd.set_current_mesh(self.dense_type_mesh),
            typecheck(strict_mode="strict", local=False),
        ):
            spmd.assert_type(
                inp,
                {self.dp_axis: spmd.V, self.cp_axis: spmd.V, self.tp_axis: spmd.I},
                partition_spec=spmd.PartitionSpec(self.dp_axis, self.cp_axis, None),
            )
            loss = model(inp)
            self.assertEqual(
                dict(spmd.get_local_type(loss)),
                {self.dp_axis: spmd.P, self.cp_axis: spmd.P, self.tp_axis: spmd.I},
            )

    def test_mixed_dense_sparse_params_use_per_param_restore_meshes(self):
        """
        Test restoring per-param dense and sparse type meshes.

        This models TorchTitan's FSDP setup, where one fully_shard call handles
        both dense-mesh & EP-sharded params, separated by shard_placement_fn.

        dense typecheck mesh: [dp, cp, tp]
        dense storage mesh: [dp_replicate, dp_shard, cp, tp]
        sparse mesh: [dp_replicate, efsdp, ep]

        With dense params annotated {dp: R, cp: R, tp: S(0)},
             sparse params annotated {dp_replicate: R, efsdp: R, ep: S(0)},
        FSDP restores each param on its original type mesh.
        """
        model = DenseSparseParams()
        spmd.assert_type(
            model.dense_weight,
            {self.dp_axis: spmd.R, self.cp_axis: spmd.R, self.tp_axis: spmd.S(0)},
        )
        spmd.assert_type(
            model.sparse_weight,
            {
                self.dp_replicate_axis: spmd.R,
                self.efsdp_axis: spmd.R,
                self.ep_axis: spmd.S(0),
            },
        )

        dense_mesh_info = _get_mesh_info(
            self.dense_storage_mesh,
            DataParallelMeshDims(shard=("dp_shard", "cp"), replicate="dp_replicate"),
        )
        sparse_mesh_info = _get_mesh_info(
            self.sparse_mesh,
            DataParallelMeshDims(shard="efsdp", replicate="dp_replicate"),
        )
        sparse_param = model.sparse_weight

        def shard_placement_fn(param):
            if param is sparse_param:
                return ShardPlacementResult(
                    placement=Shard(0),
                    mesh_info=sparse_mesh_info,
                )
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=dense_mesh_info,
            )

        fully_shard(
            model,
            mesh=self.dense_storage_mesh,
            dp_mesh_dims=DataParallelMeshDims(
                shard=("dp_shard", "cp"), replicate="dp_replicate"
            ),
            shard_placement_fn=shard_placement_fn,
        )

        with typecheck(strict_mode="strict", local=False):
            model()
        self.assertEqual(
            model.compute_param_types,
            (
                {self.dp_axis: spmd.R, self.cp_axis: spmd.R, self.tp_axis: spmd.V},
                {
                    self.dp_replicate_axis: spmd.R,
                    self.efsdp_axis: spmd.R,
                    self.ep_axis: spmd.V,
                },
            ),
        )

    def test_spmd_params_require_dp_mesh_dims(self):
        """Require explicit DP mesh axes before FSDP handles spmd_types params."""
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.dense_type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
            )
        with (
            self.assertRaises(ValueError) as cm,
            spmd.set_current_mesh(self.dense_type_mesh),
        ):
            fully_shard(model, mesh=self.dense_type_mesh["dp"])
        self.assertExpectedInline(
            str(cm.exception),
            "spmd_types parameters require fully_shard() to be called with both "
            "a named full DeviceMesh for FSDP storage and "
            "dp_mesh_dims=DataParallelMeshDims(shard=..., replicate=...), "
            "where shard names the mesh axis or axes FSDP shards parameters "
            "across and replicate names the HSDP/DDP mesh axis or axes FSDP "
            "replicates across.",
        )

    def test_fully_annotated_sparse_param_requires_sparse_storage_mesh(self):
        """Fully-annotated sparse param + dense FSDP mesh errors out."""
        model = nn.Linear(16, 16, bias=False)
        spmd.assert_type(
            model.weight,
            {
                self.dp_replicate_axis: spmd.R,
                self.efsdp_axis: spmd.R,
                self.ep_axis: spmd.S(0),
            },
        )

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'weight' has spmd_types annotation on axis mesh_ep, "
            "which is neither a non-FSDP storage-mesh axis nor contained in "
            "the FSDP DP mesh.",
        )

    def test_spmd_restore_mesh_must_contain_annotated_axes(self):
        """Partial TP annotations w/ sparse current_mesh context fail."""
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)
        spmd.assert_type(model.unsharded_weight, {self.tp_axis: spmd.I})

        with (
            self.assertRaises(ValueError) as cm,
            spmd.set_current_mesh(self.sparse_mesh),
        ):
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertIn(
            "parameter is annotated on axes",
            str(cm.exception),
        )
        self.assertIn("Annotate only axes in the compute mesh", str(cm.exception))

    def test_partial_non_storage_annotations_require_current_mesh(self):
        """Partial non-storage annotations need a shared current mesh."""
        model = nn.Linear(16, 16, bias=False)
        spmd.assert_type(model.weight, {self.ep_axis: spmd.S(0)})

        with self.assertRaises(ValueError) as cm:
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'weight' has partial spmd_types annotations that cannot "
            "be restored from its FSDP storage mesh. Fully annotate the "
            "parameter on its storage mesh, or wrap fully_shard() in "
            "spmd.set_current_mesh(...) when all parameters in the FSDP unit "
            "share one typechecking mesh. Mixed per-parameter typechecking "
            "meshes with partial annotations are not supported. Annotated "
            "axes: (mesh_ep,). Storage mesh axes: (mesh_dp_replicate, "
            "mesh_dp_shard, mesh_cp, mesh_tp).",
        )

    def test_partial_param_annotations_missing_non_fsdp_axis_errors(self):
        """Require annotations for storage axes that FSDP does not manage."""
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        spmd.assert_type(
            model.unsharded_weight,
            {self.dp_axis: spmd.R, self.cp_axis: spmd.R},
        )
        spmd.assert_type(
            model.sharded_weight,
            {self.tp_axis: spmd.S(0)},
        )
        with (
            self.assertRaises(ValueError) as cm,
            spmd.set_current_mesh(self.dense_type_mesh),
        ):
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Parameter 'unsharded_weight' has incomplete spmd_types annotations "
            "for FSDP storage. Annotated axes: (mesh_dp, mesh_cp). Storage "
            "mesh axes: (mesh_dp_replicate, mesh_dp_shard, mesh_cp, mesh_tp). FSDP mesh "
            "dims: DataParallelMeshDims(shard=('dp_shard', 'cp'), "
            "replicate='dp_replicate'). Missing non-FSDP storage axes: "
            "(mesh_tp,).",
        )

    def test_fsdp_dp_axes_must_be_r(self):
        """FSDP-managed DP axes must be Replicate in spmd_types annotations."""
        model = SpmdLinear(self.dense_type_mesh, seq_parallel=False)

        with spmd.set_current_mesh(self.dense_type_mesh):
            spmd.assert_type(
                model.unsharded_weight,
                {"dp": spmd.I, "cp": spmd.R, "tp": spmd.I},
            )
        with (
            self.assertRaises(ValueError) as cm,
            spmd.set_current_mesh(self.dense_type_mesh),
        ):
            fully_shard(
                model,
                mesh=self.dense_storage_mesh,
                dp_mesh_dims=DataParallelMeshDims(
                    shard=("dp_shard", "cp"),
                    replicate="dp_replicate",
                ),
            )
        self.assertExpectedInline(
            str(cm.exception),
            "Expected spmd.R on FSDP DP axis mesh_dp for parameter "
            "'unsharded_weight' but got PerMeshAxisLocalSpmdType.I. FSDP "
            "requires DP parameters to be R since it handles the DP gradient "
            "reduction.",
        )


if __name__ == "__main__":
    run_tests()
