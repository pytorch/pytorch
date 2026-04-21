# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.spmd_types import (
    assert_type,
    get_axis_local_type,
    get_partition_spec,
    has_local_type,
    I,
    normalize_axis,
    PartitionSpec,
    R,
    S,
)
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests
from torch.utils._debug_mode import _OpCall, DebugMode


def _count_fsdp_ops(debug_mode):
    ops = [str(o.op) for o in debug_mode.operators if isinstance(o, _OpCall)]
    ag = sum(1 for o in ops if o == "fsdp.all_gather_copy_in.default")
    rs = sum(1 for o in ops if o == "fsdp.chunk_cat.default")
    return ag, rs


class TestFullyShardSpmdTypes(FSDPTestMultiThread):
    """Integration tests verifying that fully_shard preserves spmd_types
    annotations through the shard -> unshard round-trip."""

    @property
    def world_size(self) -> int:
        return 4

    def _get_dp_axis(self, model):
        state = model._get_fsdp_state()
        pg = state._fsdp_param_groups[0].mesh_info.shard_process_group
        return normalize_axis(pg)

    def _get_fsdp_params(self, model):
        state = model._get_fsdp_state()
        return [p for g in state._fsdp_param_groups for p in g.fsdp_params]

    def test_saved_annotation_while_sharded(self):
        """After fully_shard, the FSDPParam has the correct saved annotation
        (I transformed to R) before any unshard happens."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: I})

        fully_shard(model)

        fsdp_params = self._get_fsdp_params(model)
        self.assertEqual(len(fsdp_params), 1)
        dp_axis = self._get_dp_axis(model)
        self.assertIsNotNone(fsdp_params[0]._spmd_local_type)
        self.assertIs(fsdp_params[0]._spmd_local_type[dp_axis], R)
        self.assertIsNone(fsdp_params[0]._spmd_partition_spec)

    def test_saved_annotation_with_partition_spec(self):
        """S(0) annotation produces a PartitionSpec referencing the DP axis."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: S(0)})

        fully_shard(model)

        fsdp_params = self._get_fsdp_params(model)
        dp_axis = self._get_dp_axis(model)
        self.assertEqual(
            fsdp_params[0]._spmd_partition_spec, PartitionSpec(dp_axis, None)
        )

    def test_no_saved_annotation_when_unannotated(self):
        """Unannotated params have None saved state after fully_shard."""
        model = nn.Linear(8, 4, bias=False)
        fully_shard(model)

        fsdp_params = self._get_fsdp_params(model)
        self.assertIsNone(fsdp_params[0]._spmd_local_type)
        self.assertIsNone(fsdp_params[0]._spmd_partition_spec)

    def test_annotated_param_survives_shard_unshard(self):
        """Annotations set before fully_shard appear on the unsharded param
        after an explicit unshard."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: I})

        fully_shard(model)
        model.unshard()

        dp_axis = self._get_dp_axis(model)
        self.assertIs(get_axis_local_type(model.weight, dp_axis), R)

    def test_unannotated_param_no_annotation_after_unshard(self):
        """Params without annotations remain unannotated after unshard."""
        model = nn.Linear(8, 4, bias=False)
        fully_shard(model)
        model.unshard()
        self.assertFalse(has_local_type(model.weight))

    def test_annotation_after_forward(self):
        """Annotations are present on the unsharded param exposed during
        forward, and an all-gather is performed."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: I})

        fully_shard(model, reshard_after_forward=False)
        inp = torch.randn(2, 8)
        with DebugMode() as dm:
            model(inp)

        ag, _ = _count_fsdp_ops(dm)
        self.assertGreater(ag, 0)
        dp_axis = self._get_dp_axis(model)
        self.assertIs(get_axis_local_type(model.weight, dp_axis), R)

    def test_annotation_restored_after_forward_backward(self):
        """Annotations survive a full forward + backward pass, which performs
        an all-gather (forward) and a reduce-scatter (backward)."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: I})

        fully_shard(model)
        inp = torch.randn(2, 8)
        with DebugMode() as dm:
            model(inp).sum().backward()

        ag, rs = _count_fsdp_ops(dm)
        self.assertGreater(ag, 0)
        self.assertGreater(rs, 0)

        model.unshard()
        dp_axis = self._get_dp_axis(model)
        self.assertIs(get_axis_local_type(model.weight, dp_axis), R)

    def test_multi_module_annotations(self):
        """Each submodule's annotations are independently preserved."""
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 4, bias=False),
        )
        dp = self._get_default_pg()
        assert_type(model[0].weight, {dp: I})
        # Leave model[2] unannotated

        fully_shard(model[0])
        fully_shard(model)

        model[0].unshard()
        model.unshard()

        dp_axis_0 = normalize_axis(
            model[0]
            ._get_fsdp_state()
            ._fsdp_param_groups[0]
            .mesh_info.shard_process_group
        )
        self.assertIs(get_axis_local_type(model[0].weight, dp_axis_0), R)
        self.assertFalse(has_local_type(model[2].weight))

    def test_dp_axis_becomes_R_not_I(self):
        """The DP axis annotation is transformed from I to R (not preserved
        as-is), matching the all-gather semantics."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: I})

        fully_shard(model)
        model.unshard()

        dp_axis = self._get_dp_axis(model)
        self.assertIs(get_axis_local_type(model.weight, dp_axis), R)
        self.assertIsNot(get_axis_local_type(model.weight, dp_axis), I)

    def test_partition_spec_preserved(self):
        """A PartitionSpec set alongside a local type is restored."""
        model = nn.Linear(8, 4, bias=False)
        dp = self._get_default_pg()
        assert_type(model.weight, {dp: S(0)})

        fully_shard(model)
        model.unshard()

        spec = get_partition_spec(model.weight)
        dp_axis = self._get_dp_axis(model)
        self.assertEqual(spec, PartitionSpec(dp_axis, None))

    def _get_default_pg(self):
        import torch.distributed as dist

        return dist.distributed_c10d._get_default_group()


class TestFullyShardSpmdTypesMLP(FSDPTestMultiThread):
    """Test annotations on a multi-layer model through training steps."""

    @property
    def world_size(self) -> int:
        return 4

    def test_mlp_annotations_through_training(self):
        """Annotations survive multiple train steps on an MLP."""
        model = MLP(8)
        dp = self._get_default_pg()
        assert_type(model.in_proj.weight, {dp: I})
        assert_type(model.out_proj.weight, {dp: I})

        fully_shard(model.in_proj)
        fully_shard(model.out_proj)
        fully_shard(model)

        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        total_ag, total_rs = 0, 0
        for _ in range(3):
            inp = torch.randn(2, 8)
            with DebugMode() as dm:
                model(inp).sum().backward()
            ag, rs = _count_fsdp_ops(dm)
            total_ag += ag
            total_rs += rs
            optim.step()
            optim.zero_grad()

        self.assertGreater(total_ag, 0)
        self.assertGreater(total_rs, 0)

        for submodule in [model.in_proj, model.out_proj]:
            submodule.unshard()
        model.unshard()

        dp_axis = normalize_axis(
            model.in_proj._get_fsdp_state()
            ._fsdp_param_groups[0]
            .mesh_info.shard_process_group
        )
        for name in ["in_proj", "out_proj"]:
            param = getattr(model, name).weight
            self.assertIs(get_axis_local_type(param, dp_axis), R)

    def _get_default_pg(self):
        import torch.distributed as dist

        return dist.distributed_c10d._get_default_group()


class TestFullyShardSpmdTypesDTensor(FSDPTestMultiThread):
    """Tests that spmd_types annotations on plain tensors produce multi-dim
    DTensors that are aware of non-DP (e.g. TP) placements."""

    @property
    def world_size(self) -> int:
        return 4

    def _make_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

    def test_tp_shard_round_trip(self):
        """S(0) on TP: fully_shard creates 2D DTensor, unshard produces
        TP DTensor with correct shape and spmd_types annotation."""
        from torch.distributed.tensor import DTensor, Shard
        from torch.distributed.tensor.placement_types import _StridedShard

        mesh = self._make_mesh()
        dp_pg = mesh.get_group("dp")
        tp_pg = mesh.get_group("tp")
        model = nn.Linear(8, 4, bias=False)
        local_shape = model.weight.shape
        assert_type(model.weight, {dp_pg: I, tp_pg: S(0)})

        fully_shard(model, mesh=mesh["dp"])
        param = model.weight
        self.assertIsInstance(param, DTensor)
        self.assertEqual(param._spec.mesh.ndim, 2)
        dp_p, tp_p = param._spec.placements
        self.assertIsInstance(dp_p, _StridedShard)
        self.assertIsInstance(tp_p, Shard)
        self.assertEqual(param._spec.tensor_meta.shape[0], local_shape[0] * 2)

        model.unshard()
        w = model.weight
        self.assertIsInstance(w, DTensor)
        self.assertEqual(w._spec.placements, (Shard(0),))
        self.assertEqual(w._local_tensor.shape, local_shape)
        self.assertTrue(has_local_type(w))
        self.assertIs(get_axis_local_type(w, normalize_axis(dp_pg)), R)

    def test_tp_replicate(self):
        """R on TP: fully_shard creates 2D DTensor with Replicate on TP dim."""
        from torch.distributed.tensor import DTensor, Replicate, Shard

        mesh = self._make_mesh()
        model = nn.Linear(8, 4, bias=False)
        assert_type(
            model.weight,
            {mesh.get_group("dp"): I, mesh.get_group("tp"): R},
        )
        fully_shard(model, mesh=mesh["dp"])

        param = model.weight
        self.assertIsInstance(param, DTensor)
        self.assertEqual(param._spec.mesh.ndim, 2)
        dp_p, tp_p = param._spec.placements
        self.assertIsInstance(dp_p, Shard)
        self.assertIsInstance(tp_p, Replicate)
        self.assertEqual(param._spec.tensor_meta.shape, torch.Size([4, 8]))

    def test_no_root_mesh_uses_plain_path(self):
        """Standalone 1D DP mesh with annotation falls through to plain path."""
        model = nn.Linear(8, 4, bias=False)
        import torch.distributed as dist

        dp = dist.distributed_c10d._get_default_group()
        assert_type(model.weight, {dp: I})

        fully_shard(model)
        state = model._get_fsdp_state()
        fsdp_param = state._fsdp_param_groups[0].fsdp_params[0]
        self.assertIsNone(fsdp_param._unsharded_dtensor_spec)


if __name__ == "__main__":
    run_tests()
