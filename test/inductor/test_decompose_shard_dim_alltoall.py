# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch import fx
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.post_grad import _decompose_shard_dim_alltoall
from torch._inductor.fx_utils import FakeTensorUpdater
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import IS_LINUX


class TestDecomposeShardDimAllToAll(TestCase):
    def _reference_shard_dim_alltoall(
        self,
        inputs: list[torch.Tensor],
        *,
        gather_dim: int,
        shard_dim: int,
    ) -> list[torch.Tensor]:
        group_size = len(inputs)
        ndim = inputs[0].dim()
        gathered = torch.cat(inputs, dim=gather_dim % ndim)
        return list(torch.chunk(gathered, group_size, dim=shard_dim % ndim))

    def _make_graph_module(
        self,
        *,
        shape: tuple[int, ...] = (5, 7),
        dtype: torch.dtype = torch.float32,
        gather_dim: int = 0,
        shard_dim: int = 1,
    ) -> fx.GraphModule:
        graph = fx.Graph()
        inp = graph.placeholder("inp")
        inp.meta["val"] = torch.empty(shape, dtype=dtype)
        shard_dim_alltoall = graph.call_function(
            torch.ops._dtensor.shard_dim_alltoall.default,
            args=(inp, gather_dim, shard_dim, "test_pg"),
        )
        graph.output(shard_dim_alltoall)
        return fx.GraphModule({}, graph)

    def _run_pass(
        self,
        gm: fx.GraphModule,
        *,
        group_size: int = 4,
    ) -> None:
        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                node.meta["val"] = fake_mode.from_tensor(val)

        try:
            fake_tensor_updater = FakeTensorUpdater(gm)
        except AttributeError:
            fake_tensor_updater = FakeTensorUpdater(gm.graph)

        with (
            mock.patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved_pg",
            ),
            mock.patch(
                "torch.distributed.distributed_c10d._get_group_size_by_name",
                return_value=group_size,
            ),
        ):
            _decompose_shard_dim_alltoall(gm)
            with V.set_fake_mode(fake_mode):
                fake_tensor_updater.incremental_update()

    def _movedim_args(self, gm: fx.GraphModule) -> list[tuple[int, int]]:
        return [
            (node.args[1], node.args[2])
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.movedim.int
        ]

    def _wait_tensor_node(self, gm: fx.GraphModule) -> fx.Node:
        return next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.wait_tensor.default
        )

    def test_decomposes_divisible_shard_dim_alltoall(self) -> None:
        counters.clear()
        gm = self._make_graph_module(shape=(5, 8))

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertIn(torch.ops._c10d_functional.wait_tensor.default, targets)

        alltoall_node = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.all_to_all_single.default
        )
        self.assertEqual(alltoall_node.args[1], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[2], [1, 1, 1, 1])
        self.assertEqual(alltoall_node.args[3], "test_pg")

        view_shapes = [
            node.args[1]
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.view.default
        ]
        self.assertIn([5, 4, 2], view_shapes)
        self.assertIn([20, 2], view_shapes)
        self.assertEqual(self._movedim_args(gm), [(1, 0), (0, 0)])
        self.assertEqual(self._wait_tensor_node(gm).meta["val"].shape, (4, 5, 2))
        self.assertIn(
            torch.ops.aten.clone.default,
            [node.target for node in gm.graph.nodes],
        )
        self.assertNotIn(
            torch.ops.aten.contiguous.default,
            [node.target for node in gm.graph.nodes],
        )
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 1)

    def test_decomposition_matches_local_reference_semantics(self) -> None:
        group_size = 4
        gather_dim = 2
        shard_dim = 0
        inputs = torch.arange(group_size * 8 * 5 * 6, dtype=torch.float32).reshape(
            group_size, 8, 5, 6
        )
        per_rank_inputs = list(inputs.unbind(0))

        expected = self._reference_shard_dim_alltoall(
            per_rank_inputs,
            gather_dim=gather_dim,
            shard_dim=shard_dim,
        )

        # This mirrors the pass: expose the process-group dimension as dim0,
        # all-to-all one slice per rank, then restore the final logical shape.
        pre_alltoall = [
            inp.view(group_size, 2, 5, 6).movedim(shard_dim, 0)
            for inp in per_rank_inputs
        ]
        actual = [
            torch.cat(
                [pre.narrow(0, rank, 1) for pre in pre_alltoall],
                dim=0,
            )
            .movedim(0, gather_dim)
            .clone(memory_format=torch.contiguous_format)
            .view(2, 5, 24)
            for rank in range(group_size)
        ]

        self.assertEqual(actual, expected)

    def test_skips_non_divisible_shard_dim_alltoall(self) -> None:
        counters.clear()
        gm = self._make_graph_module(shape=(5, 7))

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertNotIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 0)

    def test_decomposes_when_gather_dim_is_after_shard_dim(self) -> None:
        counters.clear()
        gm = self._make_graph_module(shape=(8, 5, 6), gather_dim=2, shard_dim=0)

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertNotIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertIn(torch.ops._c10d_functional.all_to_all_single.default, targets)

        view_shapes = [
            node.args[1]
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.view.default
        ]
        self.assertIn([4, 2, 5, 6], view_shapes)
        self.assertIn([2, 5, 24], view_shapes)
        self.assertEqual(self._movedim_args(gm), [(0, 0), (0, 2)])
        self.assertEqual(self._wait_tensor_node(gm).meta["val"].shape, (4, 2, 5, 6))
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 1)

    def test_skips_complex_dtype(self) -> None:
        counters.clear()
        gm = self._make_graph_module(shape=(5, 8), dtype=torch.complex64)

        self._run_pass(gm)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(torch.ops._dtensor.shard_dim_alltoall.default, targets)
        self.assertNotIn(torch.ops._c10d_functional.all_to_all_single.default, targets)
        self.assertEqual(counters["inductor"]["decompose_shard_dim_alltoall"], 0)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
