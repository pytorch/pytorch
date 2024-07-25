# Owner(s): ["module: c10d"]
import unittest

import torch
import torch.distributed as dist
from functorch import make_fx
from torch._inductor.fx_passes.micro_pipeline_tp import (
    _get_unexposed_collectives,
    find_all_gather_patterns,
    find_reduce_scatter_patterns,
)
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    reduce_scatter_tensor,
)
from torch.distributed._symmetric_memory import _test_mode
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._triton import has_triton


@instantiate_parametrized_tests
class MicroPipelineTPTest(TestCase):
    def setUp(self):
        torch._inductor.config._micro_pipeline_tp = True

        self.rank = 0
        self.world_size = 2
        torch.cuda.set_device("cuda:0")

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        dist.destroy_process_group()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_find_all_gather_patterns(self):
        group = dist.group.WORLD

        def func(inp: torch.Tensor) -> torch.Tensor:
            a = all_gather_tensor(inp, gather_dim=0, group=group.group_name)
            b = all_gather_tensor(inp, gather_dim=1, group=group.group_name)
            return a, b

        inp = torch.rand(64, 32, device="cuda")

        gm = make_fx(func)(inp)
        all_gathers = find_all_gather_patterns(gm.graph)
        self.assertEqual(len(all_gathers), 2)

        # If this test fails, please update find_all_gather_patterns instead of
        # modifying the following assertions.
        for all_gather in all_gathers:
            self.assertEqual(
                all_gather.shard_node.op,
                "placeholder",
            )
            self.assertEqual(
                all_gather.ag_node.target,
                torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            self.assertEqual(all_gather.group_name, group.group_name)

        self.assertEqual(all_gathers[0].gather_dim, 0)
        self.assertEqual(
            all_gathers[0].res_node.target,
            torch.ops._c10d_functional.wait_tensor.default,
        )

        self.assertEqual(all_gathers[1].gather_dim, 1)
        self.assertEqual(
            all_gathers[1].res_node.target,
            torch.ops.aten.cat.default,
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_find_reduce_scatter_patterns(self):
        group = dist.group.WORLD

        def func(inp: torch.Tensor) -> torch.Tensor:
            a = reduce_scatter_tensor(inp, "sum", scatter_dim=0, group=group.group_name)
            b = reduce_scatter_tensor(inp, "avg", scatter_dim=1, group=group.group_name)
            return a, b

        inp = torch.rand(64, 32, device="cuda")

        gm = make_fx(func)(inp)
        reduce_scatters = find_reduce_scatter_patterns(gm.graph)
        self.assertEqual(len(reduce_scatters), 2)

        # If this test fails, please update find_reduce_scatter_patterns
        # instead of modifying the following assertions.
        for reduce_scatter in reduce_scatters:
            self.assertEqual(
                reduce_scatter.input_node.op,
                "placeholder",
            )
            self.assertEqual(
                reduce_scatter.rs_node.target,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            self.assertEqual(
                reduce_scatter.res_node.target,
                torch.ops._c10d_functional.wait_tensor.default,
            )
            self.assertEqual(reduce_scatter.group_name, group.group_name)

        self.assertEqual(reduce_scatters[0].reduce_op, "sum")
        self.assertEqual(reduce_scatters[0].scatter_dim, 0)

        self.assertEqual(reduce_scatters[1].reduce_op, "avg")
        self.assertEqual(reduce_scatters[1].scatter_dim, 1)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @fresh_inductor_cache()
    def test_get_unexposed_collectives(self):
        group = dist.group.WORLD

        def func(inp: torch.Tensor) -> torch.Tensor:
            a = inp @ inp.T
            # b is unexposed (hidden by a)
            b = all_gather_tensor(inp, gather_dim=0, group=group.group_name)
            c = b @ inp.T
            # d is unexposed (hidden by c)
            d = reduce_scatter_tensor(b, "avg", scatter_dim=0, group=group.group_name)
            # e is exposed
            e = all_gather_tensor(d, gather_dim=0, group=group.group_name)
            return a, c, e

        inp = torch.rand(64, 32, device="cuda")

        gm = make_fx(func)(inp)
        overlappable_collectives = _get_unexposed_collectives(gm.graph)
        self.assertEqual(
            list(map(str, overlappable_collectives)),
            ["all_gather_into_tensor", "reduce_scatter_tensor"],
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("gather_dim", [0, 1, 2])
    @fresh_inductor_cache()
    def test_fuse_all_gather_matmul(self, A_dims, gather_dim):
        if gather_dim >= A_dims:
            return

        group = dist.group.WORLD

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            A = all_gather_tensor(A_shard, gather_dim=gather_dim, group=group)
            return A @ B

        if A_dims == 2:
            A_shard_shape = [64, 32]
        elif A_dims == 3:
            A_shard_shape = [2, 64, 32]
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")

        A_shard_shape[gather_dim] //= self.world_size
        A_shard = torch.rand(*A_shard_shape, device="cuda")
        B = torch.rand(32, 16, device="cuda")

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, A_shard, B)

        if gather_dim == A_dims - 1:
            assert "fused_all_gather_matmul" not in code
            assert "all_gather_into_tensor" in code
        else:
            # Decomposing the matmul on the K dimension is not supported
            assert "fused_all_gather_matmul" in code
            assert "all_gather_into_tensor" not in code

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("scatter_dim", [0, 1, 2])
    @fresh_inductor_cache()
    def test_fuse_matmul_reduce_scatter(self, A_dims, scatter_dim):
        if scatter_dim >= A_dims:
            return

        group = dist.group.WORLD

        def func(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return reduce_scatter_tensor(A @ B, "avg", scatter_dim, group)

        if A_dims == 2:
            A = torch.rand(64, 32, device="cuda")
        elif A_dims == 3:
            A = torch.rand(2, 64, 32, device="cuda")
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")
        B = torch.rand(32, 16, device="cuda")

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, A, B)

        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("shard_dim", [0, 1])
    @fresh_inductor_cache()
    def test_dtensor_seq_par(self, shard_dim: int):
        model = MLPModule(device="cuda", bias=False)
        device_mesh = DeviceMesh(
            "cuda",
            torch.arange(0, self.world_size),
        )
        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(shard_dim)),
            "net2": RowwiseParallel(output_layouts=Shard(shard_dim)),
        }
        model = parallelize_module(model, device_mesh, parallelize_plan)
        if shard_dim == 0:
            inp = torch.rand(8, 10, device="cuda")
        elif shard_dim == 1:
            inp = torch.rand(2, 8, 10, device="cuda")
        else:
            raise AssertionError("Invalid shard_dim")

        with _test_mode():
            compiled = torch.compile(model)
            code = run_and_get_triton_code(compiled, inp)

        self.assertIn("fused_all_gather_matmul", code)
        self.assertNotIn("all_gather_into_tensor", code)
        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)


if __name__ == "__main__":
    run_tests()
