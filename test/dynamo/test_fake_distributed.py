# Owner(s): ["module: dynamo"]
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._dynamo.test_case import TestCase as DynamoTestCase
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import instantiate_parametrized_tests


if dist.is_available():
    from torch.distributed._functional_collectives import (
        all_to_all_single_autograd,
        wait_tensor,
    )
    from torch.testing._internal.distributed.fake_pg import FakeStore


def normalize_graph(gm):
    return normalize_gm(gm.print_readable(print_output=False))


@skipIf(not dist.is_available(), "requires distributed")
class TestFakeDistributed(DynamoTestCase):
    def setUp(self):
        # Use FakeProcessGroup to run tests on a single process
        self.store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=self.store)

    def tearDown(self):
        dist.destroy_process_group()

    def test_all_to_all_single_autograd(self):
        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x):
            return all_to_all_single_autograd(
                x,
                None,  # Will use equal splits
                None,  # Will use equal splits
                group=dist.group.WORLD,
            )

        # Test backed shapes
        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", primals_3: "f32[s77, s27]"):
        floordiv: "Sym((s77//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(primals_3, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_3 = None

        wait_tensor: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s27)", floordiv: "Sym((s77//2))", tangents_1: "f32[2*((s77//2)), s27]"):
        all_to_all_single_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((s77//2)), s27]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, wait_tensor_1)
""",  # noqa: B950
        )

        backend.fw_graphs.clear()
        backend.bw_graphs.clear()

        # Test unbacked shapes
        x = torch.randn(8, 8, 8, requires_grad=True)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        torch._dynamo.decorators.mark_unbacked(x, 2)
        wait_tensor(fn(x))
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", primals_4: "f32[u0, u1, u2]"):
        ge_1: "Sym(u0 >= 0)" = primals_1 >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge_1 = _assert_scalar = None
        ge_3: "Sym(u1 >= 0)" = primals_2 >= 0
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(ge_3, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_3 = _assert_scalar_1 = None
        ge_5: "Sym(u2 >= 0)" = primals_3 >= 0
        _assert_scalar_2 = torch.ops.aten._assert_scalar.default(ge_5, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_5 = _assert_scalar_2 = None

        floordiv: "Sym((u0//2))" = primals_1 // 2

        all_to_all_single: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(primals_4, [floordiv, floordiv], [floordiv, floordiv], '0');  primals_4 = None

        wait_tensor: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single);  all_to_all_single = None
        return (wait_tensor, primals_1, primals_2, primals_3, floordiv)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(u0)", primals_2: "Sym(u1)", primals_3: "Sym(u2)", floordiv: "Sym((u0//2))", tangents_1: "f32[2*((u0//2)), u1, u2]"):
        all_to_all_single_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.all_to_all_single.default(tangents_1, [floordiv, floordiv], [floordiv, floordiv], '0');  tangents_1 = floordiv = None
        wait_tensor_1: "f32[2*((u0//2)), u1, u2]" = torch.ops._c10d_functional.wait_tensor.default(all_to_all_single_1);  all_to_all_single_1 = None
        return (None, None, None, wait_tensor_1)
""",  # noqa: B950
        )


instantiate_parametrized_tests(TestFakeDistributed)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
