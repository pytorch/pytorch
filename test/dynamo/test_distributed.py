# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing

from torch._dynamo.testing import normalize_gm


class DistributedTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(trace_distributed=True)
    def test_fsdp_same_storage_size_allowed(self):
        import torch.distributed.fsdp._flat_param as flat_param

        def foo(x, y):
            return flat_param._same_storage_size(x, y)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        foo_cmp = torch._dynamo.optimize(backend, nopython=True)(foo)
        foo_aot_autograd = torch._dynamo.optimize("aot_eager", nopython=True)(foo)

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])
        self.assertEqual(foo_cmp(x, y), foo(x, y))
        self.assertEqual(foo_aot_autograd(x, y), foo(x, y))
        self.assertEqual(foo_cmp(x, x), foo(x, x))
        self.assertEqual(foo_aot_autograd(x, x), foo(x, x))
        gm = backend.graphs[0]
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
    class GraphModule(torch.nn.Module):
        def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
            l_x_ = L_x_
            l_y_ = L_y_

            _same_storage_size = torch.distributed.fsdp._flat_param._same_storage_size(l_x_, l_y_);  l_x_ = l_y_ = None
            return (_same_storage_size,)
    """,
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _same_storage_size = torch.distributed.fsdp._flat_param._same_storage_size(l_x_, l_y_);  l_x_ = l_y_ = None
        return (_same_storage_size,)
""",
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
