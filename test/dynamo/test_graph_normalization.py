# Owner(s): ["module: dynamo"]
# flake8: noqa: B950
import torch
import torch.fx
from torch._dynamo.test_case import TestCase


class GraphNormalizationTests(TestCase):
    def test_basic_normlization(self):
        # We do a normalization pass that effectively ensures that
        # the name indexes monotonically increase. This typically
        # already happens but in some cases, such as in HOPs, the
        # invariant could be broken without normalization. Below we
        # show an example where cond previously would have jumped
        # from getitem_3 to get_item_2, but with normalization correctly
        # uses getitem_4 after getitem_3.

        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x[:2]

                return cond(x.shape[0] <= 2, true_fn, false_fn, [x])

        x = torch.randn(2, 2)
        mod = Module()
        out_graph, _ = torch._dynamo.export(mod)(x)

        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    l_x_ = arg0
    sym_size_int_1 = torch.ops.aten.sym_size.int(l_x_, 0)
    le_1 = sym_size_int_1 <= 2;  sym_size_int_1 = None
    cond_true_1 = self.cond_true_0
    cond_false_1 = self.cond_false_0
    cond_1 = torch.ops.higher_order.cond(le_1, cond_true_1, cond_false_1, [l_x_]);  le_1 = cond_true_1 = cond_false_1 = l_x_ = None
    getitem_1 = cond_1[0]
    sym_size_int_2 = torch.ops.aten.sym_size.int(getitem_1, 0);  getitem_1 = None
    sym_constrain_range_for_size_default_1 = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_2);  sym_constrain_range_for_size_default_1 = None
    ge_1 = sym_size_int_2 >= 2;  sym_size_int_2 = None
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 2 on node 'ge'");  ge_1 = _assert_scalar_default_1 = None
    getitem_2 = cond_1[0];  cond_1 = None
    return pytree.tree_unflatten([getitem_2], self._out_spec)""",  # noqa: B950
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
