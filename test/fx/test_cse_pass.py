# Owner(s): ["oncall: fx"]

import random

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.dialect.common.cse_pass import CSEPass, get_CSE_banned_ops
from torch.testing._internal.common_utils import run_tests, TestCase


banned_ops = get_CSE_banned_ops()
P_default = CSEPass(banned_ops=banned_ops)


def check(self, f, t, delta, check_val=True, graph_input=False, P=None):
    """
    check if the CSE modified graph of ``f``
    1) has delta less nodes, and
    2) do not reduce the number of nodes further on a second pass, and
    3) modified returned is true only if the number of nodes decreases.

    Args:
        f: function to be checked
        t: tensor to be passed to f
        delta: an integer >= -1.
               If delta = -1, it only checks if the new graph has less or equal number of nodes
        check_val: if True, check if the output of f is correct
        graph_input: True is f is type GraphModule
        P: the pass to use. If None, use P_default
    """
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)

    if P is None:
        P = P_default

    res = P(fx_g)
    new_g = res.graph_module
    new_graph = new_g.graph
    modified = res.modified

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)

    assert (
        new_num_nodes < old_num_nodes
    ) == modified, "modified should be True if the number of nodes decrease"

    if delta == -1:
        self.assertTrue(
            old_num_nodes >= new_num_nodes,
            (f"number of nodes increased {old_num_nodes}, {new_num_nodes}"),
        )
    else:
        self.assertTrue(
            old_num_nodes == new_num_nodes + delta,
            (
                f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
            ),
        )

    # a second pass should not reduce more nodes
    res = P(new_g)
    pass_2_graph = res.graph_module.graph
    pass_2_num_nodes = len(pass_2_graph.nodes)
    self.assertTrue(
        pass_2_num_nodes == new_num_nodes,
        (
            f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
        ),
    )

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:  # both return None
            self.assertTrue(
                our_result is None, f"true result is None, CSE result is {our_result}"
            )
        else:  # results returned are the same
            self.assertTrue(
                torch.all(true_result == our_result),
                (f"results are different {true_result}, {our_result}"),
            )  # check results are the same


class TestCSEPass(TestCase):
    def test_nochange(self):
        def f(x):
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d

        t = torch.randn(2, 2)
        check(self, f, t, 0)

    def test_empty(self):
        def f(x):
            pass

        t = torch.randn(2, 2)
        check(self, f, t, 0)

    def test_immutable_list_type(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_immutable_list_multiple_entries(self):
        def f(x):
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple(self):
        def f(x):
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple_2(self):
        def f(x):
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(1)
        check(self, f, t, 3)

    def test_two_args_default(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 3)

    def test_two_args(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple_multiple_same_ops(self):
        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 3)

    def test_nested_immutable_list_type(self):
        def f(x):
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 1)

    def test_kwarg(self):
        def f(x):
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 1)

    """
    Generate function with random ops and check if the result is the same
    """

    def test_random(self):
        def f(x):
            vals = [x]
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        fx_g = symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = torch.randn(2, 2)

        for _ in range(30):
            check(self, fx_g, t, -1, graph_input=True)

    """
    Test that banned list ban ops as expected.
    """

    def test_banned_list(self):
        def f(x):
            a = x + 1
            b = x + 1
            return a + b

        t = torch.randn(2, 2)
        P_ban_add = P = CSEPass(banned_ops=[torch.ops.aten.add])
        check(self, f, t, 0, P=P_ban_add)  # check that add is banned
        check(self, f, t, 1)  # check that add is not banned by default

    def test_rand_like(self):
        def f(x):
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 0, check_val=False)

    def test_rand_n(self):
        def f(x):
            a = torch.randn(4)
            b = torch.randn(4)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 0, check_val=False)


if __name__ == "__main__":
    run_tests()
