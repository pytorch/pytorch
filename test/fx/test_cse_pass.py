# Owner(s): ["oncall: fx"]
import operator
import random
from copy import deepcopy

import torch
from torch._ops import OpOverload
from torch.fx import Node, symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


P_default = CSEPass()


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

    assert (new_num_nodes < old_num_nodes) == modified, (
        "modified should be True if the number of nodes decrease"
    )

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
        true_result = fx_g(t.clone())
        our_result = new_g(t.clone())
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

        def banned_impure(n: Node) -> bool:
            banned_ops = [torch.ops.aten.add]
            if isinstance(n.target, OpOverload):
                return n.target.overloadpacket in banned_ops
            return False

        P_ban_add = CSEPass(is_impure_node=banned_impure)
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

    """
    Check that common subexpressions across two computation chains are eliminated.
    """

    def test_double_elimination(self):
        def f(x):
            a = x + 0.3
            b = x + 0.3
            c = 3.0 * a
            d = 3.0 * b
            return c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2, check_val=False)

    """
    Check that in-place operations are not eliminated.
    """

    def test_inplace(self):
        def f(x):
            x.add_(2)
            z = 2 * x
            x.add_(2)
            y = 2 * x
            return y + z

        t = torch.randn(2, 2)

        def is_impure(n: Node) -> bool:
            if n.is_impure():
                return True
            if n.target == "add_":
                return True
            return False

        graph, _ = torch._dynamo.export(f)(t)
        cse_graph = CSEPass(is_impure).call(graph).graph_module
        assert torch.allclose(f(deepcopy(t)), cse_graph(deepcopy(t)))

    def test_inplace2(self):
        def f(x):
            z = 2 * x
            x += 2
            y = 2 * x
            return y + z

        t = torch.randn(2, 2)

        graph, _ = torch._dynamo.export(f)(t)
        cse_graph = P_default.call(graph).graph_module
        assert torch.allclose(f(deepcopy(t)), cse_graph(deepcopy(t)))

    def test_inplace_with_double_elimination(self):
        def f(x):
            l = torch.zeros(())
            l += 2 * x.sum()

            l2 = torch.zeros(())
            l2 += 2 * x.sum() + 5

            return l + l2

        t = torch.randn(2, 2)

        def is_impure(n: Node) -> bool:
            if n.is_impure():
                return True
            if n.target == operator.iadd:
                return True
            return False

        graph, _ = torch._dynamo.export(f)(t)
        cse_graph = CSEPass(is_impure).call(graph).graph_module
        assert torch.allclose(f(deepcopy(t)), cse_graph(deepcopy(t)))

    """
    Check that different gradient modes prevent elimination.
    """

    def test_with_grad(self):
        def f(x):
            with torch.no_grad():
                y = 2 * x
            z = 2 * x
            return y + z

        t = torch.randn(2, 2, requires_grad=True)
        t.requires_grad = True

        graph, _ = torch._dynamo.export(f)(t)

        f(t).sum().backward()
        eager_grad = t.grad.clone()

        t.grad = None
        cse_graph = P_default.call(graph).graph_module
        cse_graph(t).sum().backward()
        compile_grad = t.grad.clone()
        assert torch.allclose(eager_grad, compile_grad)

    def test_module(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10, bias=False)
                with torch.no_grad():
                    self.linear.weight.fill_(1)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # Perform the same operation twice.
                x = self.linear(inp)
                y = self.linear(inp)
                return (x * 200000) + (y * 3000000)

        t = torch.randn(1, 10)
        model = LinearModel()
        graph, _ = torch._dynamo.export(model)(t)

        model(t).sum().backward()
        eager_grad = model.linear.weight.grad.clone()

        model.zero_grad()
        cse_graph = P_default.call(graph).graph_module
        cse_graph(t).sum().backward()
        compile_grad = model.linear.weight.grad.clone()
        assert torch.allclose(eager_grad, compile_grad)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
