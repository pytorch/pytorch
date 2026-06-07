# Owner(s): ["module: functorch"]

import torch
from functorch import make_fx
from functorch.compile import minifier
from torch._functorch.compile_utils import get_outputs, get_placeholders
from torch._functorch.fx_minifier import is_uninitialized_tensor_factory_node
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMinifier(TestCase):
    def _only_output_node(self, fx_g):
        output_node = next(n for n in fx_g.graph.nodes if n.op == "output")
        output = output_node.args[0]
        if isinstance(output, (list, tuple)) and len(output) == 1:
            output = output[0]
        return output

    def _assert_minifier_skips_uninitialized_output(
        self, failing_f, inps, uninitialized_targets
    ):
        found_uninitialized_target = False
        for node in failing_f.graph.nodes:
            if node.target in uninitialized_targets:
                found_uninitialized_target = True
                self.assertTrue(is_uninitialized_tensor_factory_node(node))
        self.assertTrue(found_uninitialized_target)

        def module_fails(fx_g, inps):
            output = self._only_output_node(fx_g)
            if (
                isinstance(output, torch.fx.Node)
                and output.target in uninitialized_targets
            ):
                return True
            return torch.ops.aten.relu.default in (n.target for n in fx_g.graph.nodes)

        min_f, inps = minifier(
            failing_f,
            inps,
            module_fails,
            max_granularity=1,
            skip_output_node=is_uninitialized_tensor_factory_node,
        )

        output = self._only_output_node(min_f)
        self.assertFalse(
            isinstance(output, torch.fx.Node) and output.target in uninitialized_targets
        )
        self.assertTrue(
            torch.ops.aten.relu.default in (n.target for n in min_f.graph.nodes)
        )
        self.assertEqual(len(min_f.graph.nodes), 3)
        self.assertEqual(len(inps), 1)

    def test_has_mul_minifier(self):
        def failing_f(x, y):
            y = y / 3
            x = x + 3
            x = x * y
            return x + y

        inps = [torch.randn(3), torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_mul(fx_g, inps):
            return torch.ops.aten.mul.Tensor in (i.target for i in fx_g.graph.nodes)

        min_f, inps = minifier(failing_f, inps, has_mul)
        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_has_add_mul(self):
        def failing_f(x):
            x = x * 3
            x = x + 5
            x = x.cos()
            zero = x - x
            result = zero / zero
            result = result + 3
            return (result * 2,)

        inps = [torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_nans(fx_g, inps):
            # Basically, make sure none of the nodes are computing nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, has_nans)
        self.assertEqual(len(min_f.graph.nodes), 3)
        self.assertEqual(len(inps), 1)

    def test_input_returned(self):
        def f(a, b, c):
            a = a.sin()
            c = c.cos()
            d = a * c
            return (a, b, c, d)

        inps = [torch.randn(3) for _ in range(3)]

        def inputs_returned(fx_g, inps):
            inps = set(get_placeholders(fx_g.graph))
            outs = set(get_outputs(fx_g.graph))
            return len(inps & outs) > 0

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, inputs_returned)
        self.assertEqual(len(min_f.graph.nodes), 2)
        self.assertEqual(len(inps), 1)

    def test_skip_uninitialized_suffix_outputs(self):
        def f(x):
            y = x.new_empty((10,))
            y.fill_(0)
            r = torch.relu(x)
            return (y, r)

        inps = [torch.randn(3)]
        failing_f = make_fx(f)(*inps)

        self._assert_minifier_skips_uninitialized_output(
            failing_f, inps, {torch.ops.aten.new_empty.default}
        )

    def test_skip_prims_empty_strided_suffix_outputs(self):
        def empty_strided_decomp(
            size, stride, *, dtype, device, layout=None, pin_memory=False
        ):
            return torch.ops.prims.empty_strided.default(
                size, stride, dtype=dtype, device=device, requires_grad=False
            )

        def f(x):
            y = torch.empty_strided((10,), (1,), dtype=x.dtype, device=x.device)
            y.fill_(0)
            r = torch.relu(x)
            return (y, r)

        inps = [torch.randn(3)]
        failing_f = make_fx(
            f,
            decomposition_table={
                torch.ops.aten.empty_strided.default: empty_strided_decomp
            },
        )(*inps)

        self._assert_minifier_skips_uninitialized_output(
            failing_f, inps, {torch.ops.prims.empty_strided.default}
        )

    def test_tup_use(self):
        def f(a, b):
            tup = torch.std_mean(a)
            return (tup[0] + b * tup[1],)

        inps = [torch.randn(3), torch.randn(3)]

        def has_add(fx_g, inps):
            return torch.ops.aten.add.Tensor in (i.target for i in fx_g.graph.nodes)

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, has_add)

        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.relu(x)
                zero = y - y
                result = zero / zero
                result = result + 3
                return result

        mod = MockModule()
        failing_f = torch.fx.symbolic_trace(mod)

        inps = [torch.randn(3)]

        def pass_checker(fx_g, inps):
            # Basically, make sure none of the inputs are nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, pass_checker)
        if len(min_f.graph.nodes) != 3:
            raise AssertionError(
                f"Expected 3 graph nodes, got {len(min_f.graph.nodes)}"
            )
        if len(inps) != 1:
            raise AssertionError(f"Expected 1 input, got {len(inps)}")


if __name__ == "__main__":
    run_tests()
