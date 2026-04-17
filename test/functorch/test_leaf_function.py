# Owner(s): ["oncall: pt2"]

"""Tests for @leaf_function with make_fx, aot_function, and torch.compile."""

import copy
from functools import partial
from unittest.mock import patch

import torch
import torch._dynamo.config as config
import torch._dynamo.testing
from functorch.compile import aot_function, nop
from torch._dynamo.decorators import leaf_function
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFx(TestCase):
    def _has_invoke_leaf_function_node(self, gm):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is invoke_leaf_function:
                return True
        return False

    def test_make_fx_simple(self):
        @leaf_function
        def my_fn(x, y):
            if x.sum() > 0:
                return (x + y,)
            return (x - y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x + y,)

        def f(x, y):
            return my_fn(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_with_closure(self):
        config = {"scale": 2.0}

        @leaf_function
        def scaled_fn(x):
            return (x * config["scale"],)

        @scaled_fn.register_fake
        def scaled_fn_fake(x):
            return (torch.empty_like(x),)

        def f(x):
            return scaled_fn(x)

        x = torch.randn(3, 3)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        # Closure change reflected at runtime
        config["scale"] = 3.0
        x2 = torch.randn(3, 3)
        self.assertEqual(gm(x2), f(x2))

    def test_make_fx_data_dependent(self):
        @leaf_function
        def data_dep_fn(x):
            nz = (x > 0).nonzero()
            return (x.relu(), nz)

        @data_dep_fn.register_fake
        def data_dep_fn_fake(x):
            return (x.relu(), (x > 0).nonzero())

        def f(x):
            return data_dep_fn(x)

        x = torch.randn(4, 4)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        x2 = torch.randn(4, 4)
        eager_out = f(x2)
        gm_out = gm(x2)
        self.assertEqual(gm_out[0], eager_out[0])
        self.assertEqual(gm_out[1], eager_out[1])

    def test_make_fx_pytree_inputs(self):
        @leaf_function
        def pytree_fn(inputs):
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            return pytree_fn({"x": x, "y": y})

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_no_fake_impl(self):
        @leaf_function
        def no_fake_fn(x):
            return (x * 2,)

        def f(x):
            return no_fake_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            make_fx(f)(x)

    def test_make_fx_with_module(self):
        @leaf_function
        def mod_fn(mod, x):
            return (mod(x),)

        @mod_fn.register_fake
        def mod_fn_fake(mod, x):
            return (mod(x),)

        linear = torch.nn.Linear(3, 3)

        def f(w, b, x):
            with torch.nn.utils.stateless._reparametrize_module(
                linear, {"weight": w, "bias": b}
            ):
                return mod_fn(linear, x)

        w = linear.weight.detach().clone()
        b = linear.bias.detach().clone()
        x = torch.randn(2, 3)
        gm = make_fx(f)(w, b, x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        w2 = torch.randn(3, 3)
        b2 = torch.randn(3)
        x2 = torch.randn(2, 3)
        self.assertEqual(gm(w2, b2, x2), f(w2, b2, x2))

    def test_make_fx_over_invoke_leaf_function_hop(self):
        """Tracing invoke_leaf_function HOP directly with make_fx should produce
        a single opaque invoke_leaf_function node, not trace into its internals."""
        import torch.utils._pytree as pytree
        from torch._higher_order_ops.invoke_leaf_function import _LeafCallable

        def real_fn(x, y):
            return (x * y + x,)

        def fake_fn(x, y):
            return (x * y + x,)

        real_fn_callable = _LeafCallable(real_fn)
        fake_fn_callable = _LeafCallable(fake_fn)

        def f(x, y):
            _, input_spec = pytree.tree_flatten(((x, y), {}))
            return invoke_leaf_function(
                real_fn_callable, fake_fn_callable, input_spec, "", x, y
            )

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_input_mutation(self):
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            return mutate_fn(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        buf_fx = buf.clone()
        gm = make_fx(f)(x, buf_fx)
        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        buf_gm = buf.clone()
        gm_out = gm(x, buf_gm)
        self.assertEqual(gm_out, eager_out)
        self.assertEqual(buf_gm, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionAotFunction(TestCase):
    def test_aot_function_simple(self):
        @leaf_function
        def my_fn(x, y):
            return (x @ y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x @ y,)

        def f(x, y):
            return my_fn(x, y)[0]

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        )

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        self.assertExpectedInline(
            normalize_gm(fw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_2, primals_3, requires_grad_indices = '0,1');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_2 = primals_3 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            normalize_gm(bw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_3: "f32[3, 3]" = with_effects_1[1]
        getitem_4: "f32[3, 3]" = with_effects_1[2];  with_effects_1 = None
        return (getitem_3, getitem_4, getitem_2)
""",
        )

    def test_aot_function_gradients(self):
        @leaf_function
        def my_fn(x, y):
            return (x * y + x,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (x * y + x,)

        def f(x, y):
            return my_fn(x, y)[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_closure(self):
        config = {"use_double": True}

        @leaf_function
        def closure_fn(x, y):
            if config["use_double"]:
                return (x @ y * 2,)
            return (x @ y * 3,)

        @closure_fn.register_fake
        def closure_fn_fake(x, y):
            return (x @ y,)

        def f(x, y):
            return closure_fn(x, y)[0]

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        x_val = torch.randn(3, 3)
        y_val = torch.randn(3, 3)

        # Test with use_double=True
        config["use_double"] = True
        x = x_val.clone().requires_grad_(True)
        y = y_val.clone().requires_grad_(True)
        x_clone = x_val.clone().requires_grad_(True)
        y_clone = y_val.clone().requires_grad_(True)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Change closure and verify result changes
        config["use_double"] = False
        x2 = x_val.clone().requires_grad_(True)
        y2 = y_val.clone().requires_grad_(True)
        x2_clone = x_val.clone().requires_grad_(True)
        y2_clone = y_val.clone().requires_grad_(True)

        out_eager2 = f(x2, y2)
        out_compiled2 = compiled_f(x2_clone, y2_clone)
        self.assertEqual(out_eager2, out_compiled2)

        out_eager2.sum().backward()
        out_compiled2.sum().backward()
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        # Different closures => different results
        self.assertNotEqual(out_eager, out_eager2)

    def test_aot_function_pytree_inputs(self):
        @leaf_function
        def pytree_fn(inputs):
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            return pytree_fn({"x": x, "y": y})[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_input_mutation(self):
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            return mutate_fn(x, buf)[0]

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        compiled_f = aot_function(f, nop)
        buf_compiled = buf.clone()
        compiled_out = compiled_f(x, buf_compiled)
        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(buf_compiled, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionEscapedGradients(TestCase):
    def test_aot_function_escaped_gradient_multiple_closures(self):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_multiple_closures(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                compiled_f(x)

    def test_aot_function_escaped_gradient_disabled(self):
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=False):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_input_no_grad(self):
        """No false positive when input doesn't require grad."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=False)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_actually_lost(self):
        """Verify gradients don't flow to closure tensors."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        result = compiled_f(x)
        result.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_make_fx_escaped_gradient_multiple_closures(self):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                make_fx(f)(x)

    def test_make_fx_escaped_gradient_disabled(self):
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=False):
            gm = make_fx(f)(x)
            x2 = torch.randn(2, 3)
            result = gm(x2)
            self.assertEqual(result[0].shape, (2, 3))


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFxAndCompile(TestCase):
    """Tests for @leaf_function when mixing torch.compile and make_fx."""

    def test_not_called_during_compilation(self):
        """The real leaf_fn body runs only at runtime, not during compilation."""
        torch._dynamo.reset()
        call_count = 0

        @leaf_function
        def my_leaf(x, y):
            nonlocal call_count
            call_count += 1
            return (x + y,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            return (torch.empty_like(x),)

        def f(x, y):
            return my_leaf(x, y)[0]

        compiled_f = torch.compile(f, backend="eager", fullgraph=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Compilation + first call
        result = compiled_f(x, y)
        self.assertEqual(call_count, 1)
        self.assertEqual(result, x + y)

        # Second call reuses compiled code, leaf_fn called again exactly once
        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        result2 = compiled_f(x2, y2)
        self.assertEqual(call_count, 2)
        self.assertEqual(result2, x2 + y2)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_compile(self):
        """Leaf function only inside the torch.compile'd region."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x, y):
            return (x * y + x,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            return (x * y + x,)

        def inner(x, y):
            return my_leaf(x, y)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x, y):
            z = x + 1
            return compiled(z, y) - 1

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(getitem_1, 1);  getitem_1 = None
        return sub

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, arg2_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = arg2_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        expected = (x2 + 1) * y2 + (x2 + 1) - 1
        self.assertEqual(traced(x2, y2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_make_fx(self):
        """Leaf function only in the make_fx-traced code, outside torch.compile."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            return (torch.empty_like(x),)

        def inner(x):
            return x + 1

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem = None
        getitem_1: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_1,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            return (add,)
""",
        )

        x2 = torch.randn(3, 3)
        expected = ((x2 + 1) * 2,)
        self.assertEqual(traced(x2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_in_both_compile_and_make_fx(self):
        """Same leaf function used inside torch.compile and in make_fx-traced code."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            return (torch.empty_like(x),)

        def inner(x):
            return my_leaf(x)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",
        )


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionDynamo(PytreeRegisteringTestCase):
    def _assert_models_equal(
        self,
        model_expected,
        model_test,
        x_expected,
        x_test,
    ):
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

        expected_grads = {
            name: param.grad for name, param in model_expected.named_parameters()
        }
        test_grads = {name: param.grad for name, param in model_test.named_parameters()}

        self.assertEqual(set(expected_grads.keys()), set(test_grads.keys()))
        for name in expected_grads:
            if expected_grads[name] is not None:
                self.assertEqual(
                    expected_grads[name],
                    test_grads[name],
                    msg=f"Gradient mismatch for parameter {name}",
                )

    def _test_leaf_function_helper(self, mod_class, args_fn, loss_fn):
        import torch.utils._pytree as pytree
        from torch._dynamo.testing import AotEagerAndRecordGraphs, EagerAndRecordGraphs

        mod_eager = mod_class()
        mod_compile_eager = mod_class()
        mod_compile_eager.load_state_dict(dict(mod_eager.state_dict()))
        mod_compile_aot = mod_class()
        mod_compile_aot.load_state_dict(dict(mod_eager.state_dict()))

        eager_backend = EagerAndRecordGraphs()
        compiled_eager = torch.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )

        backend = AotEagerAndRecordGraphs()
        compiled_aot = torch.compile(mod_compile_aot, backend=backend, fullgraph=True)

        for _ in range(2):
            mod_eager.zero_grad()
            mod_compile_eager.zero_grad()
            mod_compile_aot.zero_grad()

            args = args_fn()
            args_clone = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )
            args_clone2 = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )

            out_eager = mod_eager(*args)
            loss_fn(out_eager).backward()

            out_compile_eager = compiled_eager(*args_clone)
            loss_fn(out_compile_eager).backward()

            out_compile_aot = compiled_aot(*args_clone2)
            loss_fn(out_compile_aot).backward()

            self.assertEqual(out_eager, out_compile_eager)
            self.assertEqual(out_eager, out_compile_aot)

            for (name_eager, param_eager), (_, param_compile_eager), (
                _,
                param_compile_aot,
            ) in zip(
                mod_eager.named_parameters(),
                mod_compile_eager.named_parameters(),
                mod_compile_aot.named_parameters(),
            ):
                self.assertEqual(
                    param_eager.grad,
                    param_compile_eager.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_eager",
                )
                self.assertEqual(
                    param_eager.grad,
                    param_compile_aot.grad,
                    msg=f"Gradient mismatch for {name_eager} between eager and compile_aot",
                )

            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone,
            )
            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone2,
            )

        return (
            normalize_gm(eager_backend.graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
        )

    def test_leaf_function_simple(self):
        @leaf_function
        def non_tracable_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @non_tracable_forward.register_fake
        def non_tracable_forward_fake(mod, x):
            return (mod.linear(x),)

        class NonTracable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return non_tracable_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            NonTracable, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_2, requires_grad_indices = '1,2,3');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4];  with_effects_1 = None
        return (getitem_6, getitem_4, getitem_5, getitem_2)
""",
        )

    def test_leaf_function_with_logging(self):
        @leaf_function
        def logging_forward(mod, x):
            print("Processing input")
            return (mod.linear(x),)

        @logging_forward.register_fake
        def logging_forward_fake(mod, x):
            return (mod.linear(x),)

        class LoggingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return logging_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            self.assertEqual(mock_print.call_count, 6)

    def test_leaf_function_dynamic_autograd_module_config(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        @leaf_function
        def configurable_scale(mod, x):
            if mod.use_double_scale:
                return (mod.linear(x) * 2,)
            else:
                return (mod.linear(x) * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(mod, x):
            return (mod.linear(x),)

        class ConfigurableModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.use_double_scale = True

            def forward(self, x):
                return configurable_scale(self, x)

        mod_eager = ConfigurableModule()
        mod_compiled = ConfigurableModule()
        mod_compiled.load_state_dict(dict(mod_eager.state_dict()))

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(mod_compiled, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)

        mod_eager.use_double_scale = True
        mod_compiled.use_double_scale = True

        x1 = x_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)

        out_eager_1 = mod_eager(x1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)

        mod_eager.zero_grad()
        mod_compiled.zero_grad()

        mod_eager.use_double_scale = False
        mod_compiled.use_double_scale = False

        x2 = x_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)

        out_eager_2 = mod_eager(x2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)

        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_dynamic_autograd_closure(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        closure_config = {"use_double_scale": True}

        @leaf_function
        def configurable_scale(x, y):
            if closure_config["use_double_scale"]:
                return (x @ y * 2,)
            else:
                return (x @ y * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(x, y):
            return (x @ y,)

        def fn(x, y):
            return configurable_scale(x, y)

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)
        y_value = torch.randn(3, 3)

        closure_config["use_double_scale"] = True

        x1 = x_value.clone().requires_grad_(True)
        y1 = y_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)
        y1_clone = y_value.clone().requires_grad_(True)

        out_eager_1 = fn(x1, y1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone, y1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)
        self.assertEqual(y1.grad, y1_clone.grad)

        closure_config["use_double_scale"] = False

        x2 = x_value.clone().requires_grad_(True)
        y2 = y_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)
        y2_clone = y_value.clone().requires_grad_(True)

        out_eager_2 = fn(x2, y2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone, y2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)
        self.assertNotEqual(y1.grad, y2.grad)

        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_closure_constants_without_grad(self):
        closure_scale = 2.0
        closure_tensor = torch.tensor([1.0, 2.0, 3.0])

        @leaf_function
        def closure_forward(mod, x):
            out = mod.linear(x) * closure_scale * mod.scale
            out = out + closure_tensor + mod.offset
            return (out,)

        @closure_forward.register_fake
        def closure_forward_fake(mod, x):
            return (mod.linear(x) + mod.offset,)

        class ClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return closure_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            ClosureModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_parameters_offset_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_parameters_offset_ = L_self_parameters_offset_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_parameters_offset_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_parameters_offset_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_5, primals_2, requires_grad_indices = '1,2,3,4');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_5 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3]" = with_effects_1[2]
        getitem_5: "f32[3, 3]" = with_effects_1[3]
        getitem_6: "f32[3]" = with_effects_1[4]
        getitem_7: "f32[3, 3]" = with_effects_1[5];  with_effects_1 = None
        return (getitem_7, getitem_4, getitem_5, getitem_6, getitem_2)
""",
        )

    def test_leaf_function_pytree_inputs(self):
        @leaf_function
        def pytree_forward(mod, inputs):
            if inputs["x"].sum() > 0:
                return (mod.linear(inputs["x"]), inputs["y"] + 1)
            return (mod.linear(inputs["x"]) + inputs["y"], inputs["y"] - 1)

        @pytree_forward.register_fake
        def pytree_forward_fake(mod, inputs):
            return (mod.linear(inputs["x"]), inputs["y"])

        class PytreeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, inputs):
                return pytree_forward(self, inputs)

        def args_fn():
            return (
                {
                    "x": torch.randn(3, 3, requires_grad=True),
                    "y": torch.randn(3, 3, requires_grad=True),
                },
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PytreeModule, args_fn, loss_fn)

    def test_leaf_function_nested_annotations(self):
        @leaf_function
        def inner_leaf_forward(mod, x):
            y = mod.linear(x)
            return (y + x,)

        @inner_leaf_forward.register_fake
        def inner_leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class InnerLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return inner_leaf_forward(self, x)

        @leaf_function
        def outer_leaf_forward(mod, x):
            z = mod.linear(x)
            return mod.inner(z + x)

        @outer_leaf_forward.register_fake
        def outer_leaf_forward_fake(mod, x):
            return mod.inner(mod.linear(x))

        class OuterLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerLeaf()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return outer_leaf_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            OuterLeaf, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_inner_modules_linear_parameters_weight_ = L_self_modules_inner_modules_linear_parameters_weight_
        l_self_modules_inner_modules_linear_parameters_bias_ = L_self_modules_inner_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_inner_modules_linear_parameters_weight_, l_self_modules_inner_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_inner_modules_linear_parameters_weight_ = l_self_modules_inner_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]", primals_5: "f32[3, 3]", primals_6: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_5, primals_6, primals_2, requires_grad_indices = '1,2,3,4,5');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_5 = primals_6 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4]
        getitem_7: "f32[3]" = with_effects_1[5]
        getitem_8: "f32[3, 3]" = with_effects_1[6];  with_effects_1 = None
        return (getitem_8, getitem_4, getitem_5, getitem_6, getitem_7, getitem_2)
""",
        )

    def test_leaf_function_data_dependent_nonzero(self):
        @leaf_function
        def nonzero_forward(mod, x):
            out = mod.linear(x)
            nonzero_indices = (out > 0).nonzero()
            return (out, nonzero_indices)

        @nonzero_forward.register_fake
        def nonzero_forward_fake(mod, x):
            out = mod.linear(x)
            return out, (out > 0).nonzero()

        class NonzeroModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return nonzero_forward(self, x)

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = torch.nn.Linear(3, 3)
                self.nonzero_module = NonzeroModule()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                x = self.pre_linear(x)
                x = torch.relu(x)
                out, nonzero_indices = self.nonzero_module(x)
                num_nonzero = nonzero_indices.shape[0]
                scaled_out = out * self.scale + num_nonzero
                return scaled_out, nonzero_indices

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(OuterModule, args_fn, loss_fn)

    @skipIfCrossRef
    def test_leaf_function_data_dependent_item(self):
        @leaf_function
        def item_forward(mod, x):
            out = mod.linear(x)
            scalar_value = out.sum().item()
            return (out, scalar_value)

        @item_forward.register_fake
        def item_forward_fake(mod, x):
            out = mod.linear(x)
            return (out, out.sum().item())

        class ItemModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return item_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(ItemModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_multiple_compiled_submodules(self, backend):
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

        class CompiledSubmodule1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = torch.nn.Linear(4, 4)
                self.leaf = LeafModule(4, 4)

            def forward(self, x):
                x = self.pre_linear(x)
                x = torch.relu(x)
                out = self.leaf(x)[0]
                return out

        class CompiledSubmodule2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule(4, 4)
                self.post_linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                out = self.leaf(x)[0]
                out = self.post_linear(out)
                return torch.sigmoid(out)

        class CompiledSubmodule3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)

            def forward(self, x):
                out1 = self.leaf1(x)[0]
                out2 = self.leaf2(x)[0]
                return out1 + out2

        class TopLevelModule(torch.nn.Module):
            def __init__(self, compile_submodules=False):
                super().__init__()
                self.submodule1 = CompiledSubmodule1()
                self.submodule2 = CompiledSubmodule2()
                self.submodule3 = CompiledSubmodule3()
                self.final_linear = torch.nn.Linear(4, 4)
                self.compile_submodules = compile_submodules

            def forward(self, x):
                if self.compile_submodules:
                    out1 = torch.compile(self.submodule1, backend=backend)(x)
                    out2 = torch.compile(self.submodule2, backend=backend)(out1)
                    out3 = torch.compile(self.submodule3, backend=backend)(out2)
                else:
                    out1 = self.submodule1(x)
                    out2 = self.submodule2(out1)
                    out3 = self.submodule3(out2)
                final = self.final_linear(out3)
                return final

        model_eager = TopLevelModule(compile_submodules=False)
        model_compiled = TopLevelModule(compile_submodules=True)
        model_compiled.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_compiled = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(
            model_eager,
            model_compiled,
            x,
            x_compiled,
        )

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("do_compile", [False, True])
    def test_leaf_function_with_graph_breaks(self, backend, do_compile):
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + 1,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

        class TopLevelModule(torch.nn.Module):
            def __init__(self, do_compile=False, backend="eager"):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)
                self.leaf3 = LeafModule(4, 4)
                self.final_linear = torch.nn.Linear(4, 4)
                self.do_compile = do_compile
                self.backend = backend

            def _forward(self, x):
                out1 = self.leaf1(x)[0]
                torch._dynamo.graph_break()
                out2 = self.leaf2(out1)[0]
                torch._dynamo.graph_break()
                out3 = self.leaf3(out2)[0]
                result = self.final_linear(out3)
                return result

            def forward(self, x):
                if self.do_compile:
                    return torch.compile(
                        self._forward, backend=self.backend, fullgraph=False
                    )(x)
                else:
                    return self._forward(x)

        model_eager = TopLevelModule(do_compile=False)
        model_test = TopLevelModule(do_compile=do_compile, backend=backend)
        model_test.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_test = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(model_eager, model_test, x, x_test)

    def test_leaf_function_with_module_in_pytree(self):
        @leaf_function
        def main_forward(modules_dict, x):
            if x.sum() > 0:
                return (modules_dict["first"](x) + modules_dict["second"](x),)
            else:
                return (modules_dict["first"](x) - modules_dict["second"](x),)

        @main_forward.register_fake
        def main_forward_fake(modules_dict, x):
            return (modules_dict["first"](x) + modules_dict["second"](x),)

        class HelperModule(torch.nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = scale

            def forward(self, x):
                return self.linear(x) * self.scale

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper1 = HelperModule(scale=1.0)
                self.helper2 = HelperModule(scale=0.5)

            def forward(self, x):
                modules_dict = {"first": self.helper1, "second": self.helper2}
                return main_forward(modules_dict, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_as_kwarg(self):
        @leaf_function
        def main_forward(x, helper_mod=None):
            if x.sum() > 0:
                return (helper_mod(x),)
            else:
                return (helper_mod(x) + x,)

        @main_forward.register_fake
        def main_forward_fake(x, helper_mod=None):
            return (helper_mod(x),)

        class HelperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper = HelperModule()

            def forward(self, x):
                return main_forward(x, helper_mod=self.helper)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_missing_fake_impl_error(self):
        @leaf_function
        def no_fake_impl_forward(mod, x):
            return (mod.linear(x),)

        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return no_fake_impl_forward(self, x)

        mod = SimpleModule()
        x = torch.randn(3, 3)

        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            mod(x)

        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_constant_tensor_closure_error(self, backend):
        constant_weight = torch.randn(3, 3)

        @leaf_function
        def constant_closure_forward(x):
            return (x @ constant_weight,)

        @constant_closure_forward.register_fake
        def constant_closure_forward_fake(x):
            return (x @ constant_weight,)

        class ConstantClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return constant_closure_forward(x)

        mod = ConstantClosureModule()
        x = torch.randn(3, 3, requires_grad=True)

        result = mod(x)
        expected = x @ constant_weight
        self.assertEqual(result[0], expected)

        compiled_mod = torch.compile(mod, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_error(self, backend):
        @leaf_function
        def mutate_input(x):
            x.add_(1)
            return (x,)

        @mutate_input.register_fake
        def mutate_input_fake(x):
            x.add_(1)
            return (x,)

        def fn(x):
            return mutate_input(x)

        x = torch.randn(3, 3)

        x_eager = x.clone()
        with self.assertRaisesRegex(RuntimeError, "Undeclared in-place mutation"):
            fn(x_eager)

        x = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
            compiled_fn(x.clone().requires_grad_(True))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_validation_dtype_mismatch(self, backend):
        @leaf_function
        def dtype_mismatch_forward(mod, x):
            return (mod.linear(x),)

        @dtype_mismatch_forward.register_fake
        def dtype_mismatch_forward_fake(mod, x):
            return (mod.linear(x).double(),)

        class DtypeMismatchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return dtype_mismatch_forward(self, x)

        mod = DtypeMismatchModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=True):
            compiled_mod = torch.compile(mod, backend=backend)
            with self.assertRaisesRegex(RuntimeError, "Dtype mismatch"):
                compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("validate_outputs", [True, False])
    def test_leaf_function_validation_shape_mismatch(self, backend, validate_outputs):
        @leaf_function
        def mismatched_forward(mod, x):
            return (mod.linear(x),)

        @mismatched_forward.register_fake
        def mismatched_forward_fake(mod, x):
            return (torch.zeros(x.shape[0], 6),)

        class MismatchedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return mismatched_forward(self, x)

        mod = MismatchedModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=validate_outputs):
            compiled_mod = torch.compile(mod, backend=backend)
            if validate_outputs:
                with self.assertRaises((RuntimeError, AssertionError)):
                    compiled_mod(x)
            else:
                result = compiled_mod(x)
                self.assertEqual(result[0].shape, (3, 3))

    def test_leaf_function_no_module_inputs(self):
        @leaf_function
        def my_custom_fn(inputs: dict[str, torch.Tensor], scale: float, offset: int):
            x = inputs["x"]
            y = inputs["y"]
            if x.sum() > 0:
                return (x * scale + y + offset, x.sum() + y.sum())
            return (x * scale - y + offset, x.sum() - y.sum())

        @my_custom_fn.register_fake
        def my_custom_fn_fake(
            inputs: dict[str, torch.Tensor], scale: float, offset: int
        ):
            x = inputs["x"]
            y = inputs["y"]
            return (x * scale + y + offset, x.sum() + y.sum())

        class NoModuleInputsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2.0
                self.offset = 1

            def forward(self, x, y):
                inputs = {"x": x, "y": y}
                return my_custom_fn(inputs, self.scale, self.offset)

        def args_fn():
            return (
                torch.randn(3, 3, requires_grad=True),
                torch.randn(3, 3, requires_grad=True),
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(NoModuleInputsModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_multiple_tensors(
        self, backend, check_escaped_gradients
    ):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x)
            else:
                result = compiled_fn(x)
                self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_input_no_grad(
        self, backend, check_escaped_gradients
    ):
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            result = compiled_fn(x)
            self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_mixed_inputs(
        self, backend, check_escaped_gradients
    ):
        base1 = torch.randn(3, 3, requires_grad=True)
        base2 = torch.randn(3, 4, requires_grad=True)
        closure_weight1 = base1 * 2
        closure_weight2 = base2 * 3

        @leaf_function
        def mixed_inputs(x, y):
            out1 = x @ closure_weight1 + y
            out2 = x @ closure_weight2
            return (out1, out2)

        @mixed_inputs.register_fake
        def mixed_inputs_fake(x, y):
            return (torch.empty(x.shape[0], 3), torch.empty(x.shape[0], 4))

        def fn(x, y):
            return mixed_inputs(x, y)

        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x, y)
            else:
                result = compiled_fn(x, y)
                self.assertEqual(result[0].shape, (2, 3))
                self.assertEqual(result[1].shape, (2, 4))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_error_message_contains_tensor_info(
        self, backend
    ):
        closure_weight = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 5),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 4, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, r"shape=\[4, 5\].*dtype="):
                compiled_fn(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_actually_lost(self, backend):
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        loss = result[0].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_leaf_function_and_nonstrict_trace_mutually_exclusive(self):
        from torch._dynamo.decorators import leaf_function, nonstrict_trace

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @leaf_function
            @nonstrict_trace
            def bad_fn1(x):
                return (x,)

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @nonstrict_trace
            @leaf_function
            def bad_fn2(x):
                return (x,)

    @skipIfCrossRef
    def test_leaf_function_no_return_value(self):
        printed = []

        @leaf_function
        def fn_no_return(x):
            print("processing")

        @fn_no_return.register_fake
        def fn_no_return_fake(x):
            pass

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                fn_no_return(x)
                return (self.linear(x),)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print", lambda *args, **kwargs: printed.append(args)):
            eager_graph, fw_graph, bw_graph = self._test_leaf_function_helper(
                Mod, args_fn, loss_fn
            )
        self.assertTrue(any("processing" in p for p in printed))
        self.assertExpectedInline(
            eager_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', l_x_);  real_fn = fake_fn = input_spec = invoke_leaf_function = None

        linear: "f32[3, 3]" = torch._C._nn.linear(l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  l_x_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        return (linear,)
""",
        )
        self.assertExpectedInline(
            fw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_2, requires_grad_indices = '0');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None

        t: "f32[3, 3]" = torch.ops.aten.t.default(primals_3)
        addmm: "f32[3, 3]" = torch.ops.aten.addmm.default(primals_4, primals_2, t);  primals_4 = t = None
        return (getitem, addmm, primals_2, primals_3)
""",
        )
        self.assertExpectedInline(
            bw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        t: "f32[3, 3]" = torch.ops.aten.t.default(primals_3);  primals_3 = None
        t_1: "f32[3, 3]" = torch.ops.aten.t.default(t);  t = None
        mm: "f32[3, 3]" = torch.ops.aten.mm.default(tangents_1, t_1);  t_1 = None
        t_2: "f32[3, 3]" = torch.ops.aten.t.default(tangents_1)
        mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t_2, primals_2);  t_2 = primals_2 = None
        t_3: "f32[3, 3]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
        sum_1: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[3]" = torch.ops.aten.view.default(sum_1, [3]);  sum_1 = None
        t_4: "f32[3, 3]" = torch.ops.aten.t.default(t_3);  t_3 = None
        return (mm, t_4, view)
""",
        )

    def test_leaf_function_output_structure_mismatch(self):
        @leaf_function
        def mismatched_fn(x):
            return {"a": x, "b": x * 2}

        @mismatched_fn.register_fake
        def mismatched_fn_fake(x):
            return (x, x * 2)

        def fn(x):
            return mismatched_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(AssertionError, "output structure mismatch"):
            torch.compile(fn, backend="eager")(x)

    def test_leaf_function_nested_output(self):
        @leaf_function
        def nested_output_fn(linear1, linear2, linear3, x):
            if x.sum() > 0:
                return {
                    "out": (linear1(x), linear2(x)),
                    "extra": linear3(x),
                    "count": 42,
                }
            else:
                return {
                    "out": (linear1(x) + 1, linear2(x) + 1),
                    "extra": linear3(x) + 1,
                    "count": 42,
                }

        @nested_output_fn.register_fake
        def nested_output_fn_fake(linear1, linear2, linear3, x):
            return {
                "out": (linear1(x), linear2(x)),
                "extra": linear3(x),
                "count": 42,
            }

        class NestedOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                result = nested_output_fn(self.linear1, self.linear2, self.linear3, x)
                return (
                    result["out"][0] * result["count"]
                    + result["out"][1]
                    + result["extra"]
                )

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out.sum()

        self._test_leaf_function_helper(NestedOutputModule, args_fn, loss_fn)

    def test_leaf_function_custom_pytree_output(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @leaf_function
        def point_fn(linear1, linear2, x):
            return (Point(linear1(x), linear2(x)), 0.5)

        @point_fn.register_fake
        def point_fn_fake(linear1, linear2, x):
            return (Point(linear1(x), linear2(x)), 0.5)

        class PointModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, x):
                p, scale = point_fn(self.linear1, self.linear2, x)
                return (p.x * scale, p.y * scale)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PointModule, args_fn, loss_fn)

    def test_leaf_function_fake_requires_grad_ignored(self):
        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x).requires_grad_(False),)

        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return my_fn(x)

        x = torch.randn(3, 3, requires_grad=True)
        out = fn(x)

        self.assertTrue(out[0].requires_grad)
        out[0].sum().backward()
        self.assertIsNotNone(x.grad)

        graph = backend.graphs[0]
        for node in graph.graph.nodes:
            if node.op == "call_function" and "invoke_leaf_function" in str(
                node.target
            ):
                example_value = node.meta.get("example_value")
                self.assertIsNotNone(example_value)
                self.assertTrue(example_value[0].requires_grad)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_non_grad(self, backend):
        @leaf_function(mutates_args={"buf"})
        def mutate_buffer(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_buffer.register_fake
        def mutate_buffer_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def fn(x, buf):
            return mutate_buffer(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x + buf + 1
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_mixed(self, backend):
        @leaf_function(mutates_args={"buf"})
        def mixed_fn(x, buf):
            buf.mul_(2)
            return (x * buf,)

        @mixed_fn.register_fake
        def mixed_fn_fake(x, buf):
            buf.mul_(2)
            return (x * buf,)

        def fn(x, buf):
            return mixed_fn(x, buf)

        x = torch.randn(3, 3, requires_grad=True)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x * (buf * 2)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf * 2)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf * 2)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_module_buffer(self, backend):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return update_stats(self, x)

        @leaf_function(mutates_args={"model.running_mean"})
        def update_stats(model, x):
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        @update_stats.register_fake
        def update_stats_fake(model, x):
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        mod = MyModule()
        x = torch.randn(4, 3)

        mod_eager = copy.deepcopy(mod)
        result_eager = mod_eager(x)
        expected_mean = torch.zeros(3) + x.mean(dim=0)
        self.assertEqual(mod_eager.running_mean, expected_mean)

        mod_compiled = copy.deepcopy(mod)
        compiled_mod = torch.compile(mod_compiled, backend=backend, fullgraph=True)
        result_compiled = compiled_mod(x)
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(mod_compiled.running_mean, expected_mean)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree(self, backend):
        @leaf_function(mutates_args={"buffers"})
        def update_buffers(x, buffers):
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        @update_buffers.register_fake
        def update_buffers_fake(x, buffers):
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        def fn(x, buffers):
            return update_buffers(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + (bufs[1] + 1)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1] + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1] + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree_fine_grained(self, backend):
        @leaf_function(mutates_args={"buffers[0]"})
        def update_first(x, buffers):
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        @update_first.register_fake
        def update_first_fake(x, buffers):
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        def fn(x, buffers):
            return update_first(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + bufs[1]
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1])

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1])

    def test_leaf_function_mutates_args_invalid_parameter(self):
        with self.assertRaisesRegex(ValueError, "refers to parameter 'buf'"):

            @leaf_function(mutates_args={"buf"})
            def bad_fn(x, buffers):
                buffers.add_(1)
                return (x + buffers,)

        with self.assertRaisesRegex(ValueError, "refers to parameter 'mdl'"):

            @leaf_function(mutates_args={"mdl.running_mean"})
            def bad_fn2(x, model):
                model.running_mean.add_(1)
                return (x,)

    def test_leaf_function_mutates_args_non_leaf_expression(self):
        @leaf_function(mutates_args={"model"})
        def bad_fn(x, model):
            model.running_mean.add_(1)
            return (x,)

        @bad_fn.register_fake
        def bad_fn_fake(x, model):
            model.running_mean.add_(1)
            return (x,)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))

            def forward(self, x):
                return bad_fn(x, self)

        mod = MyModule()
        x = torch.randn(3)
        compiled_fn = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, "resolved to a non-leaf value"
        ):
            compiled_fn(x)


instantiate_parametrized_tests(TestLeafFunctionDynamo)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionRegisterHook(TestCase):
    """Tests for @leaf_function's register_multi_grad_hook API."""

    def test_hook_fires_on_backward(self):
        hook_grads = []

        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_grads.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()

        self.assertEqual(len(hook_grads), 1)
        self.assertEqual(hook_grads[0], torch.full((3,), 2.0))

    def test_hook_with_non_tensor_args(self):
        hook_grads = []

        @leaf_function
        def my_fn(x, tag, scale):
            return (x * scale,)

        @my_fn.register_fake
        def my_fn_fake(x, tag, scale):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_grads.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x, "hello", 5.0)[0]
        out.sum().backward()

        self.assertEqual(len(hook_grads), 1)
        self.assertEqual(hook_grads[0], torch.full((3,), 5.0))

    def test_hook_multiple_tensor_inputs(self):
        hook_calls = []

        @leaf_function
        def my_fn(x, y):
            return (x * 2 + y * 3,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad, y_grad):
            hook_calls.append((x_grad.clone(), y_grad.clone()))

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        out = my_fn(x, y)[0]
        out.sum().backward()

        self.assertEqual(len(hook_calls), 1)
        self.assertEqual(hook_calls[0][0], torch.full((3,), 2.0))
        self.assertEqual(hook_calls[0][1], torch.full((3,), 3.0))

    def test_hook_only_fires_for_requires_grad_inputs(self):
        hook_calls = []

        @leaf_function
        def my_fn(x, y):
            return (x * 5 + y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_calls.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=False)
        out = my_fn(x, y)[0]
        out.sum().backward()

        self.assertEqual(len(hook_calls), 1)
        self.assertEqual(hook_calls[0], torch.full((3,), 5.0))

    def test_hook_no_requires_grad_no_fire(self):
        hook_count = [0]

        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_count[0] += 1

        x = torch.randn(3, requires_grad=False)
        my_fn(x)[0]
        self.assertEqual(hook_count[0], 0)

    def test_hook_side_effect_only_fn(self):
        fwd_called = [False]
        hook_grads = []

        @leaf_function
        def log_fn(x, tag):
            fwd_called[0] = True
            return None

        @log_fn.register_fake
        def log_fn_fake(x, tag):
            return None

        @log_fn.register_multi_grad_hook
        def log_fn_hook(x_grad):
            hook_grads.append(x_grad.clone())

        x = torch.randn(4, requires_grad=True)
        y = x * 2
        log_fn(y, "test")
        y.sum().backward()

        self.assertTrue(fwd_called[0])
        self.assertEqual(len(hook_grads), 1)

    def test_hook_gradient_values_correct(self):
        hook_grads = []

        @leaf_function
        def my_fn(x):
            return (x**2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_grads.append(x_grad.clone())

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()

        self.assertEqual(hook_grads[0], torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(x.grad, torch.tensor([2.0, 4.0, 6.0]))

    def test_hook_with_downstream_computation(self):
        hook_grads = []

        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_grads.append(x_grad.clone())

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = my_fn(x)[0]
        z = y * 3
        z.sum().backward()

        self.assertEqual(hook_grads[0], torch.tensor([6.0, 6.0]))

    def test_hook_with_retain_graph(self):
        hook_count = [0]

        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            hook_count[0] += 1

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()
        self.assertEqual(hook_count[0], 1)


if __name__ == "__main__":
    run_tests()
