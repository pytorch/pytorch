# Owner(s): ["oncall: export"]
import copy
import unittest
from contextlib import nullcontext

import torch
from functorch.experimental import control_flow
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import _detect_fake_mode_from_gm, _get_shape_env_from_gm
from torch.export import export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import (
    fake_tensor_modes,
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
@instantiate_parametrized_tests
class TestPassInfra(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @fake_tensor_modes
    @parametrize("metadata_key", ["val", "example_value"])
    @parametrize("placeholder", [False, True])
    def test_fake_mode_from_graph(self, mode_cls, metadata_key, placeholder):
        mode = mode_cls(shape_env=ShapeEnv())
        fake = mode.from_tensor(torch.randn(3, 4))
        graph = torch.fx.Graph()
        node = (
            graph.placeholder("x")
            if placeholder
            else graph.call_function(torch.ops.aten.ones.default, ([3, 4],))
        )
        node.meta[metadata_key] = fake
        graph.output(node)
        gm = torch.fx.GraphModule({}, graph)
        self.assertIs(_detect_fake_mode_from_gm(gm), mode)
        self.assertIs(_get_shape_env_from_gm(gm), mode.shape_env)

    @fake_tensor_modes
    def test_extract_fake_inputs(self, mode_cls):
        from torch.export._trace import _extract_fake_inputs

        mode = mode_cls(shape_env=ShapeEnv())
        x = torch.randn(3, 4)
        with mode:
            gm = make_fx(lambda x, n: x + n, tracing_mode="symbolic")(x, 2)
        args, kwargs, detected_mode = _extract_fake_inputs(gm, (x,), {"n": 2})
        placeholders = list(gm.graph.find_nodes(op="placeholder"))
        self.assertIs(args[0], placeholders[0].meta["val"])
        self.assertIs(kwargs["n"], placeholders[1].meta["val"])
        self.assertIs(detected_mode, mode)

    @fake_tensor_modes
    def test_runtime_assertion_pass_fake_mode(self, mode_cls):
        from torch._export.utils import apply_runtime_assertion_pass
        from torch._functorch._aot_autograd.schemas import GraphSignature
        from torch.export._trace import _replace_unbacked_bindings
        from torch.utils import _pytree as pytree

        def f(x):
            result = x.nonzero()
            torch._check(result.shape[0] <= 2)
            return (result,)

        x = torch.tensor([1, 0, 1, 0])
        mode = mode_cls(shape_env=ShapeEnv())
        with mode:
            gm = make_fx(f, tracing_mode="symbolic")(x)
        _replace_unbacked_bindings(gm)
        nonzero = next(
            n for n in gm.graph.nodes if n.target == torch.ops.aten.nonzero.default
        )
        self.assertTrue(nonzero.meta["unbacked_bindings"])
        signature = GraphSignature(
            parameters=[],
            buffers=[],
            user_inputs=["x_1"],
            user_outputs=[nonzero.name],
            inputs_to_parameters={},
            inputs_to_buffers={},
            buffers_to_mutate={},
            parameters_to_mutate={},
            user_inputs_to_mutate={},
            in_spec=pytree.tree_structure(((x,), {})),
            out_spec=pytree.tree_structure((x,)),
            backward_signature=None,
            input_tokens=[],
            output_tokens=[],
        )
        gm, signature = apply_runtime_assertion_pass(gm, signature)
        self.assertEqual(gm(x)[0].shape, (2, 1))
        self.assertEqual(signature.user_outputs, [nonzero.name])
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            gm(torch.ones(4))

    @fake_tensor_modes
    @parametrize("ambient", [False, True])
    def test_export_pass_tensor_metadata(self, mode_cls, ambient):
        from torch._subclasses.fake_tensor import maybe_get_fake_mode
        from torch.fx.passes.shape_prop import _extract_tensor_metadata

        x = torch.randn(3, 4, requires_grad=True)
        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        node.meta["tensor_meta"] = _extract_tensor_metadata(x)
        graph.output(node)
        gm = torch.fx.GraphModule({}, graph)
        mode = mode_cls()
        export_pass = _ExportPassBaseDeprecatedDoNotUse()
        export_pass.fake_tensor_mode = mode
        with mode if ambient else nullcontext():
            (fake,) = export_pass.inputs(gm)
        self.assertIs(maybe_get_fake_mode(fake), mode)
        self.assertEqual(_extract_tensor_metadata(fake), node.meta["tensor_meta"])
        self.assertEqual(fake.device, x.device)

    @fake_tensor_modes
    def test_export_pass_fake_mode(self, mode_cls):
        from torch._subclasses.fake_tensor import maybe_get_fake_mode

        def f(x):
            return x.t(), x.sin()

        mode = mode_cls(shape_env=ShapeEnv())
        with mode:
            gm = make_fx(f, tracing_mode="symbolic")(torch.randn(3, 4))
        result = _ExportPassBaseDeprecatedDoNotUse()(gm).graph_module
        self.assertEqual(
            [(n.op, n.target) for n in gm.graph.nodes],
            [(n.op, n.target) for n in result.graph.nodes],
        )
        for before, after in zip(gm.graph.nodes, result.graph.nodes):
            if isinstance(val := after.meta.get("val"), torch.Tensor):
                self.assertIs(maybe_get_fake_mode(val), mode)
                self.assertEqual(val.shape, before.meta["val"].shape)
                self.assertEqual(val.stride(), before.meta["val"].stride())

    def test_export_pass_base(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            pass

        ep = export(f, (torch.ones(3, 2),), strict=True)
        old_nodes = ep.graph.nodes

        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - y

                def false_fn(x, y):
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        _ = export(mod, (torch.tensor(True), x, y), strict=True)._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )

    def test_node_name_stability(self) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # Define a parameter
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (torch.rand(1), torch.rand(1))
        m = CustomModule()

        ep_before = export(m, inps, strict=True)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()

        # Test the custom module with two input tensors
        input_tensor1 = torch.tensor(5.0)
        input_tensor2 = torch.tensor(6.0)

        ep_before = torch.export.export(
            my_module, (input_tensor1, input_tensor2), strict=True
        )
        from torch.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)

    def test_replace_hook_basic(self) -> None:
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()
        inputs = (torch.tensor(6.0), torch.tensor(7.0))
        ep_before = export(my_module, inputs, strict=True)

        def replace_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)


if __name__ == "__main__":
    run_tests()
