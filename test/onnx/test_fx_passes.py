# Owner(s): ["module: onnx"]
import pytorch_test_common

import torch
import torch._dynamo
import torch.fx

from torch.onnx._internal.fx.passes import _utils as pass_utils
from torch.testing._internal import common_utils


class TestFxPasses(common_utils.TestCase):
    def test_set_node_name_correctly_renames_when_new_name_collides_recursively(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func)(x, y, z)
        torch._dynamo.reset()

        # Purposely name the nodes in a way that will cause a recursive collision later.
        # See :func:`set_node_name` for name collision renaming logic.
        base_name = "tensor"
        nodes = list(gm.graph.nodes)
        for i, node in enumerate(nodes[1:]):
            if i == 0:
                node.name = base_name
            else:
                node.name = f"{base_name}.{i}"

        # Run `set_node_name` and verify that the names are correct.
        name_to_node = {node.name: node for node in gm.graph.nodes}
        pass_utils.set_node_name(nodes[0], base_name, name_to_node)
        assert nodes[0].name == base_name, f"Expected {base_name}, got {nodes[0].name}"
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"

    def test_set_node_name_succeeds_when_no_name_collisions(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func)(x, y, z)
        torch._dynamo.reset()

        # Run `set_node_name` and verify that the names are correct.
        new_name = "some_tensor"
        nodes = list(gm.graph.nodes)
        name_to_node = {node.name: node for node in nodes}
        pass_utils.set_node_name(nodes[1], new_name, name_to_node)
        assert nodes[1].name == new_name, f"Expected {new_name}, got {nodes[0].name}"
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"

    def test_onnx_dynamo_export_raises_when_model_contains_unsupported_fx_nodes(self):
        @torch.library.custom_op(
            "mylibrary::foo_op", device_types="cpu", mutates_args=()
        )
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @torch.library.custom_op(
            "mylibrary::bar_op", device_types="cpu", mutates_args=()
        )
        def bar_op(x: torch.Tensor) -> torch.Tensor:
            return x + 2

        @foo_op.register_fake
        def _(x):
            return torch.empty_like(x)

        @bar_op.register_fake
        def _(x):
            return torch.empty_like(x)

        def func(x, y, z):
            return foo_op(x) + bar_op(y) + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        with self.assertRaises(torch.onnx.OnnxExporterError) as ctx:
            torch.onnx.dynamo_export(func, x, y, z)
        inner_exception = ctx.exception.__cause__
        self.assertRegex(
            str(inner_exception),
            r"Unsupported FX nodes.*mylibrary\.foo_op.*mylibrary\.bar_op",
        )

        torch._dynamo.reset()


@common_utils.instantiate_parametrized_tests
class TestModularizePass(common_utils.TestCase):
    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_GELU_used_gelu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    def test_modularize_pass_succeeds_when_submodule_output_is_unused(
        self, is_exported_program
    ):
        # This is an ill-formed model, but exporter must not crash.
        # It is illegal for submodule to have zero output. For modularization pass it can happen
        # when the submodule output is unused, so no inner node is connected to any outer
        # nodes.
        # However, this also means the entire submodule should be erased by DCE. Hence
        # it should never occur.
        #
        # Minified repro from Background_Matting. https://github.com/pytorch/benchmark/issues/1768
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unused_relu = torch.nn.ReLU()
                self.used_gelu = torch.nn.GELU()

            def forward(self, x, y):
                result = self.used_gelu(x + y)
                unused_relu_result = self.unused_relu(x)
                return result

        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            model = TestModule()

        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))
        model_proto = onnx_program.model_proto
        function_proto_names = [function.name for function in model_proto.functions]
        self.assertIn(
            "torch_nn_modules_activation_GELU_used_gelu_1", function_proto_names
        )
        self.assertFalse(any("ReLU" in name for name in function_proto_names))

    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_ReLU_relu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    def test_modularize_pass_succeeds_when_a_submodule_is_called_multiple_times(
        self, is_exported_program
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x, y):
                out = x + y
                out = self.relu(out)
                out = out + x
                out = self.relu(out)
                return out

        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            model = TestModule()

        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))
        model_proto = onnx_program.model_proto
        function_proto_names = [function.name for function in model_proto.functions]
        self.assertIn("torch_nn_modules_activation_ReLU_relu_1", function_proto_names)
        self.assertIn("torch_nn_modules_activation_ReLU_relu_2", function_proto_names)

    @pytorch_test_common.xfail(
        error_message="'torch_nn_modules_activation_ReLU_inner_module_relu_1' not found",
        reason="optimizer",
    )
    @common_utils.parametrize(
        "is_exported_program",
        [
            common_utils.subtest(
                True,
                name="exported_program",
            ),
            common_utils.subtest(
                False,
                name="nn_module",
            ),
        ],
    )
    def test_modularize_pass_succeeds_when_a_submodule_is_called_from_multiple_layers(
        self, is_exported_program
    ):
        # Minified repro from basic_gnn_edgecnn.
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_module = InnerModule()

            def forward(self, x, y):
                out = x + y
                out = self.inner_module(out)
                out = out + x
                out = self.inner_module.relu(out)
                return out

        if is_exported_program:
            model = torch.export.export(
                TestModule(), args=(torch.randn(3), torch.randn(3))
            )
        else:
            model = TestModule()

        onnx_program = torch.onnx.dynamo_export(model, torch.randn(3), torch.randn(3))
        model_proto = onnx_program.model_proto
        function_proto_names = [function.name for function in model_proto.functions]
        self.assertIn(
            "torch_nn_modules_activation_ReLU_inner_module_relu_1", function_proto_names
        )
        self.assertIn(
            "torch_nn_modules_activation_ReLU_inner_module_relu_2", function_proto_names
        )
        # local module qualified name is unstable in test environment depending on different test
        # invocation methods.
        self.assertTrue(
            any("InnerModule_inner_module_1" in name for name in function_proto_names)
        )


if __name__ == "__main__":
    common_utils.run_tests()
