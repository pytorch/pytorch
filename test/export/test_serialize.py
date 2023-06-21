# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import dynamic_dim, export
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel
from torch._export.db.examples import all_examples
from torch._export.serde.serialize import (
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    deserialize,
    serialize,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def get_filtered_export_db_tests():
    unsupported_tags = {"torch.cond", "torch.map"}
    unsupported_test_names = {
        "dynamic_shape_constructor",  # 'NoneType' object has no attribute 'from_tensor'
        "dictionary",  # Graph output must be a tuple()
        "fn_with_kwargs",  # export doesn't support kwargs yet
        "scalar_output",  # Tracing through 'f' must produce a single graph
    }

    return [
        (name, case)
        for name, case in all_examples().items()
        if (
            case.support_level == SupportLevel.SUPPORTED and
            not (unsupported_tags & case.tags) and
            name not in unsupported_test_names
        )
    ]


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_serialize_multiple_returns_from_node(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
        )

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[-7]
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        # aten::native_layer_norm returns 3 tensnors
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_list_returns(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        input.requires_grad = True
        exported_module = export(MyModule(), (input,))

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.split.Tensor")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export(
            MyModule(),
            (torch.ones([512, 512], requires_grad=True),),
        )

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            values = torch.randn(3, 2)
            return torch.searchsorted(x, values, side="right", right=True)

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export(f, (x,))
        serialized, _ = ExportedProgramSerializer().serialize(exported_module)

        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 6)
        self.assertEqual(node.inputs[2].arg.as_bool, False)
        self.assertEqual(node.inputs[3].arg.as_bool, True)
        self.assertEqual(node.inputs[4].arg.as_string, "right")
        self.assertEqual(node.inputs[5].arg.as_none, ())


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestDeserialize(TestCase):
    def check_graph(self, fn, inputs, constraints=None) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""
        # TODO(angelayi): test better with some sort of wrapper
        constraints = [] if constraints is None else constraints
        ep = export(fn, inputs, constraints)
        serialized_struct, state_dict = serialize(ep, opset_version={"aten": 0})
        deserialized_ep = deserialize(serialized_struct, state_dict, expected_opset_version={"aten": 0})

        orig_outputs = ep(*inputs)
        loaded_outputs = deserialized_ep(*inputs)

        flat_orig_outputs, _ = pytree.tree_flatten(orig_outputs)
        flat_loaded_outputs, _ = pytree.tree_flatten(loaded_outputs)

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertEqual(type(orig), type(loaded))
            if isinstance(orig, torch.Tensor):
                self.assertTrue(torch.allclose(orig, loaded))
            else:
                self.assertEqual(orig, loaded)

        self.assertEqual(len(ep.graph.nodes), len(deserialized_ep.graph.nodes))
        for node1, node2 in zip(ep.graph.nodes, deserialized_ep.graph.nodes):
            # Check "val" metadata
            val1 = node1.meta.get("val", None)
            val2 = node2.meta.get("val", None)

            if val1 is None or val2 is None:
                # Either both are None
                self.assertEqual(val1, val2)
            elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                # Or both are fake tensors with the same shape/dtype
                self.assertEqual(len(val1.shape), len(val2.shape))
                for s1, s2 in zip(val1.shape, val2.shape):
                    if is_concrete_int(s1) and is_concrete_int(s2):
                        self.assertEqual(s1, s2)
                    else:
                        self.assertEqual(str(s1), str(s2))
                self.assertEqual(val1.dtype, val2.dtype)
            elif isinstance(val1, list) and isinstance(val2, list):
                # Or both are fake tensors lists with one element and with the
                # same shape/dtype
                self.assertTrue(len(val1) == 1 and len(val2) == 1)
                self.assertEqual(val1[0].shape, val2[0].shape)
                self.assertEqual(val1[0].dtype, val2[0].dtype)
            else:
                # For expressions like 's0 < 10' can only compare through string
                self.assertEqual(str(val1), str(val2))

            # Check "stack_trace" metadata
            if "None" in node1.meta.get("stack_trace"):
                self.assertTrue(
                    node2.meta.get("stack_trace") is None
                    or "None" in node2.meta.get("stack_trace")
                )
            else:
                self.assertEqual(
                    node1.meta.get("stack_trace", None),
                    node2.meta.get("stack_trace", None),
                )

            # Check "nn_module_stack" metadata
            self.assertEqual(
                node1.meta.get("nn_module_stack", None),
                node2.meta.get("nn_module_stack", None),
            )

            # Check "source_fn" metadata
            if node1.op != "get_attr":
                self.assertEqual(
                    node1.meta.get("source_fn", None),
                    node2.meta.get("source_fn", None),
                )

    def test_multi_return(self) -> None:
        """
        Test multiple return from a single node (ex. layer_norm has 2 outputs)
        """
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        inputs = (
            torch.ones([512, 512], requires_grad=True),
            torch.ones([512]),
            torch.ones([512]),
        )
        self.check_graph(MyModule(), inputs)

    def test_basic(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x * x
                x = x / x
                return x, x.clone()

        inputs = (torch.ones([512], requires_grad=True),)
        self.check_graph(MyModule(), inputs)

    def test_dynamic(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])


        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        constraints = [
            dynamic_dim(inputs[0], 0),
            dynamic_dim(inputs[2], 0),
            dynamic_dim(inputs[2], 0) == dynamic_dim(inputs[0], 0),
        ]
        self.check_graph(DynamicShapeSimpleModel(), inputs, constraints)

    def test_sym_bool(self):
        def f(x, y):
            return x.size(0) in y

        self.check_graph(f, (torch.ones(2), torch.ones(3)))

    def test_shape(self):
        def f(x):
            z, y = x.size()
            return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        constraints = [
            dynamic_dim(inputs[0], 0),
            dynamic_dim(inputs[0], 1),
        ]
        self.check_graph(f, inputs, constraints)

    def test_module(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear1(x)
                x = torch.nn.functional.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        self.check_graph(M(), inputs)

    def test_cond(self):
        from functorch.experimental.control_flow import cond
        inputs = torch.ones(4, 3), torch.zeros(4, 3)

        class M(torch.nn.Module):
            def forward(self, x, y):
                def t(x, y):
                    return x + y

                def f(x, y):
                    return x - y
                return cond(x[0][0] > 4, t, f, [x, y])

        self.check_graph(M(), inputs)

    @parametrize(
        "name,case",
        get_filtered_export_db_tests(),
        name_fn=lambda name, case: "case_{}".format(name),
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model
        inputs = normalize_inputs(case.example_inputs)
        self.check_graph(model, inputs.args)


instantiate_parametrized_tests(TestDeserialize)


class TestOpVersioning(TestCase):
    """Test if serializer/deserializer behaves correctly if version mismatch."""

    def test_empty_model_opset_version_raises(self):
        compiler_opset_version = {"aten": 4}
        model_opset_version = None
        deserializer = ExportedProgramDeserializer(compiler_opset_version)
        with self.assertRaises(RuntimeError):
            deserializer._validate_model_opset_version(model_opset_version)

    def test_opset_mismatch_raises(self):
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        deserializer = ExportedProgramDeserializer(compiler_opset_version)
        with self.assertRaises(NotImplementedError):
            deserializer._validate_model_opset_version(model_opset_version)

    def test_model_op_namespace_version_missing_from_deserializer_do_not_raises(self):
        compiler_opset_version = {"aten": 3}
        model_opset_version = {"aten": 3, "custom": 4}
        deserializer = ExportedProgramDeserializer(compiler_opset_version)
        with self.assertLogs(level='WARN') as log:
            deserializer._validate_model_opset_version(model_opset_version)
            self.assertIn("Compiler doesn't have a version table for op namespace", log.output[0])

unittest.expectedFailure(
    TestDeserialize.test_exportdb_supported_case_tensor_setattr
)
unittest.expectedFailure(
    TestDeserialize.test_exportdb_supported_case_pytree_flatten
)

if __name__ == '__main__':
    run_tests()
