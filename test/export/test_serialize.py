# Owner(s): ["module: dynamo"]
import io
import pathlib
import tempfile
import unittest
import zipfile

import torch
import torch._dynamo as torchdynamo
from torch._export import export, save, load
from torch._export.constraints import constrain_as_size
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel
from torch._export.db.examples import all_examples
from torch._export.serde.serialize import (
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    deserialize,
    serialize,
    SerializeError,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
    TemporaryFileName,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    find_library_location,
)


def get_filtered_export_db_tests():
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
        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.native_layer_norm.default")
        # aten::native_layer_norm returns 3 tensnors
        self.assertEqual(len(node.outputs), 3)

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
        # split.Tensor gets decomposed to split_with_sizes by the core ATen decomposition table
        self.assertEqual(node.target, "torch.ops.aten.split_with_sizes.default")
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
    def check_graph(self, fn, inputs, dynamic_shapes=None, _check_meta=True) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""
        # TODO(angelayi): test better with some sort of wrapper
        ep = torch.export.export(fn, inputs, {}, dynamic_shapes=dynamic_shapes)
        ep.graph.eliminate_dead_code()

        serialized_struct, state_dict = serialize(ep, opset_version={"aten": 0})
        deserialized_ep = deserialize(serialized_struct, state_dict, expected_opset_version={"aten": 0})
        deserialized_ep.graph.eliminate_dead_code()

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

        def _check_graph_nodes(gm1, gm2, _check_meta=True):
            # TODO: The _check_meta flag bypasses checking for
            # source_fn/nn_module_stack as there is an issue with
            # roundtripping the source_fn value on torch.ops.map nodes
            # original source_fn: <functorch.experimental._map.MapWrapper object at 0x7f80a0549930>
            # deserialized source_fn: 'functorch.experimental._map.map'

            self.assertEqual(len(gm1.graph.nodes), len(gm2.graph.nodes))

            for node1, node2 in zip(gm1.graph.nodes, gm2.graph.nodes):
                self.assertEqual(node1.op, node2.op)
                if node1.op == "call_function":
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
                    elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                        # Or both are fake tensors lists with one element and with the
                        # same shape/dtype
                        for v1, v2 in zip(val1, val2):
                            self.assertEqual(v1.shape, v2.shape)
                            self.assertEqual(v1.dtype, v2.dtype)
                    else:
                        # For expressions like 's0 < 10' can only compare through string
                        self.assertEqual(str(val1), str(val2))

                    # Check "stack_trace" metadata
                    self.assertEqual(
                        node1.meta.get("stack_trace", None),
                        node2.meta.get("stack_trace", None),
                    )

                    if node1.target == torch.ops.higher_order.cond:
                        true_graph1 = getattr(gm1, node1.args[1].target)
                        true_graph2 = getattr(gm2, node2.args[1].target)
                        _check_graph_nodes(true_graph1, true_graph2)

                        false_graph1 = getattr(gm1, node1.args[2].target)
                        false_graph2 = getattr(gm2, node2.args[2].target)
                        _check_graph_nodes(false_graph1, false_graph2)
                    elif node1.target == torch.ops.map_impl:
                        map_graph1 = getattr(gm1, node1.args[0].target)
                        map_graph2 = getattr(gm2, node2.args[0].target)
                        _check_graph_nodes(map_graph1, map_graph2, False)

                if (
                    _check_meta and
                    node1.op not in ("get_attr", "placeholder", "output")
                ):
                    # Check "nn_module_stack" metadata
                    # TODO nn_module_stack is not roundtrippable.
                    # self.assertEqual(
                    #     node1.meta.get("nn_module_stack", None),
                    #     node2.meta.get("nn_module_stack", None),
                    # )
                    # Check "source_fn" metadata
                    self.assertEqual(
                        node1.meta.get("source_fn", None),
                        node2.meta.get("source_fn", None),
                    )

        _check_graph_nodes(ep.graph_module, deserialized_ep.graph_module, _check_meta)

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
        dim0_ac = torch.export.Dim("dim0_ac")
        dynamic_shapes = {"a": {0: dim0_ac}, "b": None, "c": {0: dim0_ac}}
        self.check_graph(DynamicShapeSimpleModel(), inputs, dynamic_shapes)

    def test_sym_bool(self):
        def f(x, y):
            return x.size(0) in y

        self.check_graph(f, (torch.ones(2), torch.ones(3)))

    def test_shape(self):
        def f(x):
            z, y = x.size()
            return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        self.check_graph(f, inputs, dynamic_shapes)

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

    def test_map(self):
        from functorch.experimental import control_flow

        def f(x, y):
            return x + y

        def g(xs, y):
            return control_flow.map(f, xs, y)

        inputs = (torch.ones(3, 2, 2), torch.ones(2))
        self.check_graph(g, inputs, _check_meta=False)

    @parametrize(
        "name,case",
        get_filtered_export_db_tests(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model
        inputs = normalize_inputs(case.example_inputs)
        _check_meta = "map" not in name
        self.check_graph(model, inputs.args, _check_meta=_check_meta)

    def test_constraints(self):
        def f(x, y):
            n = x.item()
            constrain_as_size(n, min=2)
            return y.sum() + torch.ones(n, 5).sum()

        self.check_graph(f, (torch.tensor(3), torch.randn(4, 5)))

    def test_get_attr(self) -> None:
        def f(x):
            return x + torch.tensor(3)

        self.check_graph(f, (torch.tensor(3),))

    def test_get_attr_list(self) -> None:
        def f(x):
            return torch.cat([x, torch.tensor([1, 1])])

        self.check_graph(f, (torch.tensor([1, 1]),))


instantiate_parametrized_tests(TestDeserialize)

@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSchemaVersioning(TestCase):
    def test_error(self):
        def f(x):
            return x + x

        ep = export(f, (torch.randn(1, 3),))

        serialized_ep, serialized_state_dict = ExportedProgramSerializer().serialize(ep)
        serialized_ep.schema_version = -1
        with self.assertRaisesRegex(SerializeError, r"Serialized schema version -1 does not match our current"):
            ExportedProgramDeserializer().deserialize(serialized_ep, serialized_state_dict)


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


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSaveLoad(TestCase):
    def test_save_buffer(self):
        inp = (torch.tensor([0.1, 0.1]),)
        linear = torch.nn.Linear(2, 2)

        class Module(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                y = x.t()
                y = y.relu()
                y = linear(y)
                return y

        ep = export(Module(), inp)

        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        self.assertTrue(torch.allclose(ep(*inp), loaded_ep(*inp)))

    def test_save_file(self):

        def f(x):
            return x * x

        inp = (torch.randn(2, 2),)
        ep = export(f, inp)

        with tempfile.NamedTemporaryFile() as f:
            save(ep, f)
            f.seek(0)
            loaded_ep = load(f)

        self.assertTrue(torch.allclose(ep(*inp), loaded_ep(*inp)))

    def test_save_path(self):
        def f(x, y):
            return x + y

        inp = (torch.tensor([6]), torch.tensor([7]))
        ep = export(f, inp)

        with TemporaryFileName() as fname:
            path = pathlib.Path(fname)
            save(ep, path)
            loaded_ep = load(path)

        self.assertTrue(torch.allclose(ep(*inp), loaded_ep(*inp)))

    def test_save_extra(self):
        inp = (torch.tensor([0.1, 0.1]),)

        def f(x):
            return x * x + x

        ep = export(f, inp)

        buffer = io.BytesIO()
        save(ep, buffer, extra_files={"extra.txt": "moo"})
        buffer.seek(0)
        extra_files = {"extra.txt": ""}
        loaded_ep = load(buffer, extra_files=extra_files)

        self.assertTrue(torch.allclose(ep(*inp), loaded_ep(*inp)))
        self.assertEqual(extra_files["extra.txt"], "moo")

    def test_version_error(self):
        def f(x):
            return x + x

        ep = export(f, (torch.randn(1, 3),))

        with tempfile.NamedTemporaryFile() as f:
            save(ep, f)
            f.seek(0)

            # Modify the version
            with zipfile.ZipFile(f, 'a') as zipf:
                zipf.writestr('version', "-1")

            with self.assertRaisesRegex(RuntimeError, r"Serialized version -1 does not match our current"):
                f.seek(0)
                loaded_ep = load(f)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerializeCustomClass(TestCase):
    def setUp(self):
        if IS_SANDCASTLE or IS_MACOS or IS_FBCODE:
            raise unittest.SkipTest("non-portable load_library call used in test")
        lib_file_path = find_library_location('libtorchbind_test.so')
        if IS_WINDOWS:
            lib_file_path = find_library_location('torchbind_test.dll')
        torch.ops.load_library(str(lib_file_path))

    def test_custom_class(self):
        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        def f(x):
            return x + x

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs)

        # Replace one of the values with an instance of our custom class
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (custom_obj,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        serialized_vals = serialize(ep)
        deserialized_ep = deserialize(*serialized_vals)

        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function" and
                node.target == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(isinstance(arg, torch._C.ScriptObject))
                self.assertEqual(arg.__getstate__(), custom_obj.__getstate__())
                self.assertEqual(arg.top(), 7)


if __name__ == '__main__':
    run_tests()
