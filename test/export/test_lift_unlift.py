# Owner(s): ["oncall: export"]
import unittest
from typing import Any, Dict, Optional, OrderedDict, Tuple

import torch
from torch._export.passes.lift_constants_pass import (
    ConstantAttrMap,
    lift_constants_pass,
)
from torch.export._unlift import _unlift_exported_program_lifted_states
from torch.export.exported_program import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)

from torch.export.graph_signature import CustomObjArgument
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


class GraphBuilder:
    def __init__(self):
        self.graph = torch.fx.Graph()
        self.nodes = {}
        self.values = {}
        self.nn_module_stack_key: Dict[str, int] = {}
        self.latest_id = 0
        self.input_to_kind: Dict[torch.fx.Node, InputKind] = {}

    def input(self, name: str, value: torch.Tensor, kind: InputKind):
        node = self.graph.placeholder(name)
        node.meta["val"] = value
        self.nodes[name] = node
        self.values[name] = value
        self.input_to_kind[node] = kind

    def add(self, x: str, y: str, out: str, module_fqn: str = ""):
        node = self.graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            (self.nodes[x], self.nodes[y]),
            name=out,
        )
        self.values[out] = self.values[x] + self.values[y]
        node.meta["val"] = self.values[out]
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[out] = node

    def call_function(self, target, args, out: str, module_fqn: str = ""):
        arg_nodes = tuple(self.nodes[arg] for arg in args)
        arg_values = tuple(self.values[arg] for arg in args)
        node = self.graph.create_node(
            "call_function",
            target,
            arg_nodes,
            name=out,
        )
        self.values[out] = target(*arg_values)
        node.meta["val"] = self.values[out]
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[out] = node

    def constant(
        self, name: str, value: Any, target: Optional[str] = None, module_fqn: str = ""
    ):
        if target is None:
            target = name
        node = self.graph.get_attr(target)
        node.meta["val"] = value
        node.meta["nn_module_stack"] = self.create_nn_module_stack(module_fqn)
        self.nodes[name] = node
        self.values[name] = value

    def output(self, out: str):
        self.graph.output(self.nodes[out])

    def create_nn_module_stack(
        self, module_fqn: str
    ) -> OrderedDict[int, Tuple[str, type]]:
        cur_name = ""
        nn_module_stack = OrderedDict()
        for atom in module_fqn.split("."):
            if cur_name == "":
                cur_name = atom
            else:
                cur_name = cur_name + "." + atom

            if cur_name not in self.nn_module_stack_key:
                id_counter = self.latest_id
                self.latest_id += 1
                self.nn_module_stack_key[cur_name] = id_counter
            else:
                id_counter = self.nn_module_stack_key[cur_name]

            nn_module_stack[id_counter] = (cur_name, torch.nn.Module)
        return nn_module_stack

    def create_input_specs(self):
        input_specs = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                input_specs.append(
                    InputSpec(
                        kind=self.input_to_kind[node],
                        arg=TensorArgument(name=node.name),
                        target=None,
                        persistent=(
                            True
                            if self.input_to_kind[node] == InputKind.BUFFER
                            else None
                        ),
                    )
                )
        return input_specs

    # NOTE: does not handle non-user-outputs atm
    def gen_graph_signature(self) -> ExportGraphSignature:
        output = [n for n in self.graph.nodes if n.op == "output"]
        assert len(output) == 1
        output = output[0]
        assert len(output.args) == 1, "multiple outputs NYI"

        return ExportGraphSignature(
            input_specs=self.create_input_specs(),
            output_specs=[
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=n.name),
                    target=None,
                )
                for n in output.args
            ],
        )


class TestLift(TestCase):
    def setUp(self):
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def test_lift_basic(self):
        builder = GraphBuilder()

        builder.input("param", torch.rand(2, 3), InputKind.PARAMETER)
        builder.input("buffer", torch.rand(2, 3), InputKind.BUFFER)
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("y", torch.rand(2, 3), InputKind.USER_INPUT)

        builder.add("x", "y", out="foo")
        builder.add("foo", "param", out="bar")
        builder.add("bar", "buffer", out="baz")
        builder.constant("const_tensor", torch.rand(2, 3))
        builder.constant("const_obj", torch.classes._TorchScriptTesting._Foo(10, 20))
        builder.add("baz", "const_tensor", out="out")
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj", "x"),
            out="torchbind_out",
        )
        builder.add("out", "torchbind_out", out="final_out")
        builder.output("final_out")

        builder.graph.lint()
        graph = builder.graph
        const_tensor = builder.values["const_tensor"]
        const_obj = builder.values["const_obj"]

        root = {"const_tensor": const_tensor, "const_obj": const_obj}
        gm = torch.fx.GraphModule(root, graph)
        graph_signature = builder.gen_graph_signature()
        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 2)

        # The key of the constants table should match the fqn of the constant.
        # In this case, it's just the name of the constant, since the constant
        # is at the root submodule.
        # TODO(suo): we shouldn't hardcode these names in the test, this is an
        # internal detail of the pass.
        self.assertIn("lifted_tensor_0", constants)
        self.assertEqual(constants["lifted_tensor_0"], const_tensor)
        self.assertIn("lifted_custom_0", constants)
        self.assertEqual(constants["lifted_custom_0"], const_obj)

        # The constant node should be removed.
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # The constant should be lifted to a placeholder node.
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 6)

        # The lifted constant should be placed before user inputs but after params/buffers
        lifted_tensor_placeholder = placeholder_nodes[2]
        self.assertEqual(lifted_tensor_placeholder.target, "lifted_tensor_0")
        # It should have a val equivalent to the constant
        self.assertEqual(lifted_tensor_placeholder.meta["val"], const_tensor)

        lifted_obj_placeholder = placeholder_nodes[3]
        self.assertEqual(lifted_obj_placeholder.target, "lifted_custom_0")
        # It should have a val equivalent to the constant
        self.assertEqual(
            lifted_obj_placeholder.meta["val"],
            CustomObjArgument(
                name="lifted_custom_0",
                class_fqn="__torch__.torch.classes._TorchScriptTesting._Foo",
            ),
        )

        # Graph signature should have been mutated a way that reflects the placeholders.
        tensor_constant_input_spec = graph_signature.input_specs[2]
        self.assertEqual(tensor_constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(tensor_constant_input_spec.arg, TensorArgument)
        self.assertEqual(
            tensor_constant_input_spec.arg.name, lifted_tensor_placeholder.name
        )

        obj_constant_input_spec = graph_signature.input_specs[3]
        self.assertEqual(obj_constant_input_spec.kind, InputKind.CUSTOM_OBJ)
        self.assertIsInstance(obj_constant_input_spec.arg, CustomObjArgument)
        self.assertEqual(obj_constant_input_spec.arg.name, lifted_obj_placeholder.name)

    def test_lift_nested(self):
        builder = GraphBuilder()
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("y", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.input("z", torch.rand(2, 3), InputKind.USER_INPUT)

        builder.add("x", "y", out="foo")
        builder.add("foo", "z", out="bar", module_fqn="foo")
        builder.constant("const_tensor", torch.rand(2, 3), module_fqn="foo")
        builder.add("bar", "const_tensor", "out")
        builder.output("out")

        graph = builder.graph
        graph.lint()

        const_tensor = builder.values["const_tensor"]
        root = {"const_tensor": builder.values["const_tensor"]}

        graph_signature = builder.gen_graph_signature()
        gm = torch.fx.GraphModule(root, graph)

        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 1)

        # The key of the constants table should match the fqn of the constant.
        self.assertIn("foo.lifted_tensor_0", constants)
        self.assertEqual(constants["foo.lifted_tensor_0"], const_tensor)

        # The constant node should be removed.
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # The constant should be lifted to a placeholder node.
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 4)

        # The lifted constant should be placed before user inputs but after params/buffers
        lifted_constant_placeholder = placeholder_nodes[0]
        self.assertEqual(lifted_constant_placeholder.target, "lifted_tensor_0")

        # Graph signature should have been mutated a way that reflects the placeholders.
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)
        self.assertEqual(constant_input_spec.arg.name, lifted_constant_placeholder.name)

    def test_duplicate_constant_access(self):
        const = torch.rand(2, 3)
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)

        builder = GraphBuilder()
        builder.input("x", torch.rand(2, 3), InputKind.USER_INPUT)
        builder.constant("const_tensor", const, target="const_tensor")
        # loading the same target twice
        builder.constant("const_tensor2", const, target="const_tensor")

        # loading the same object twice with different targets
        builder.constant("const_obj", const_obj)
        builder.constant("const_obj2", const_obj)
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj", "x"),
            out="torchbind_out",
        )
        builder.call_function(
            torch.ops._TorchScriptTesting.takes_foo,
            ("const_obj2", "x"),
            out="torchbind_out2",
        )
        builder.add("x", "const_tensor", out="foo")
        builder.add("foo", "const_tensor2", out="tensor_out")
        builder.add("torchbind_out", "torchbind_out2", out="obj_out")
        builder.add("tensor_out", "obj_out", out="out")
        builder.output("out")
        graph = builder.graph
        graph.lint()

        input_specs = builder.create_input_specs()
        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=builder.nodes["out"].name),
                target=None,
            )
        ]
        graph_signature = ExportGraphSignature(input_specs, output_specs)

        root = {"const_tensor": const, "const_obj": const_obj, "const_obj2": const_obj}
        gm = torch.fx.GraphModule(root, graph)

        constants = lift_constants_pass(gm, graph_signature, {})
        gm.graph.lint()

        self.assertEqual(len(constants), 2)

        # All get_attr nodes should be removed
        getattr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 0)

        # There should only be two additional inputs (plus the existing user input)
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholder_nodes), 3)

        # Graph signature should have been mutated a way that reflects the placeholders.
        self.assertEqual(len(graph_signature.input_specs), 3)
        constant_input_spec = graph_signature.input_specs[0]
        self.assertEqual(constant_input_spec.kind, InputKind.CONSTANT_TENSOR)
        self.assertIsInstance(constant_input_spec.arg, TensorArgument)

    def test_unlift_nonpersistent_buffer(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "non_persistent_buf", torch.zeros(1), persistent=False
                )

            def forward(self, x):
                self.non_persistent_buf.add_(1)
                return x.sum() + self.non_persistent_buf.sum()

        foo = Foo()
        exported = torch.export.export(foo, (torch.ones(5, 5),), strict=False)
        stateful_gm = _unlift_exported_program_lifted_states(exported)

        # Check the unlifted stateful_gm contains the original non-persistent buffer
        self.assertTrue(hasattr(stateful_gm, "non_persistent_buf"))
        non_persistent_buf = stateful_gm.get_buffer("non_persistent_buf")
        self.assertEqual(non_persistent_buf, foo.get_buffer("non_persistent_buf"))
        self.assertIn("non_persistent_buf", stateful_gm._non_persistent_buffers_set)
        self.assertNotIn("non_persistent_buf", stateful_gm.state_dict())


class ConstantAttrMapTest(TestCase):
    def setUp(self):
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def test_dict_api(self):
        constant_attr_map = ConstantAttrMap()
        const_obj = torch.classes._TorchScriptTesting._Foo(10, 20)
        const_tensor = torch.ones(2, 3)
        constant_attr_map[const_obj] = "foo.bar"
        constant_attr_map[const_tensor] = "foo.bar.baz"
        self.assertEqual(len(constant_attr_map), 2)
        self.assertEqual(list(constant_attr_map), [const_obj, const_tensor])
        self.assertEqual(list(constant_attr_map.keys()), [const_obj, const_tensor])
        self.assertEqual(list(constant_attr_map.values()), ["foo.bar", "foo.bar.baz"])
        self.assertEqual(constant_attr_map[const_obj], "foo.bar")
        self.assertEqual(constant_attr_map[const_tensor], "foo.bar.baz")
        self.assertTrue(const_obj in constant_attr_map)
        with self.assertRaises(TypeError):
            constant_attr_map[1] = "foo.bar"

        del constant_attr_map[const_obj]
        self.assertEqual(len(constant_attr_map), 1)


if __name__ == "__main__":
    run_tests()
