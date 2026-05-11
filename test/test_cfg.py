# Owner(s): ["module: fx"]

import torch
import torch.cfg as cfg
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.testing._internal.common_utils import run_tests, TestCase


class SampleObject:
    pass


class TestTorchCfg(TestCase):
    def test_manual_cfg_is_valid_and_printable(self):
        x = cfg.Value("x", cfg.TensorSpec.from_tensor(torch.randn(2, 3)))
        negative_input = cfg.Value("negative_input", x.spec)
        result = cfg.Value("result", x.spec)
        pred = cfg.Value("pred", cfg.ScalarSpec(bool))
        negated = cfg.Value("negated", x.spec)

        graph = cfg.Graph(
            name="branchy",
            entry="entry",
            blocks=(
                cfg.Block(
                    name="entry",
                    parameters=(x,),
                    instructions=(
                        cfg.Instruction(
                            name="lt_zero",
                            opcode="call_function",
                            target=torch.ops.aten.lt.Scalar,
                            inputs=(x, 0),
                            outputs=(pred,),
                        ),
                    ),
                    terminator=cfg.Branch(
                        pred,
                        cfg.Successor("negative", (x,)),
                        cfg.Successor("done", (x,)),
                    ),
                ),
                cfg.Block(
                    name="negative",
                    parameters=(negative_input,),
                    instructions=(
                        cfg.Instruction(
                            name="negate",
                            opcode="call_function",
                            target=torch.neg,
                            inputs=(negative_input,),
                            outputs=(negated,),
                        ),
                    ),
                    terminator=cfg.Jump(cfg.Successor("done", (negated,))),
                ),
                cfg.Block(
                    name="done",
                    parameters=(result,),
                    instructions=(),
                    terminator=cfg.Return(result),
                ),
            ),
        )

        rendered = graph.format()
        self.assertIn("graph branchy:", rendered)
        self.assertIn("block entry", rendered)
        self.assertIn("branch %pred -> negative(%x), done(%x)", rendered)
        self.assertIn("jump done(%negated)", rendered)

        self.assertEqual(graph.block("done").name, "done")
        with self.assertRaisesRegex(KeyError, "Unknown block"):
            graph.block("missing")

    def test_spec_and_format_helpers_cover_non_tensor_values(self):
        tuple_spec = cfg.Spec.from_value((1, None))
        list_spec = cfg.Spec.from_value([True, 3.5])
        dict_spec = cfg.Spec.from_value({"value": [1, None]})
        object_spec = cfg.Spec.from_value(SampleObject())

        self.assertEqual(tuple_spec.format(), "tuple[int, Optional[unknown]]")
        self.assertEqual(list_spec.format(), "list[bool, float]")
        self.assertEqual(
            dict_spec.format(),
            "dict[value: list[int, Optional[unknown]]]",
        )
        self.assertIsInstance(object_spec, cfg.ObjectSpec)
        self.assertIs(object_spec.python_type, SampleObject)
        self.assertRegex(object_spec.format(), r"(?:^|\.)SampleObject$")
        self.assertEqual(
            cfg.Location(file="example.py", line=7).format(),
            "example.py:7",
        )
        self.assertEqual(
            cfg.Location(file="example.py", line=7, function="forward").format(),
            "example.py:7 in forward",
        )
        self.assertEqual(cfg.Location(function="forward").format(), "forward")
        self.assertEqual(
            cfg.Location(stack="frame 0\nexample.py:9 in call").format(),
            "example.py:9 in call",
        )
        self.assertEqual(cfg.Location().format(), "<unknown>")

        x = cfg.Value("x")
        y = cfg.Value("y")
        instruction = cfg.Instruction(
            name="add",
            opcode="call_function",
            target="add",
            inputs=(x, (1,), [2], {"scale": 3}, slice(0, x, None)),
            attributes={"alpha": 4},
            outputs=(y,),
        )
        self.assertEqual(
            instruction.format(),
            "%y = call_function[target=add](%x, (1,), [2], {scale: 3}, "
            "slice(0, %x, None), alpha=4)",
        )

    def test_tensor_spec_from_tensor_handles_dense_and_nested_tensors(self):
        dense = torch.randn(2, 3, requires_grad=True)
        dense_spec = cfg.TensorSpec.from_tensor(dense)

        self.assertEqual(dense_spec.shape, (2, 3))
        self.assertEqual(dense_spec.dtype, dense.dtype)
        self.assertEqual(dense_spec.device, dense.device)
        self.assertEqual(dense_spec.stride, dense.stride())
        self.assertTrue(dense_spec.requires_grad)

        if not hasattr(torch, "nested") or not hasattr(torch.nested, "nested_tensor"):
            self.skipTest("nested tensors are unavailable")

        nested = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 3)])
        nested_spec = cfg.TensorSpec.from_tensor(nested)

        self.assertIsNone(nested_spec.shape)
        self.assertIsNone(nested_spec.stride)
        self.assertEqual(nested_spec.nested_size, ((2, 3), (4, 3)))

    def test_from_fx_normalizes_metadata_and_preserves_structure(self):
        fx_graph = Graph()
        x = fx_graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3)
        add = fx_graph.call_function(torch.add, args=(x, 1.0))
        add.meta["val"] = torch.randn(2, 3)
        fx_graph.output((x, {"sum": add}))

        graph = cfg.from_fx(fx_graph, name="from_fx")

        entry = graph.block("entry")
        self.assertEqual(entry.parameters[0].name, "x")
        self.assertIsInstance(entry.parameters[0].spec, cfg.TensorSpec)
        self.assertEqual(len(entry.instructions), 1)
        instruction = entry.instructions[0]
        self.assertEqual(instruction.name, "add")
        self.assertEqual(str(instruction.inputs[1]), "1.0")
        self.assertIsInstance(instruction.outputs[0].spec, cfg.TensorSpec)
        self.assertEqual(
            entry.terminator.format(),
            "return (%x, {sum: %add})",
        )

    def test_from_fx_accepts_graph_module_and_preserves_stack_trace(self):
        fx_graph = Graph()
        x = fx_graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3)
        add = fx_graph.call_function(torch.add, args=(x, 1.0))
        add.meta["val"] = torch.randn(2, 3)
        add.meta["stack_trace"] = "frame 1\nexample.py:5 in forward"
        fx_graph.output(add)

        graph = cfg.from_fx(
            GraphModule(torch.nn.Module(), fx_graph),
            name="graph_module",
        )

        entry = graph.block("entry")
        self.assertEqual(graph.name, "graph_module")
        self.assertEqual(entry.terminator.format(), "return %add")
        self.assertEqual(
            entry.instructions[0].location.format(),
            "example.py:5 in forward",
        )

    def test_from_fx_uses_tensor_meta_and_python_type_fallbacks(self):
        fx_graph = Graph()
        tensor = fx_graph.placeholder("tensor")
        tensor.meta["tensor_meta"] = TensorMetadata(
            shape=torch.Size([2, 3]),
            dtype=torch.float32,
            requires_grad=True,
            stride=(3, 1),
            memory_format=torch.contiguous_format,
            is_quantized=False,
            qparams={},
        )
        flag = fx_graph.placeholder("flag")
        flag.type = bool
        obj = fx_graph.placeholder("obj")
        obj.type = SampleObject
        fx_graph.output((tensor, flag, obj))

        graph = cfg.from_fx(fx_graph, name="fallbacks")
        entry = graph.block("entry")

        tensor_spec = entry.parameters[0].spec
        self.assertIsInstance(tensor_spec, cfg.TensorSpec)
        self.assertEqual(tensor_spec.shape, (2, 3))
        self.assertEqual(tensor_spec.dtype, torch.float32)
        self.assertEqual(tensor_spec.device, torch.device("meta"))
        self.assertEqual(tensor_spec.stride, (3, 1))
        self.assertTrue(tensor_spec.requires_grad)
        self.assertEqual(entry.parameters[1].spec, cfg.ScalarSpec(bool))
        self.assertEqual(entry.parameters[2].spec, cfg.ObjectSpec(SampleObject))

    def test_from_fx_requires_output_node(self):
        fx_graph = Graph()
        fx_graph.placeholder("x")

        with self.assertRaisesRegex(ValueError, "FX graph is missing an output node"):
            cfg.from_fx(fx_graph)

    def test_invalid_cfg_raises_validation_error(self):
        x = cfg.Value("x")
        with self.assertRaisesRegex(
            cfg.ValidationError,
            "expects 1 arguments but received 2",
        ):
            cfg.Graph(
                name="invalid",
                entry="entry",
                blocks=(
                    cfg.Block(
                        name="entry",
                        parameters=(x,),
                        terminator=cfg.Jump(
                            cfg.Successor("done", (x, cfg.literal(1))),
                        ),
                    ),
                    cfg.Block(
                        name="done",
                        parameters=(cfg.Value("y"),),
                        terminator=cfg.Return(None),
                    ),
                ),
            )

    def test_invalid_cfg_rejects_other_validation_failures(self):
        x = cfg.Value("x")
        y = cfg.Value("y")

        cases = [
            (
                "empty graph name",
                lambda: cfg.Graph(
                    name="",
                    entry="entry",
                    blocks=(cfg.Block(name="entry", terminator=cfg.Return(None)),),
                ),
                "graph name must be non-empty",
            ),
            (
                "empty block name",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(cfg.Block(name="", terminator=cfg.Return(None)),),
                ),
                "block name must be non-empty",
            ),
            (
                "duplicate block names",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            terminator=cfg.Jump(cfg.Successor("done")),
                        ),
                        cfg.Block(name="entry", terminator=cfg.Return(None)),
                    ),
                ),
                "duplicate block name 'entry'",
            ),
            (
                "missing entry block",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(cfg.Block(name="other", terminator=cfg.Return(None)),),
                ),
                "entry block 'entry' does not exist",
            ),
            (
                "empty instruction name",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            parameters=(x,),
                            instructions=(
                                cfg.Instruction(
                                    name="",
                                    opcode="call_function",
                                    target=torch.neg,
                                    inputs=(x,),
                                    outputs=(y,),
                                ),
                            ),
                            terminator=cfg.Return(y),
                        ),
                    ),
                ),
                "contains an instruction with an empty name",
            ),
            (
                "empty value name",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            parameters=(cfg.Value(""),),
                            terminator=cfg.Return(None),
                        ),
                    ),
                ),
                "defines a value with an empty name",
            ),
            (
                "value redefinition in block",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            parameters=(x,),
                            instructions=(
                                cfg.Instruction(
                                    name="negate",
                                    opcode="call_function",
                                    target=torch.neg,
                                    inputs=(x,),
                                    outputs=(cfg.Value("tmp"),),
                                ),
                                cfg.Instruction(
                                    name="add",
                                    opcode="call_function",
                                    target=torch.add,
                                    inputs=(x, 1),
                                    outputs=(cfg.Value("tmp"),),
                                ),
                            ),
                            terminator=cfg.Return(None),
                        ),
                    ),
                ),
                "redefines value 'tmp' in the same block",
            ),
            (
                "value collision across blocks",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            parameters=(cfg.Value("x"),),
                            terminator=cfg.Jump(cfg.Successor("done", (1,))),
                        ),
                        cfg.Block(
                            name="done",
                            parameters=(cfg.Value("x"),),
                            terminator=cfg.Return(None),
                        ),
                    ),
                ),
                "collides with existing value 'x'",
            ),
            (
                "jump to unknown block",
                lambda: cfg.Graph(
                    name="invalid",
                    entry="entry",
                    blocks=(
                        cfg.Block(
                            name="entry",
                            terminator=cfg.Jump(cfg.Successor("missing")),
                        ),
                    ),
                ),
                "jumps to unknown block 'missing'",
            ),
        ]

        for name, build_graph, error in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(cfg.ValidationError, error):
                    build_graph()

    def test_invalid_cfg_rejects_cross_block_reference(self):
        x = cfg.Value("x")
        y = cfg.Value("y")

        with self.assertRaisesRegex(
            cfg.ValidationError,
            "references undefined value 'x'",
        ):
            cfg.Graph(
                name="invalid",
                entry="entry",
                blocks=(
                    cfg.Block(
                        name="entry",
                        parameters=(x,),
                        terminator=cfg.Jump(cfg.Successor("done")),
                    ),
                    cfg.Block(
                        name="done",
                        instructions=(
                            cfg.Instruction(
                                name="negate",
                                opcode="call_function",
                                target=torch.neg,
                                inputs=(x,),
                                outputs=(y,),
                            ),
                        ),
                        terminator=cfg.Return(y),
                    ),
                ),
            )


if __name__ == "__main__":
    run_tests()
