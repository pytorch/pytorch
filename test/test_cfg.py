# Owner(s): ["module: fx"]

import torch
import torch.cfg as cfg
from torch.testing._internal.common_utils import run_tests, TestCase


class TestControlFlowGraph(TestCase):
    def test_build_and_validate_branching_graph(self):
        graph = cfg.Graph("branching")
        tensor_info = cfg.ValueInfo(
            tensor=cfg.TensorSpec(
                dtype=torch.float32,
                shape=(2, 4),
                device=torch.device("cpu"),
                requires_grad=False,
            )
        )

        entry = graph.new_block(
            "entry",
            parameters=["x", cfg.BlockParameterSpec("flag", cfg.ValueInfo(bool))],
            is_entry=True,
        )
        then_block = graph.new_block(
            "then",
            parameters=[cfg.BlockParameterSpec("then_x", tensor_info)],
        )
        else_block = graph.new_block(
            "else",
            parameters=[cfg.BlockParameterSpec("else_x", tensor_info)],
        )
        merge = graph.new_block(
            "merge",
            parameters=[cfg.BlockParameterSpec("merged", tensor_info)],
        )

        doubled = entry.call_function(
            "doubled",
            torch.add,
            args=(entry.parameters[0], entry.parameters[0]),
            info=tensor_info,
        )
        entry.branch(
            entry.parameters[1],
            true_target=then_block,
            false_target=else_block,
            true_arguments=(doubled,),
            false_arguments=(entry.parameters[0],),
        )

        then_relu = then_block.call_method(
            "then_relu",
            "relu",
            args=(then_block.parameters[0],),
            info=tensor_info,
        )
        then_block.jump(merge, then_relu)

        else_neg = else_block.call_function(
            "else_neg",
            torch.neg,
            args=(else_block.parameters[0],),
            info=tensor_info,
        )
        else_block.jump(merge, else_neg)

        merge.return_(merge.parameters[0])
        self.assertIs(graph.validate(), graph)

        rendered = str(graph)
        self.assertIn("block entry(%x: unknown, %flag: bool):", rendered)
        self.assertIn("branch %flag -> then(%doubled), else(%x)", rendered)
        self.assertIn("return %merged", rendered)

    def test_values_must_flow_through_block_parameters(self):
        graph = cfg.Graph("strict_scope")
        entry = graph.new_block("entry", parameters=["x"], is_entry=True)
        consumer = graph.new_block("consumer")

        temp = entry.call_function("temp", torch.neg, args=(entry.parameters[0],))
        entry.jump(consumer)
        consumer.return_(temp)

        with self.assertRaisesRegex(ValueError, "without receiving it as a block parameter"):
            graph.validate()

    def test_duplicate_value_names_are_rejected(self):
        graph = cfg.Graph("duplicate_names")
        entry = graph.new_block("entry", parameters=["x"], is_entry=True)
        with self.assertRaisesRegex(ValueError, "already contains a value named 'x'"):
            entry.call_function("x", torch.neg, args=(entry.parameters[0],))

    def test_from_fx_lifts_metadata_into_value_info(self):
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                y = self.linear(x)
                z = torch.add(y, 1)
                return z.relu()

        gm = torch.fx.symbolic_trace(Module())

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = torch.randn(2, 4)
            elif node.op == "call_module":
                node.meta["example_value"] = torch.randn(2, 4)
            elif node.op == "call_function":
                node.meta["example_value"] = torch.randn(2, 4)
            elif node.op == "call_method":
                node.meta["val"] = torch.randn(2, 4)

        graph = cfg.Graph.from_fx(gm, name="from_fx")

        entry = graph.entry
        self.assertEqual(len(entry.parameters), 1)
        self.assertTrue(entry.parameters[0].info.has_example_value)
        self.assertEqual(entry.parameters[0].info.tensor.shape, (2, 4))

        instructions = entry.instructions
        self.assertEqual(
            [instruction.kind for instruction in instructions],
            [
                cfg.InstructionKind.CALL_MODULE,
                cfg.InstructionKind.CALL_FUNCTION,
                cfg.InstructionKind.CALL_METHOD,
            ],
        )
        self.assertTrue(all(instruction.result is not None for instruction in instructions))
        self.assertFalse(hasattr(instructions[0].result, "meta"))
        self.assertEqual(entry.terminator.value, instructions[-1].result)

    def test_torch_module_exposes_cfg_lazily(self):
        self.assertIs(torch.cfg.Graph, cfg.Graph)
        self.assertIs(torch.cfg.ControlFlowGraph, cfg.Graph)


if __name__ == "__main__":
    run_tests()
