# Owner(s): ["module: onnx"]

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.onnx._internal.fx.passes import type_promotion
from torch.testing._internal import common_utils


# The following ops are ignored because we do not need these rules enabled for ONNX
IGNORED_OPS = {
    "pow",
    "pow_",
}


class TestGeneratedTypePromotionRuleSet(common_utils.TestCase):
    def test_generated_rule_set_is_up_to_date(self):
        generated_set = type_promotion._GENERATED_ATEN_TYPE_PROMOTION_RULE_SET
        latest_set = type_promotion.ElementwiseTypePromotionRuleSetGenerator.generate_from_torch_refs()
        latest_set = {rule for rule in latest_set if rule.op_name not in IGNORED_OPS}

        # Please update the list in torch/onnx/_internal/fx/passes/type_promotion.py following the instruction
        # if this test fails
        self.assertEqual(generated_set, latest_set)

    def test_initialize_type_promotion_table_succeeds(self):
        type_promotion.TypePromotionTable()


class TestFindCompatibleOpOverload(common_utils.TestCase):
    def test_selects_schema_compatible_overload(self):
        condition = torch.tensor([True, False])
        args = (condition, torch.tensor(0.0), -1000.0)

        overload = type_promotion.find_compatible_op_overload(
            torch.ops.aten.where, args, {}
        )

        self.assertEqual(overload, torch.ops.aten.where.ScalarOther)


class TestInsertTypePromotion(common_utils.TestCase):
    def test_materializes_only_scalar_without_compatible_overload(self):
        class Model(torch.nn.Module):
            def forward(self, x, mask):
                return 1.0 - mask, x + 1.0

        x = torch.randn(2, 3)
        mask = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
        exported_program = torch.export.export(Model(), (x, mask)).run_decompositions()
        graph_module = exported_program.graph_module

        type_promotion.InsertTypePromotion(graph_module).run()

        scalar_tensor_nodes = graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.scalar_tensor.default
        )
        sub_nodes = graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.sub.Tensor
        )
        add_nodes = graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )
        self.assertEqual(len(scalar_tensor_nodes), 1)
        self.assertEqual(len(sub_nodes), 1)
        self.assertEqual(len(add_nodes), 1)
        self.assertIs(sub_nodes[0].args[0], scalar_tensor_nodes[0])
        self.assertEqual(add_nodes[0].args[1], 1.0)


class TestTypePromotionONNXExport(common_utils.TestCase):
    def test_scalar_minus_int_tensor_exports(self):
        class Model(torch.nn.Module):
            def forward(self, x, mask):
                return (1.0 - mask) * x

        model = Model().eval()
        x = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        mask = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
        exported_program = torch.export.export(model, (x, mask))

        onnx_program = torch.onnx.export(
            exported_program,
            (x, mask),
            dynamo=True,
            verbose=False,
        )

        onnx_testing.assert_onnx_program(
            onnx_program, args=(x, mask), strategy=None
        )

    def test_where_with_bool_tensor_and_mixed_scalars_exports(self):
        class WhereModel(torch.nn.Module):
            def forward(self, condition):
                return torch.where(condition, 0, -1000.0)

        model = WhereModel().eval()
        condition = torch.randn(1, 1, 4, 4) > 0

        onnx_program = torch.onnx.export(
            model,
            (condition,),
            dynamo=True,
            verbose=False,
        )

        self.assertIsNotNone(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
