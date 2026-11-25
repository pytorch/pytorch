# Owner(s): ["module: onnx"]

import onnx
import onnxruntime as ort

import torch
from torch.onnx._internal.fx.passes import type_promotion
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification


# The following ops are ignored because we do not need these rules enabled for ONNX
IGNORED_OPS = {
    "pow",
    "pow_",
}


class TestGeneratedTypePromotionRuleSet(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

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
    hw_classification = HardwareClassification.GENERIC

    def test_selects_schema_compatible_overload(self):
        condition = torch.tensor([True, False])
        args = (condition, torch.tensor(0.0), -1000.0)

        overload = type_promotion.find_compatible_op_overload(
            torch.ops.aten.where, args, {}
        )

        self.assertEqual(overload, torch.ops.aten.where.ScalarOther)


class TestTypePromotionONNXExport(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    def test_stack_with_mixed_dtypes_exports(self):
        class StackModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.stack([a, b], dim=0)

        model = StackModel().eval()
        dtype_pairs = [
            (torch.int8, torch.int64),
            (torch.int32, torch.float32),
            (torch.float16, torch.float32),
        ]
        for dtype_a, dtype_b in dtype_pairs:
            with self.subTest(dtype_a=dtype_a, dtype_b=dtype_b):
                a = torch.ones(2, 3, dtype=dtype_a)
                b = torch.full((2, 3), 2, dtype=dtype_b)
                expected = model(a, b)

                onnx_program = torch.onnx.export(
                    model,
                    (a, b),
                    dynamo=True,
                    opset_version=18,
                    verbose=False,
                )
                self.assertIsNotNone(onnx_program)
                onnx_model = onnx_program.model_proto
                onnx.checker.check_model(onnx_model, full_check=True)

                session = ort.InferenceSession(
                    onnx_model.SerializeToString(),
                    providers=["CPUExecutionProvider"],
                )
                input_names = [inp.name for inp in session.get_inputs()]
                (actual,) = session.run(
                    None,
                    {input_names[0]: a.numpy(), input_names[1]: b.numpy()},
                )
                self.assertEqual(actual.dtype, expected.numpy().dtype)
                self.assertEqual(actual, expected.numpy())


if __name__ == "__main__":
    common_utils.run_tests()
