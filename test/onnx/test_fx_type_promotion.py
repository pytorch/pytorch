# Owner(s): ["module: onnx"]

import pytorch_test_common
from torch.onnx._internal.fx.passes import type_promotion
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGeneratedTypePromotionRuleSet(TestCase):
    @pytorch_test_common.skip_in_ci(
        "Reduce noise in CI. "
        "The test serves as a tool to validate if the generated rule set is current. "
    )
    def test_generated_rule_set_is_up_to_date(self):
        generated_set = type_promotion._GENERATED_ATEN_TYPE_PROMOTION_RULE_SET
        latest_set = (
            type_promotion.TypePromotionRuleSetGenerator.generate_from_torch_refs()
        )

        self.assertEqual(generated_set, latest_set)

    def test_initialize_type_promotion_table_from_generated_set_success(self):
        type_promotion.TypePromotionTable()
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests()
