# Owner(s): ["module: onnx"]

import pytorch_test_common

from torch.onnx._internal.fx.passes import type_promotion
from torch.testing._internal import common_utils


class TestGeneratedTypePromotionRuleSet(common_utils.TestCase):
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

    def test_initialize_type_promotion_table_succeeds(self):
        type_promotion.TypePromotionTable()


if __name__ == "__main__":
    common_utils.run_tests()
