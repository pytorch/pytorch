import unittest

from auto_classify_skip_reasons import classify_test


class AutoClassifySkipReasonsTest(unittest.TestCase):
    def test_strict_numerics_is_inductor(self):
        self.assertEqual(
            classify_test(
                "requires CUDA and Triton on sm_89, sm_90 or sm_100",
                "inductor.test_strict_numerics",
                "PointwiseStrictNumericsTestCUDA",
                "test_pointwise_backward_add_cuda_float32",
            ),
            "PT2.0 - Inductor",
        )

    def test_accelerator_device_requirement(self):
        for message in (
            "Need at least 4 CUDA devices",
            "Need at least 4 accelerator devices",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    classify_test(
                        message,
                        "distributed.tensor.test_common_rules",
                        "CommonRulesTest",
                        "test_pointwise_rules_broadcasting",
                    ),
                    "Greater than 4 GPU",
                )


if __name__ == "__main__":
    unittest.main()
