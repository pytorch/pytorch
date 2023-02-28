# Owner(s): ["module: onnx"]
"""Unit tests on `torch.onnx.symbolic_helper`."""

import torch
from torch.onnx import symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.testing._internal import common_utils


class TestHelperFunctions(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self._initial_training_mode = GLOBALS.training_mode

    def tearDown(self):
        GLOBALS.training_mode = self._initial_training_mode

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.PRESERVE], name="export_mode_is_preserve"
            ),
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.EVAL],
                name="modes_match_op_train_mode_0_export_mode_eval",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.TRAINING],
                name="modes_match_op_train_mode_1_export_mode_training",
            ),
        ],
    )
    def test_check_training_mode_does_not_warn_when(
        self, op_train_mode: int, export_mode: torch.onnx.TrainingMode
    ):
        GLOBALS.training_mode = export_mode
        self.assertNotWarn(
            lambda: symbolic_helper.check_training_mode(op_train_mode, "testop")
        )

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.TRAINING],
                name="modes_do_not_match_op_train_mode_0_export_mode_training",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.EVAL],
                name="modes_do_not_match_op_train_mode_1_export_mode_eval",
            ),
        ],
    )
    def test_check_training_mode_warns_when(
        self,
        op_train_mode: int,
        export_mode: torch.onnx.TrainingMode,
    ):
        with self.assertWarnsRegex(
            UserWarning, f"ONNX export mode is set to {export_mode}"
        ):
            GLOBALS.training_mode = export_mode
            symbolic_helper.check_training_mode(op_train_mode, "testop")


common_utils.instantiate_parametrized_tests(TestHelperFunctions)


if __name__ == "__main__":
    common_utils.run_tests()
