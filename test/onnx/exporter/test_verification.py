# Owner(s): ["module: onnx"]
"""Test the verification module."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _verification
from torch.testing._internal import common_utils


class VerificationInfoTest(common_utils.TestCase):
    def test_from_tensors(self):
        # Test with tensors
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.float32)
        self.assertEqual(verification_info.actual_dtype, torch.float32)

    def test_from_tensors_int(self):
        # Test with int tensors
        expected = torch.tensor([1])
        actual = 1
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor_int", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor_int")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.int64)
        self.assertEqual(verification_info.actual_dtype, torch.int64)


class VerificationInterpreterTest(common_utils.TestCase):
    def test_interpreter_stores_correct_info(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a + b
                return c - 1

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        assert onnx_program is not None
        interpreter = _verification.VerificationInterpreter(onnx_program)
        results = interpreter.run(args)
        torch.testing.assert_close(results, model(*args))
        verification_infos = interpreter.verification_infos
        self.assertEqual(len(verification_infos), 3)
        for info in verification_infos:
            self.assertEqual(info.max_abs_diff, 0)
            self.assertEqual(info.max_rel_diff, 0)


if __name__ == "__main__":
    common_utils.run_tests()
