# Owner(s): ["module: onnx"]

import unittest

import torch
from torch.onnx import _experimental, verification


class TestVerification(unittest.TestCase):
    def test_check_jit_model_diff(self):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                # tensor.data() will be exported as a constant,
                # leading to wrong model output under different inputs.
                return x + y.data

        test_inputs = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(UnexportableModel(), test_inputs)
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Reference source location:(.|\n)*"
            r"Check source location:",
        )

    def test_check_export_model_diff(self):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                # tensor.data() will be exported as a constant,
                # leading to wrong model output under different inputs.
                return x + y.data

        test_inputs = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification._check_onnx_model_diff(
            UnexportableModel(), test_inputs, _experimental.ExportOptions()
        )
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"onnx::Constant(.|\n)*"
            r"Reference source location:(.|\n)*"
            r"Check source location:",
        )

    def test_check_jit_model_no_diff(self):
        class SupportedModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        test_input_sets = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(
            SupportedModel(), test_input_sets
        )
        self.assertEqual(results, "")
