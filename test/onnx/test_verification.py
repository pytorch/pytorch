# Owner(s): ["module: onnx"]

import unittest

import numpy as np
import pytorch_test_common

import torch
from torch.onnx import _experimental, verification
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TestVerification(pytorch_test_common.ExportTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    @common_utils.parametrize(
        "onnx_backend",
        [
            common_utils.subtest(
                verification.OnnxBackend.REFERENCE,
                # TODO: enable this when ONNX submodule catches up to >= 1.13.
                decorators=[unittest.expectedFailure],
            ),
            verification.OnnxBackend.ONNX_RUNTIME_CPU,
        ],
    )
    def test_verify_found_mismatch_when_export_is_wrong(
        self, onnx_backend: verification.OnnxBackend
    ):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def incorrect_add_symbolic_function(g, self, other, alpha):
            return self

        opset_version = 13
        torch.onnx.register_custom_op_symbolic(
            "aten::add", incorrect_add_symbolic_function, opset_version=opset_version
        )

        with self.assertRaisesRegex(AssertionError, ".*Tensor-likes are not close!.*"):
            verification.verify(
                Model(),
                (torch.randn(2, 3),),
                opset_version=opset_version,
                options=verification.VerificationOptions(backend=onnx_backend),
            )

        torch.onnx.unregister_custom_op_symbolic(
            "aten::add", opset_version=opset_version
        )

    def test_check_export_model_diff_returns_diff_when_constant_mismatch(self):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                # tensor.data() will be exported as a constant,
                # leading to wrong model output under different inputs.
                return x + y.data

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups
        )
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Former source location:(.|\n)*"
            r"Latter source location:",
        )

    def test_check_export_model_diff_returns_diff_when_dynamic_controlflow_mismatch(
        self,
    ):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                for i in range(x.size(0)):
                    y = x[i] + y
                return y

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(4, 3), torch.randn(2, 3)), {}),
        ]

        export_options = _experimental.ExportOptions(
            input_names=["x", "y"], dynamic_axes={"x": [0]}
        )
        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups, export_options
        )
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Latter source location:(.|\n)*",
        )

    def test_check_export_model_diff_returns_empty_when_correct_export(self):
        class SupportedModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(
            SupportedModel(), test_input_groups
        )
        self.assertEqual(results, "")

    def test_compare_ort_pytorch_outputs_no_raise_with_acceptable_error_percentage(
        self,
    ):
        ort_outs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pytorch_outs = [torch.tensor([[1.0, 2.0], [3.0, 1.0]])]
        options = verification.VerificationOptions(
            rtol=1e-5,
            atol=1e-6,
            check_shape=True,
            check_dtype=False,
            ignore_none=True,
            acceptable_error_percentage=0.3,
        )
        verification._compare_onnx_pytorch_outputs(
            ort_outs,
            pytorch_outs,
            options,
        )

    def test_compare_ort_pytorch_outputs_raise_without_acceptable_error_percentage(
        self,
    ):
        ort_outs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pytorch_outs = [torch.tensor([[1.0, 2.0], [3.0, 1.0]])]
        options = verification.VerificationOptions(
            rtol=1e-5,
            atol=1e-6,
            check_shape=True,
            check_dtype=False,
            ignore_none=True,
            acceptable_error_percentage=None,
        )
        with self.assertRaises(AssertionError):
            verification._compare_onnx_pytorch_outputs(
                ort_outs,
                pytorch_outs,
                options,
            )


if __name__ == "__main__":
    common_utils.run_tests()
