# Owner(s): ["module: onnx"]

import contextlib
import io
import tempfile
import unittest

import numpy as np

import onnx
import parameterized
import pytorch_test_common
from packaging import version

import torch
from torch.onnx import _constants, _experimental, verification
from torch.testing._internal import common_utils


class TestVerification(pytorch_test_common.ExportTestCase):
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


@common_utils.instantiate_parametrized_tests
class TestVerificationOnWrongExport(pytorch_test_common.ExportTestCase):
    opset_version: int

    def setUp(self):
        super().setUp()

        def incorrect_add_symbolic_function(g, self, other, alpha):
            return self

        self.opset_version = _constants.ONNX_DEFAULT_OPSET
        torch.onnx.register_custom_op_symbolic(
            "aten::add",
            incorrect_add_symbolic_function,
            opset_version=self.opset_version,
        )

    def tearDown(self):
        super().tearDown()
        torch.onnx.unregister_custom_op_symbolic(
            "aten::add", opset_version=self.opset_version
        )

    @common_utils.parametrize(
        "onnx_backend",
        [
            common_utils.subtest(
                verification.OnnxBackend.REFERENCE,
                decorators=[
                    unittest.skipIf(
                        version.Version(onnx.__version__) < version.Version("1.13"),
                        reason="Reference Python runtime was introduced in 'onnx' 1.13.",
                    )
                ],
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

        with self.assertRaisesRegex(AssertionError, ".*Tensor-likes are not close!.*"):
            verification.verify(
                Model(),
                (torch.randn(2, 3),),
                opset_version=self.opset_version,
                options=verification.VerificationOptions(backend=onnx_backend),
            )


@parameterized.parameterized_class(
    [
        # TODO: enable this when ONNX submodule catches up to >= 1.13.
        # {"onnx_backend": verification.OnnxBackend.ONNX},
        {"onnx_backend": verification.OnnxBackend.ONNX_RUNTIME_CPU},
    ],
    class_name_func=lambda cls, idx, input_dicts: f"{cls.__name__}_{input_dicts['onnx_backend'].name}",
)
class TestFindMismatch(pytorch_test_common.ExportTestCase):
    onnx_backend: verification.OnnxBackend
    opset_version: int
    graph_info: verification.GraphInfo

    def setUp(self):
        super().setUp()
        self.opset_version = _constants.ONNX_DEFAULT_OPSET

        def incorrect_relu_symbolic_function(g, self):
            return g.op("Add", self, g.op("Constant", value_t=torch.tensor(1.0)))

        torch.onnx.register_custom_op_symbolic(
            "aten::relu",
            incorrect_relu_symbolic_function,
            opset_version=self.opset_version,
        )

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(3, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 6),
                )

            def forward(self, x):
                return self.layers(x)

        self.graph_info = verification.find_mismatch(
            Model(),
            (torch.randn(2, 3),),
            opset_version=self.opset_version,
            options=verification.VerificationOptions(backend=self.onnx_backend),
        )

    def tearDown(self):
        super().tearDown()
        torch.onnx.unregister_custom_op_symbolic(
            "aten::relu", opset_version=self.opset_version
        )
        delattr(self, "opset_version")
        delattr(self, "graph_info")

    def test_pretty_print_tree_visualizes_mismatch(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.graph_info.pretty_print_tree()
        self.assertExpected(f.getvalue())

    def test_preserve_mismatch_source_location(self):
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()

        self.assertTrue(len(mismatch_leaves) > 0)

        for leaf_info in mismatch_leaves:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                leaf_info.pretty_print_mismatch(graph=True)
            self.assertRegex(
                f.getvalue(),
                r"(.|\n)*" r"aten::relu.*/torch/nn/functional.py:[0-9]+(.|\n)*",
            )

    def test_find_all_mismatch_operators(self):
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()

        self.assertEqual(len(mismatch_leaves), 2)

        for leaf_info in mismatch_leaves:
            self.assertEqual(leaf_info.essential_node_count(), 1)
            self.assertEqual(leaf_info.essential_node_kinds(), {"aten::relu"})

    def test_find_mismatch_prints_correct_info_when_no_mismatch(self):
        self.maxDiff = None

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            verification.find_mismatch(
                Model(),
                (torch.randn(2, 3),),
                opset_version=self.opset_version,
                options=verification.VerificationOptions(backend=self.onnx_backend),
            )
        self.assertExpected(f.getvalue())

    def test_export_repro_for_mismatch(self):
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()
        self.assertTrue(len(mismatch_leaves) > 0)
        leaf_info = mismatch_leaves[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            repro_dir = leaf_info.export_repro(temp_dir)

            with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
                options = verification.VerificationOptions(backend=self.onnx_backend)
                verification.OnnxTestCaseRepro(repro_dir).validate(options)


if __name__ == "__main__":
    common_utils.run_tests()
