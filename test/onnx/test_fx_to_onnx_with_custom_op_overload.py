# Owner(s): ["module: onnx"]
import onnxscript
import pytorch_test_common
import torch
import torch.onnx._internal.fx.custom_operator
from torch.onnx._internal import fx
from torch.testing._internal import common_utils


class TestFxToOnnx(pytorch_test_common.ExportTestCase):
    def setUp(self):
        super().setUp()
        self.opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET

    def test_simple_function(self):
        # Steps to put custom op into onnx model
        #  1. Define a custom function and register it as torch._ops.OpOverload so that
        #     this function can pass through PyTorch 2.0's FX-generation pipeline
        #     (e.g., dynamo.export). Otherwise, the custom function will be traced
        #     into (or decomposed) and exporter won't see any custom functions.
        #  2. Write a custom exporter using ONNX Script for this custom function.
        #  3. Register the custom exporter to the custom function (type: torch._ops.OpOverload).

        # Step 1
        @fx.custom_operator._register_onnx_custom_op_overload(
            "torch_custom_op(Tensor x) -> Tensor"
        )
        def custom_op(x):
            return x * 2 + 1

        CUSTOM_OPSET = onnxscript.values.Opset(domain="com.custom", version=1)

        # Step 2
        @onnxscript.script(opset=CUSTOM_OPSET)
        def custom_op_exporter(x):
            return CUSTOM_OPSET.onnx_custom_op(x)

        # Step 3
        fx.custom_operator._register_exporter_for_op_overload(
            torch.ops.onnx_custom.torch_custom_op.default,
            "custom_op",
            custom_op_exporter,
        )

        def f(x):
            return 2 + custom_op(x) * 3

        onnx_model = fx.export_after_normalizing_args_and_kwargs(
            f, torch.randn(3), use_binary_format=False
        )

        self.assertIn("onnx_custom_op", str(onnx_model))


if __name__ == "__main__":
    common_utils.run_tests()
