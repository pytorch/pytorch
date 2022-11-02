# Owner(s): ["module: onnx"]

"""Test the support on onnxscript in PyTorch-ONNX converter."""
import io
from typing import List

import onnx
import onnxscript
import torch
from onnxscript.onnx_types import FLOAT
from torch.onnx._internal import jit_utils
from torch.testing._internal import common_utils


class TestONNXScriptExport(common_utils.TestCase):

    # opset version is
    # 1. local function is supported after opset 15
    # 2. onnx-script requires users to determine opset in local function
    opset_version = 15

    def test_onnxscript_registration_with_multiple_models(self):

        from onnxscript.onnx_opset import opset15 as op

        # 1. Register Selu onnxscript function as custom Op
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

        @onnxscript.script(custom_opset)
        def Selu(X):
            # TODO: onnx/ort doesn't support default values for now
            # move this when they do
            alpha = 1.67326  # auto wrapped as Constants
            gamma = 1.0507
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g: jit_utils.GraphContext, X):
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=self.opset_version,
        )

        # 2. Register layer_norm onnxscript function as custom Op
        @onnxscript.script(custom_opset)
        def layer_norm(
            X, axes: List[int], weight: FLOAT[...], bias: FLOAT[...], eps: float
        ):
            mean = op.ReduceMean(X, axes=axes)
            D = X - mean  # op.Sub(X, mean)
            DD = D * D  # op.Mul(D, D)
            var = op.ReduceMean(DD, axes=axes)
            vareps = var + eps  # op.Add(var, eps)
            stddev = op.Sqrt(vareps)
            invstddev = op.Reciprocal(stddev)
            normalized = D * invstddev  # op.Mul(D, invstddev)
            normalizedw = op.CastLike(
                normalized, weight
            )  # Type issue if missing this Op
            normalizedscaled = normalizedw * weight  # op.Mul(normalized, weight)
            return normalizedscaled + bias

        @torch.onnx.symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
        def custom_layer_norm(
            g, input, normalized_shape, weight, bias, eps, cudnn_enable
        ):
            # TODO: move the comprehension into local function once
            # it's supported by onnxscript
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            return g.onnxscript_op(
                layer_norm, input, weight, bias, axes_i=axes, eps_f=eps
            ).setType(input.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::layer_norm",
            symbolic_fn=custom_layer_norm,
            opset_version=self.opset_version,
        )

        # 3. export two models
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model_selu = torch.nn.SELU()
        selu_onnx = io.BytesIO()
        torch.onnx.export(model_selu, x, selu_onnx, opset_version=self.opset_version)

        N, C = 3, 4
        y = torch.randn(N, C)
        model_layer_norm = torch.nn.LayerNorm(C)
        layer_norm_onnx = io.BytesIO()
        torch.onnx.export(
            model_layer_norm, y, layer_norm_onnx, opset_version=self.opset_version
        )

        # 4. test on models
        selu_proto = onnx.load(io.BytesIO(selu_onnx.getvalue()))
        layer_norm_proto = onnx.load(io.BytesIO(layer_norm_onnx.getvalue()))

        self.assertEqual(len(selu_proto.functions), 1)
        self.assertEqual(len(layer_norm_proto.functions), 1)
        self.assertEqual(selu_proto.functions[0].name, "Selu")
        self.assertEqual(layer_norm_proto.functions[0].name, "layer_norm")
