# Owner(s): ["module: onnx"]

import unittest

import onnx_test_common

import onnxruntime  # noqa: F401
import parameterized

import torch
from pytorch_test_common import (
    skipIfNoBFloat16Cuda,
    skipIfNoCuda,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)
from test_pytorch_onnx_onnxruntime import (
    _parameterized_class_attrs_and_values,
    MAX_ONNX_OPSET_VERSION,
    MIN_ONNX_OPSET_VERSION,
)
from torch.cuda.amp import autocast
from torch.testing._internal import common_utils


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestONNXRuntime_cuda(onnx_test_common._TestONNXRuntime):
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    def test_gelu_fp16(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        x = torch.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        class LayerNormModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                return self.layer_norm(x)

        x = torch.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(LayerNormModel().cuda(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipIfNoCuda
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        class FusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input, target):
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, dtype=torch.float16, device=torch.device("cuda"))
        target = torch.empty(N, dtype=torch.long, device=torch.device("cuda")).random_(
            0, C
        )

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(FusionModel(), (input, target))

    @skipIfNoCuda
    @skipScriptTest()
    def test_apex_o2(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        try:
            from apex import amp
        except Exception:
            raise unittest.SkipTest("Apex is not available")
        input = torch.randn(3, 3, device=torch.device("cuda"))
        model = amp.initialize(LinearModel(), opt_level="O2")
        self.run_test(model, input)

    # ONNX supports bfloat16 for opsets >= 13
    # Add, Sub and Mul ops don't support bfloat16 cpu in onnxruntime.
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16Cuda
    def test_arithmetic_bfp16(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4, dtype=torch.bfloat16, device=torch.device("cuda"))
                x = x.type_as(y)
                return torch.mul(torch.add(x, y), torch.sub(x, y)).to(
                    dtype=torch.float16
                )

        x = torch.ones(
            3, 4, requires_grad=True, dtype=torch.float16, device=torch.device("cuda")
        )
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(
                    torch.ones(2, 3, device=torch.device("cpu"))
                )
                self.b = torch.nn.Parameter(torch.ones(3, device=torch.device("cuda")))

            def forward(self, x, y):
                return torch.matmul(self.w, x), y + self.b

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        self.run_test(Model(), (x, y))


if __name__ == "__main__":
    common_utils.run_tests()
