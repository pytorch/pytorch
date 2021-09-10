import torch
from torch.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
    get_tensorrt_backend_config_dict
)
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.fx2trt import TRTInterpreter, InputTensorSpec, TRTModule
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests

import unittest


@unittest.skipIf(not TEST_CUDA, "gpu is not available.")
class TestQuantizeFxTRT(QuantizationTestCase):
    def test_conv_linear(self):
        class Conv2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv2d(*args)

            def forward(self, x):
                return self.conv(x)

        conv2d_input = torch.rand(1, 3, 224, 224)
        conv2d_module_args = (3, 3, 3)

        m = Conv2d(*conv2d_module_args).eval()
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.HistogramObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=torch.quantization.default_weight_observer
        )
        m = prepare_fx(m, {"": qconfig}, backend_config_dict=get_tensorrt_backend_config_dict())
        # calibration
        m(conv2d_input)
        m = convert_fx(m, is_reference=True)
        # lower to trt
        m = acc_tracer.trace(m, [conv2d_input])  # type: ignore[attr-defined]
        interp = TRTInterpreter(m, [InputTensorSpec(data.shape[1:], torch.float, has_batch_dim=False)])
        engine, input_names, output_names = interp.run(fp16_mode=False, int8_mode=True)
        trt_mod = TRTModule(engine, input_names, output_names)
        # make sure it runs
        trt_mod(conv2d_input.cuda())

if __name__ == '__main__':
    run_tests()
