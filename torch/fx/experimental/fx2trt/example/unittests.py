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
from torch.testing._internal.common_quantization import NodeSpec as ns

import unittest

def lower_to_trt(model, sample_input, shape_ranges):
    model = acc_tracer.trace(model, [sample_input])  # type: ignore[attr-defined]
    interp = TRTInterpreter(
        model,
        [InputTensorSpec(
            torch.Size([-1, *sample_input.shape[1:]]), torch.float,
            shape_ranges=shape_ranges, has_batch_dim=True)],
        explicit_batch_dimension=True, explicit_precision=True)
    engine, input_names, output_names = interp.run(fp16_mode=False, int8_mode=True)
    trt_mod = TRTModule(engine, input_names, output_names)
    return trt_mod



@unittest.skipIf(not TEST_CUDA, "gpu is not available.")
class TestQuantizeFxTRT(QuantizationTestCase):
    def test_conv(self):
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
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        # lower to trt
        trt_mod = lower_to_trt(m, conv2d_input, [((1, 3, 224, 224), (5, 3, 224, 224), (10, 3, 224, 224))])
        # make sure it runs
        trt_mod(conv2d_input.cuda())

    def test_linear(self):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_module_input = torch.rand(8, 5)

        m = LinearModule().eval()
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.HistogramObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=torch.quantization.default_weight_observer
        )
        m = prepare_fx(m, {"": qconfig}, backend_config_dict=get_tensorrt_backend_config_dict())
        print(m)
        # calibration
        m(linear_module_input)
        m = convert_fx(m, is_reference=True)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 1,
            ns.call_method("dequantize"): 1
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
        print(m)
        # lower to trt
        trt_mod = lower_to_trt(
            m,
            linear_module_input,
            [((1, *linear_module_input.shape[1:]),
              (5, *linear_module_input.shape[1:]),
              (10, *linear_module_input.shape[1:]))])
        # make sure it runs
        trt_mod(linear_module_input.cuda())

if __name__ == '__main__':
    run_tests()
