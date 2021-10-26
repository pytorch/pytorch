# Owner(s): ["oncall: quantization"]

import torch
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
from torch.fx.experimental.fx2trt.fx2trt import (
    TRTInterpreter,
    InputTensorSpec,
    TRTModule,
)
from torch.ao.quantization import (
    default_qconfig
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    get_tensorrt_backend_config_dict,
)
from torch.ao.quantization._quantize_fx_do_not_use import (
    _convert_fx_do_not_use,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_quantization import NodeSpec as ns
import unittest

def lower_to_trt(model, inputs, shape_ranges):
    """ Lower a quantized model to TensorRT
    """
    assert len(inputs) == 1, "lower_to_trt only works for one input currently"
    model = acc_tracer.trace(model, inputs)  # type: ignore[attr-defined]
    # TODO: test multiple inputs setting and enable multiple inputs
    input_specs = [
        InputTensorSpec(
            torch.Size([-1, *inputs[0].shape[1:]]), torch.float,
            shape_ranges=shape_ranges, has_batch_dim=True)
    ]

    interp = TRTInterpreter(
        model,
        input_specs,
        explicit_batch_dimension=True, explicit_precision=True)
    engine, input_names, output_names = interp.run(fp16_mode=False, int8_mode=True)
    trt_mod = TRTModule(engine, input_names, output_names)
    return trt_mod

@unittest.skipIf(not TEST_CUDA, "gpu is not available.")
class TestQuantizeFxTRTOps(QuantizationTestCase):
    def setUp(self):
        super().setUp()
        self.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.observer.HistogramObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=torch.ao.quantization.default_weight_observer
        )
        self.backend_config_dict = get_tensorrt_backend_config_dict()

    def _test_module(
            self,
            m,
            inputs,
            shape_ranges,
            no_prepare=None,
            no_convert=None):
        """
        Args:
          m: the float module we want to test
          inputs: list of inputs for the module
          shape_ranges: a list of shape_range, where every shape_range is a tuple of
          three tuples
          ((min_input_shape), (optimized_input_shape), (max_input_shape)).
          Each shape_range is used to populate a TensorRT optimization profile.
          e.g. If the input shape varies from (1, 224) to (100, 224) and we want to optimize
          for (25, 224) because it's the most common input shape, then we set shape_ranges to
          ((1, 224), (25, 225), (100, 224))
          no_prepare: node occurrence after prepare
          no_convert: node occurrence after convert
        """
        m = m.eval()
        prepared = prepare_fx(m, {"": self.qconfig}, backend_config_dict=self.backend_config_dict)
        self.checkGraphModuleNodes(prepared, expected_node_occurrence=no_prepare)
        # calibration
        prepared(*inputs)
        quantized = _convert_fx_do_not_use(prepared, is_reference=True)
        self.checkGraphModuleNodes(quantized, expected_node_occurrence=no_convert)
        # lower to trt
        trt_mod = lower_to_trt(quantized, inputs, shape_ranges)
        inputs_cuda = [i.cuda() for i in inputs]
        # make sure it runs
        trt_mod(*inputs_cuda)


    def test_conv(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        conv2d_input = torch.rand(1, 3, 224, 224)
        no_convert = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2
        }
        self._test_module(
            Conv2dModule(),
            [conv2d_input],
            [((1, 3, 224, 224),
              (5, 3, 224, 224),
              (10, 3, 224, 224))],
            no_convert=no_convert)

    def test_linear(self):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_input = torch.rand(8, 5)

        shape_ranges = [
            ((1, 5),
             (5, 5),
             (10, 5))
        ]
        no_convert = {
            ns.call_function(torch.quantize_per_tensor): 2,
            ns.call_method("dequantize"): 2,
        }
        self._test_module(
            LinearModule(),
            [linear_input],
            shape_ranges,
            no_convert=no_convert)

    def test_unsupported_qconfig(self):
        """ Check that we won't quantize the model if the qconfig is not supported
        """
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_module_input = torch.rand(8, 5)

        m = LinearModule().eval()
        trt_unsupported_qconfig = default_qconfig
        prepared = prepare_fx(m, {"": trt_unsupported_qconfig}, backend_config_dict=self.backend_config_dict)
        # calibration
        prepared(linear_module_input)
        quantized = _convert_fx_do_not_use(prepared, is_reference=True)
        node_occurrence = {
            ns.call_function(torch.quantize_per_tensor): 0,
            ns.call_method("dequantize"): 0,
            ns.call_module(torch.nn.Linear): 1,
            ns.call_module(torch.nn.quantized._reference.Linear): 0,
        }
        # check model is not quantized
        self.checkGraphModuleNodes(quantized, expected_node_occurrence=node_occurrence)

if __name__ == "__main__":
    run_tests()
