from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import torch.jit
from jit_utils import _tmp_donotuse_dont_inline_everything
from torch._jit_internal import Optional
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_quantization import NestedModel, AnnotatedNestedModel
from torch.quantization import QuantStub, DeQuantStub, \
    quantize, default_eval_fn, QConfig

class Observer(torch.nn.Module):
    __annotations__ = {'scale' : Optional[torch.Tensor], 'zero_point': Optional[torch.Tensor]}

    def __init__(self):
        super(Observer, self).__init__()
        self.dtype = torch.quint8
        self.qscheme = torch.per_tensor_affine
        self.scale, self.zero_point = None, None

    def forward(self, x):
        self.scale = torch.tensor([2.0])
        self.zero_point = torch.tensor([3])
        return x

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

class WeightObserver(Observer):
    def __init__(self):
        super(WeightObserver, self).__init__()
        self.dtype = torch.qint8

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
@unittest.skip("temoprarily disable the test")
class QuantizerTestCase(TestCase):
    @_tmp_donotuse_dont_inline_everything
    def test_default(self):
        class TestM(nn.Module):
            def __init__(self, qconfig):
                super(TestM, self).__init__()
                self.conv = nn.Conv2d(3, 1, 3).float()
                self.conv.weight.data.fill_(1.0)
                self.conv.bias.data.fill_(0.01)
                self.qconfig = qconfig
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                return self.dequant(self.conv(self.quant(x)))

        class TestScriptM(torch.jit.ScriptModule):
            def __init__(self):
                super(TestScriptM, self).__init__()
                self.conv = nn.Conv2d(3, 1, 3).float()
                self.conv.bias.data.fill_(0.01)

            @torch.jit.script_method
            def forward(self, x):
                y = self.conv(x)
                return y

        # Test Data
        data = [(torch.randn(10, 3, 10, 10, dtype=torch.float), 1)]

        # Eager mode
        fake_qconfig = QConfig(activation=Observer, weight=WeightObserver)
        eager_module = TestM(fake_qconfig)
        # Script mode
        script_module = TestScriptM()
        script_module.conv.weight = torch.nn.Parameter(eager_module.conv.weight.detach())
        quantized_eager_module = quantize(eager_module, default_eval_fn, data)

        def get_forward(m):
            return m._c._get_method('forward')
        # TODO: test jit.script as well
        ScriptedObserver = torch.jit.script(Observer())
        ScriptedWeightObserver = torch.jit.script(WeightObserver())
        qconfig_dict = {
            '':
            QConfig(
                activation=ScriptedObserver._c,
                weight=ScriptedWeightObserver._c)
        }
        torch._C._jit_pass_insert_observers(script_module._c,
                                            "forward",
                                            qconfig_dict)
        # Run ScriptM Model and Collect statistics
        get_forward(script_module)(data[0][0])

        # Insert quantize and dequantize calls
        script_module._c = torch._C._jit_pass_insert_quant_dequant(script_module._c, "forward")
        # Note that observer modules are not removed right now
        torch._C._jit_pass_quant_fusion(script_module._c._get_method('forward').graph)
        get_forward(script_module)(data[0][0])
        eager_result = quantized_eager_module(data[0][0])
        script_result = get_forward(script_module)(data[0][0])
        self.assertEqual(eager_result, script_result)

    @_tmp_donotuse_dont_inline_everything
    def test_qconfig_dict(self):
        data = [(torch.randn(10, 5, dtype=torch.float) * 20, 1)]

        # Eager mode
        qconfig = QConfig(activation=Observer, weight=WeightObserver)
        eager_module = AnnotatedNestedModel()
        eager_module.fc3.qconfig = qconfig
        eager_module.sub2.fc1.qconfig = qconfig
        # Assign weights
        eager_module.sub1.fc.weight.data.fill_(1.0)
        eager_module.sub2.fc1.module.weight.data.fill_(1.0)
        eager_module.sub2.fc2.weight.data.fill_(1.0)
        eager_module.fc3.module.weight.data.fill_(1.0)

        script_module = torch.jit.script(NestedModel())
        # Copy weights for eager_module
        script_module.sub1.fc.weight = eager_module.sub1.fc.weight
        script_module.sub2.fc1.weight = eager_module.sub2.fc1.module.weight
        script_module.sub2.fc2.weight = eager_module.sub2.fc2.weight
        script_module.fc3.weight = eager_module.fc3.module.weight

        # Quantize eager module
        quantized_eager_module = quantize(eager_module, default_eval_fn, data)

        def get_forward(m):
            return m._c._get_method('forward')

        # Quantize script_module
        torch._C._jit_pass_constant_propagation(get_forward(script_module).graph)

        ScriptedObserver = torch.jit.script(Observer())
        ScriptedWeightObserver = torch.jit.script(WeightObserver())
        scripted_qconfig = QConfig(
            activation=ScriptedObserver._c,
            weight=ScriptedWeightObserver._c)
        qconfig_dict = {
            'sub2.fc1': scripted_qconfig,
            'fc3': scripted_qconfig
        }
        torch._C._jit_pass_insert_observers(script_module._c,
                                            "forward",
                                            qconfig_dict)

        # Run script_module and Collect statistics
        get_forward(script_module)(data[0][0])

        # Insert quantize and dequantize calls
        script_module._c = torch._C._jit_pass_insert_quant_dequant(script_module._c, "forward")
        # Note that observer modules are not removed right now
        torch._C._jit_pass_quant_fusion(script_module._c._get_method('forward').graph)
        get_forward(script_module)(data[0][0])
        eager_result = quantized_eager_module(data[0][0])
        script_result = get_forward(script_module)(data[0][0])
        self.assertEqual(eager_result, script_result)

if __name__ == '__main__':
    run_tests()
