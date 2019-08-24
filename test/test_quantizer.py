from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.jit
from jit_utils import _inline_everything
from torch._jit_internal import Optional
import torch.nn as nn
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from torch.quantization import QuantStub, DeQuantStub, \
    quantize, default_eval_fn, QConfig
# TODO : Quantizer tests to be integrated with CI once quantizer intf hardened

r"""
Default Weight Observer:
Stats needed for accumulation

Arguments:
    value: Tensor to be observed
    stats: Computed stats. Injected by the observer
wrapper

Output:
    stats: Modified stats
"""


def weightObserver(value, stats):
    if stats is None:
        stats = torch.zeros(2)
    stats[0] = torch.min(value)
    stats[1] = torch.max(value)
    return stats


r"""
Default Activation Observer:
This implementation averages over collected stats.

Arguments:
    value: Tensor to be observed
    stats: Computed stats. Injected by the observer
wrapper

Output:
    stats: Modified stats
"""


def activationObserver(value, stats):
    if stats is None:
        stats = torch.zeros(2)
    averaging_constant = 0.001
    stats[0] = (1 - averaging_constant) * stats[0] + \
        averaging_constant * torch.min(value)
    stats[1] = (1 - averaging_constant) * stats[1] + \
        averaging_constant * torch.max(value)
    return stats


r"""
Default QParam computation: This is stateless
value_stats will be input from Observer

Arguments:
    name: Key name in the stats dictionary
wrapper
    value_stats: Stats dict from observer wrapper


Output:
    scale, zero_point
"""


def calcQParamFunc(name, value_stats):
    scaleT = 2.0 * (torch.max(value_stats[name][1],
                    -value_stats[name][0]) / 255.0)
    scale = scaleT.item()
    zero_point = 0
    return scale, zero_point


r"""
 Unified Dictionary for all qparam
"""


def getAllQParamDict(allqparam_dict, quantObj):
    if allqparam_dict is None:
        allqparam_dict = {}
    qparam_dict = quantObj.getQParamDict()
    if qparam_dict is None:
        return
    allqparam_dict.update(qparam_dict)


r"""
This is an example QuantTemplate which will be used to collect
stats across batches by running torch script/trace module, from the
observer nodes inserted in the graph. These stats are used to compute
Quantization Parameters. These will be passed to quantizer to be used
as arguments for quant ops in quantization pass.
"""


class QuantTemplate:
    def __init__(self, qscheme, observerImpl=None, calcQParamImpl=None):
        self.value_stats = {}
        self.qparam_dict = {}
        self.averaging_constant = 0.001
        self.observerImpl = observerImpl
        self.calcQParamImpl = calcQParamImpl
        self.qscheme = qscheme

    def resetStats(self):
        self.value_stats = {}
        return

    def observer(self, value, name):
        if self.observerImpl is None:
            return
        if name not in self.value_stats:
            self.value_stats[name] = []
            stats = None
        else:
            stats = self.value_stats[name]
        stats = self.observerImpl(value, stats)
        self.value_stats.update({name: stats})
        return value

    def calcQParam(self):
        self.qparam_dict = {}
        if self.calcQParamImpl is None:
            return
        for name in self.value_stats:
            # This can change depending on type of quantization which will
            # be known to QuantTemplate object
            scale, zero_point = self.calcQParamImpl(name, self.value_stats)
            self.qparam_dict.update({name: (self.qscheme, scale, zero_point)})

    def getQParam(self, name):
        if name in self.qparam_dict:
            return self.qparam_dict[name]
        else:
            return ()

    def getQParamDict(self):
        return self.qparam_dict


class QuantizerTestCase(TestCase):
    @_inline_everything
    def test_compare_qparam_eager_script_default(self):
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
        torch._C._jit_pass_constant_propagation(get_forward(script_module).graph)

        ScriptedObserver = torch.jit.script(Observer())
        ScriptedWeightObserver = torch.jit.script(WeightObserver())
        script_module._c = torch._C._jit_pass_prepare_quant(script_module._c,
                                                            "forward",
                                                            ScriptedObserver._c, ScriptedWeightObserver._c)
        # Run ScriptM Model and Collect statistics
        get_forward(script_module)(data[0][0])

        # Insert quantize and dequantize calls
        torch._C._jit_pass_insert_quant_dequant(script_module._c, "forward")
        # Note that observer modules are not removed right now
        torch._C._jit_pass_quant_fusion(script_module._c._get_method('forward').graph)
        get_forward(script_module)(data[0][0])
        print(get_forward(script_module).code)
        eager_result = quantized_eager_module(data[0][0])
        script_result = get_forward(script_module)(data[0][0])
        self.assertEqual(eager_result, script_result)

if __name__ == '__main__':
    run_tests()
