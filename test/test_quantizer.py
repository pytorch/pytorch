from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from common_utils import TestCase
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
    def test_compare_qparam_eager_script_default(self):
        # Simple test case with conv->relu->maxpool
        class TestScriptM(torch.jit.ScriptModule):
            def __init__(self, init_weight=None):
                super(TestScriptM, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv1.weight.data.fill_(1.0)
                self.conv1.bias.data.fill_(0.01)

            @torch.jit.script_method
            def forward(self, x):
                y = F.relu(self.conv1(x))
                z = F.max_pool2d(y, 2, 2)
                return z

        class TestM(nn.Module):
            def __init__(self, quantObj=None):
                super(TestM, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv1.weight.data.fill_(1.0)
                self.conv1.bias.data.fill_(0.01)
                self.quantObj = quantObj

            def forward(self, x):
                y = F.relu(self.conv1(x))
                if self.quantObj is not None:
                    self.quantObj.observer(y, "y")
                z = F.max_pool2d(y, 2, 2)
                if self.quantObj is not None:
                    self.quantObj.observer(z, "z")
                return z

        # Test Data
        data = torch.ones(1, 1, 28, 28)

        # Eager mode

        # Create QuantConfig object for eager mode
        eagerQuantObj = QuantTemplate(qscheme='per_tensor_quant',
                                      observerImpl=activationObserver,
                                      calcQParamImpl=calcQParamFunc)
        eagerM = TestM(quantObj=eagerQuantObj)

        # Run EagerMode Model and Collect stats
        eagerM.forward(data)
        eagerM.quantObj.calcQParam()

        # Script mode
        scriptM = TestScriptM()

        # Create QuantConfig object for script mode
        activationQuantObj = QuantTemplate(qscheme='per_tensor_quant',
                                           observerImpl=activationObserver,
                                           calcQParamImpl=calcQParamFunc)

        # This performs type analysis to identify tensors from other
        # types. This info needed for further quantizer passes
        torch._C._jit_pass_constant_propagation(scriptM.graph)

        # Insert observers
        torch._C._jit_pass_insert_observers(scriptM._c, "forward", activationQuantObj.observer)

        # Run ScriptM Model and Collect statistics
        scriptM.forward(data)
        activationQuantObj.calcQParam()

        # Compare results for eager and graph mode
        eagerDict = eagerQuantObj.getQParamDict()
        activationDict = activationQuantObj.getQParamDict()

        self.assertTrue('z' in eagerDict and 'z' in activationDict)
        self.assertAlmostEqual(eagerDict["z"][0], activationDict["z"][0], places=15)
        self.assertAlmostEqual(eagerDict["z"][1], activationDict["z"][1], places=15)
