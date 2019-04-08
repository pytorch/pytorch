from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from common_utils import TestCase
# TODO : Quantizer tests to be integrated with CI once quantizer intf hardened

"""
This is an example observer Module which will be used to collect
stats across batches by running torch script/trace module, from the
observer nodes inserted in the graph. These stats are used to compute
Quantization Parameters. These will be passed to quantizer to be used
as arguments for quant ops in quantization pass.
"""


class ObserverModule:
    def __init__(self, userObserver=None, userCalcQParam=None):
        self.value_stats = {}
        self.qparam_dict = {}
        self.averaging_constant = 0.001
        if userObserver is not None:
            self.observerImpl = userObserver
        if userCalcQParam is not None:
            self.calcQParamImpl = userCalcQParam

    def resetStats(self):
        self.value_stats = {}
        return

    def observerImpl(self, value, stats, avgconstant):
        # Default observer. Stats needed for accumulation
        stats[0] = torch.min(value)
        stats[1] = torch.max(value)
        return value

    def observer(self, value, name):
        if name not in self.value_stats:
            self.value_stats[name] = []
            stats = torch.zeros(2)
        else:
            stats = self.value_stats[name]
        self.observerImpl(value, stats, self.averaging_constant)
        self.value_stats.update({name: stats})
        return value

    def calcQParamImpl(self, qparam_dict, value_stats):
        # Test QParam computation. Used by default for test
        for name in value_stats:
            qparam = torch.zeros(2)
            scaleT = value_stats[name][0]
            scale = scaleT.item()
            zero_pointT = value_stats[name][1]
            zero_point = zero_pointT.item()
            qparam_dict.update({name: (scale, zero_point)})

    def calcQParam(self):
        self.qparam_dict = {}
        self.calcQParamImpl(self.qparam_dict, self.value_stats)

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
            def __init__(self, obsObj=None):
                super(TestM, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv1.weight.data.fill_(1.0)
                self.conv1.bias.data.fill_(0.01)
                self.obsObj = obsObj

            def forward(self, x):
                y = F.relu(self.conv1(x))
                if self.obsObj is not None:
                    self.obsObj.observer(y, "y")
                z = F.max_pool2d(y, 2, 2)
                if self.obsObj is not None:
                    self.obsObj.observer(z, "z")
                return z

        # Test Data
        data = torch.ones(1, 1, 28, 28)

        # Eager mode

        # Create observer object for eager mode
        eagerObserver = ObserverModule()
        eagerM = TestM(obsObj=eagerObserver)

        # Run EagerMode Model and Collect stats
        eagerM.forward(data)
        eagerM.obsObj.calcQParam()

        # Script mode
        scriptM = TestScriptM()

        # Create observer object for script mode
        scriptObserver = ObserverModule()

        # This performs type analysis to identify tensors from other
        # types. This info needed for further quantizer passes
        torch._C._jit_pass_constant_propagation(scriptM.graph)

        # Insert observers
        torch._C._jit_pass_insert_observers(scriptM.graph, scriptObserver.observer)

        # Run ScriptM Model and Collect statistics
        scriptM.forward(data)
        scriptObserver.calcQParam()

        # Compare results for eager and graph mode
        eagerDict = eagerObserver.getQParamDict()
        scriptDict = scriptObserver.getQParamDict()

        self.assertTrue('z' in eagerDict and 'z' in scriptDict)
        self.assertAlmostEqual(eagerDict["z"][0], scriptDict["z"][0], places=15)
        self.assertAlmostEqual(eagerDict["z"][1], scriptDict["z"][1], places=15)

    def test_compare_qparam_eager_script_userobs(self):
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
            def __init__(self, obsObj=None):
                super(TestM, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv1.weight.data.fill_(1.0)
                self.conv1.bias.data.fill_(0.01)
                self.obsObj = obsObj

            def forward(self, x):
                y = F.relu(self.conv1(x))
                if self.obsObj is not None:
                    self.obsObj.observer(y, "y")
                z = F.max_pool2d(y, 2, 2)
                if self.obsObj is not None:
                    self.obsObj.observer(z, "z")
                return z

        # User defined observer and calcQParam
        def userObserverImpl(value, stats, averaging_constant):
            # This implementation averages over collected stats.
            # It is stateless. value and stats will be input from Observer
            stats[0] = (1 - averaging_constant) * stats[0] + averaging_constant * torch.min(value)
            stats[1] = (1 - averaging_constant) * stats[1] + averaging_constant * torch.max(value)
            return value

        def userCalcQParamImpl(qparam_dict, value_stats):
            # User defined QParam computation. This is stateless
            # qparam_dict and value_stats will be input from Observer
            for name in value_stats:
                qparam = torch.zeros(2)
                scaleT = 2.0 * (torch.max(value_stats[name][1], -value_stats[name][0]) / 255.0)
                scale = scaleT.item()
                zero_point = 0
                qparam_dict.update({name: (scale, zero_point)})

        # Test Data
        data = torch.ones(1, 1, 28, 28)

        # Eager mode

        # Create observer object for eager mode
        eagerObserver = ObserverModule(userObserver=userObserverImpl,
                                       userCalcQParam=userCalcQParamImpl)
        eagerM = TestM(obsObj=eagerObserver)

        # Run EagerMode Model and Collect stats
        eagerM.forward(data)
        eagerM.obsObj.calcQParam()

        # Script mode
        scriptM = TestScriptM()

        # Create observer object for script mode
        scriptObserver = ObserverModule(userObserver=userObserverImpl,
                                        userCalcQParam=userCalcQParamImpl)

        # This performs type analysis to identify tensors from other
        # types. This info needed for further quantizer passes
        torch._C._jit_pass_constant_propagation(scriptM.graph)

        # Insert observers
        torch._C._jit_pass_insert_observers(scriptM.graph, scriptObserver.observer)

        # Run ScriptM Model and Collect statistics
        scriptM.forward(data)
        scriptObserver.calcQParam()

        # Compare results for eager and graph mode
        eagerDict = eagerObserver.getQParamDict()
        scriptDict = scriptObserver.getQParamDict()

        self.assertTrue('z' in eagerDict and 'z' in scriptDict)
        self.assertAlmostEqual(eagerDict["z"][0], scriptDict["z"][0], places=15)
        self.assertAlmostEqual(eagerDict["z"][1], scriptDict["z"][1], places=15)
