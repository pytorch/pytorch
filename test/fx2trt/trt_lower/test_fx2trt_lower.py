# Owner(s): ["oncall: aiacc"]

import logging
# @manual=//deeplearning/trt/fx2trt_oss:torch_fx2trt
from fx2trt_oss.fx.lower import Lowerer
from fx2trt_oss.fx import LowerSetting
import torch
import torch.fx as fx
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


logger = logging.getLogger(__name__)


class Fx2trtLowerTests(TestCase):
    def test_fx2trt_lower(self):
        mod = _Mod()
        mod_traced = fx.symbolic_trace(mod)
        input = [torch.rand(4)]
        lower = Lowerer.create(LowerSetting())
        mod_lowered = lower(mod_traced, input)
        assert mod_lowered

    def test_lower_with_batchnorm_act_rewrite(self):
        class TestModule(torch.nn.BatchNorm1d):
            def forward(self, x):
                return x


        module = TestModule(2)
        inputs = torch.randn(1)
        lower = Lowerer.create(LowerSetting(ast_rewriter_allow_list={TestModule}))
        result = lower(module, inputs)
        assert result


class _Mod(nn.Module):
    def forward(self, x):
        return (x, 2 * x)

if __name__ == '__main__':
    run_tests()

def test_lower_const_fold():
    class TestModule(torch.nn.Module):
        def __init__(self):
            self.a = torch.randn(1)

        def forward(self, x):
            return (torch.sqrt(x), self.a)

    lower = Lowerer.create(LowerSetting())
    assert lower(TestModule(), [2, 2])
