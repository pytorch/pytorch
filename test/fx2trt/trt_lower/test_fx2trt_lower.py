import logging
# @manual=//caffe2:torch_fx2trt
from torch.fx.experimental.fx2trt.lower import Lowerer
from torch.fx.experimental.fx2trt import LowerSetting
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
