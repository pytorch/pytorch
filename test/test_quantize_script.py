import torch
import torch.jit

from torch.quantization._quantize_script import script_qconfig
from torch.quantization._quantize_script import prepare_dynamic_script
from torch.quantization import default_qconfig

from torch.testing._internal.common_utils import run_tests
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import attrs_with_prefix
from torch.testing._internal.jit_utils import JitTestCase

class TestScript(JitTestCase):
    def test_prepare_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        m = torch.jit.script(M())
        m = prepare_dynamic_script(m, {'': default_qconfig})

        # for input of FC for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # for weight
        assert len(attrs_with_prefix(m.fc, '_observer_')) == 1
        FileCheck().check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::GetAttr[name="fc"]') \
                   .check('prim::CallMethod') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)


    def test_prepare_dynamic_child_qconfig(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super(Sub, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        # only quantize child module.
        m = prepare_dynamic_script(m, {'sub.fc': default_qconfig})

        # input of sub for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # not quantized
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1
        FileCheck().check('prim::GetAttr[name="sub') \
                   .check('prim::CallMethod') \
                   .check('Observer = prim::GetAttr[name="_observer_') \
                   .check('prim::CallMethod') \
                   .check_not('Observer = prim::GetAttr[name="_observer_') \
                   .run(m.graph)

if __name__ == "__main__":
    run_tests()
