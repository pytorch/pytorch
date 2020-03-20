import torch
import torch.jit
from torch.jit._recursive import wrap_cpp_module
from torch.quantization._quantize_script import script_qconfig
from torch.quantization import default_observer, default_qconfig
from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.jit_utils import attrs_with_prefix
from torch.testing._internal.jit_utils import JitTestCase

class TestScript(JitTestCase):
    def test_insert_observers_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        m = torch.jit.script(M())
        observer = torch.jit.script(default_observer())
        qconfig_dict = {'': script_qconfig(default_qconfig)}
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward", qconfig_dict, False, True))

        # for input of FC for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # for weight
        assert len(attrs_with_prefix(m.fc, '_observer_')) == 1

    def test_insert_observers_child_dynamic_qconfig(self):
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
        qconfig = script_qconfig(default_qconfig)

        qconfig_dict = {
            'sub.fc': qconfig
        }
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, "forward",
                                                                qconfig_dict,
                                                                False, True))
        # input of sub for dynamic quant
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        # not quantized
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1

if __name__ == "__main__":
    run_tests()
