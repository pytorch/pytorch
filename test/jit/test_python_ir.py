import torch
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestPythonIr(JitTestCase):
    def test_param_strides(self):
        def trace_me(arg):
            return arg
        t = torch.zeros(1, 3, 16, 16)
        traced = torch.jit.trace(trace_me, t)
        value = list(traced.graph.param_node().outputs())[0]
        real_strides = list(t.stride())
        type_strides = value.type().strides()
        self.assertEqual(real_strides, type_strides)
