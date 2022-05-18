# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestOpDecompositions(JitTestCase):
    def test_op_decomposition(self):
        def foo(x):
            return torch.var(x, unbiased=True)

        # TODO: more robust testing
        foo_s = torch.jit.script(foo)
        FileCheck().check("aten::var").run(foo_s.graph)
        torch._C._jit_pass_run_decompositions(foo_s.graph)
        inp = torch.rand([10, 10])
        self.assertEqual(foo(inp), foo_s(inp))
        FileCheck().check_not("aten::var").run(foo_s.graph)
