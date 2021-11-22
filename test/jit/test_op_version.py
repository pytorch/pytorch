# Owner(s): ["oncall: jit"]

import os
import sys
import io
import inspect
import unittest

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestOpVersion(JitTestCase):

    def setUp(self):
        super(TestOpVersion, self).setUp()
        # create dummy pt file
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor):
                c = torch.div(a, b)
                return c
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(M), buffer)
