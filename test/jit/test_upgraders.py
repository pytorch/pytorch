# Owner(s): ["oncall: jit"]

import io
import os
import sys
import tempfile
import torch
from typing import List
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests that Python slice class is supported in TorchScript
class TestUpgraders(JitTestCase):
    def test_populated_upgrader_graph(self):
        @torch.jit.script
        def f():
            return 0

        buffer = io.BytesIO()
        torch.jit.save(f, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size = torch._C.get_upgraders_map_size()
        # make sure we only populate the upgrader map only once
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size_v2 = torch._C.get_upgraders_map_size()
        self.assertTrue(upgraders_size == upgraders_size_v2)
        self.assertTrue(upgraders_size > 0)
