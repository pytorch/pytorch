# Owner(s): ["oncall: jit"]

import io
import os
import sys
import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestUpgraders(JitTestCase):
    def test_populated_upgrader_graph(self):
        @torch.jit.script
        def f():
            return 0

        buffer = io.BytesIO()
        torch.jit.save(f, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size = torch._C._get_upgraders_map_size()
        upgraders_dump = torch._C._dump_upgraders_map()
        # make sure we only populate the upgrader map only once
        # so we load it again and make sure the upgrader map has
        # same content
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size_second_time = torch._C._get_upgraders_map_size()
        upgraders_dump_second_time = torch._C._dump_upgraders_map()
        self.assertTrue(upgraders_size == upgraders_size_second_time)
        self.assertTrue(upgraders_dump == upgraders_dump_second_time)
