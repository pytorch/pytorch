# Owner(s): ["oncall: jit"]

import os
import sys

import torch
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
    def test_aten_div_at_3(self):
        model_path = pytorch_test_dir + "/cpp/jit/div_at_version_3.pt"
        loaded_model = torch.jit.load(model_path)
        torch._C._jit_pass_replace_upgraders(loaded_model.graph)
        FileCheck().check("prim::If").run(loaded_model.graph)
        FileCheck().check_count("aten::div", 2).run(loaded_model.graph)
