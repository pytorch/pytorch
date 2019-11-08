import os
import sys
from typing import List

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestBuiltins(JitTestCase):
    """
    Tests for TorchScript support of Python builtin functions.
    """
    def test_has_attr(self):
        class HasA(torch.nn.Module):
            def __init__(self):
                super(HasA, self).__init__()
                self.a = 0

        class HasB(torch.nn.Module):
            def __init__(self):
                super(HasB, self).__init__()
                self.b = 1

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # use a list to encode hasattr results
                l = torch.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # actually retrieve the attr to test static refinement
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mod = torch.nn.Linear(1, 1)

            def forward(self, name):
                # not allowed, `name` must be static.
                return hasattr(self.mod, name)

        with self.assertRaisesRegex(RuntimeError, "hasattr"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            def __init__(self):
                super(Mod, self).__init__()

            def forward(self, name):
                # not allowed, `torch.rand` is not a class type
                return hasattr(torch.rand(2, 3), name)

        with self.assertRaisesRegex(RuntimeError, "hasattr"):
            torch.jit.script(Mod())
