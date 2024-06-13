# Owner(s): ["oncall: jit"]

import os
import sys

import torch
from torch.testing._internal.jit_utils import JitTestCase

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestModules(JitTestCase):
    def test_script_module_with_constants_list(self):
        """
        Test that a module that has __constants__ set to something
        that is not a set can be scripted.
        """

        # torch.nn.Linear has a __constants__ attribute defined
        # and intialized to a list.
        class Net(torch.nn.Linear):
            x: torch.jit.Final[int]

            def __init__(self):
                super().__init__(5, 10)
                self.x = 0

        self.checkModule(Net(), (torch.randn(5),))
