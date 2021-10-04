import sys
import os
import contextlib
import subprocess
from torch.testing._internal.common_utils import TestCase, run_tests, TemporaryFileName


@contextlib.contextmanager
def _jit_disabled():
    cur_env = os.environ.get("PYTORCH_JIT", "1")
    os.environ["PYTORCH_JIT"] = "0"
    try:
        yield
    finally:
        os.environ["PYTORCH_JIT"] = cur_env


class TestJitDisabled(TestCase):
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """

    def compare_enabled_disabled(self, src):
        """
        Runs the script in `src` with PYTORCH_JIT enabled and disabled and
        compares their stdout for equality.
        """
        # Write `src` out to a temporary so our source inspection logic works
        # correctly.
        with TemporaryFileName() as fname:
            with open(fname, 'w') as f:
                f.write(src)
                with _jit_disabled():
                    out_disabled = subprocess.check_output([
                        sys.executable,
                        fname])
                out_enabled = subprocess.check_output([
                    sys.executable,
                    fname])
                self.assertEqual(out_disabled, out_enabled)

    def test_attribute(self):
        _program_string = """
import torch
class Foo(torch.jit.ScriptModule):
    def __init__(self, x):
        super(Foo, self).__init__()
        self.x = torch.jit.Attribute(x, torch.Tensor)

    def forward(self, input):
        return input

s = Foo(torch.ones(2, 3))
print(s.x)
"""
        self.compare_enabled_disabled(_program_string)

    def test_script_module_construction(self):
        _program_string = """
import torch

class AModule(torch.jit.ScriptModule):
    def __init__(self):
        super(AModule, self).__init__()
    @torch.jit.script_method
    def forward(self, input):
        pass

AModule()
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

    def test_recursive_script(self):
        _program_string = """
import torch

class AModule(torch.nn.Module):
    def __init__(self):
        super(AModule, self).__init__()

    def forward(self, input):
        pass

sm = torch.jit.script(AModule())
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

if __name__ == '__main__':
    run_tests()
