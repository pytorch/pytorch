import unittest
import sys
import os
import contextlib
import subprocess


@contextlib.contextmanager
def _jit_disabled():
    cur_env = os.environ.get("PYTORCH_JIT", "1")
    os.environ["PYTORCH_JIT"] = "0"
    try:
        yield
    finally:
        os.environ["PYTORCH_JIT"] = cur_env

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

class TestJitDisabled(unittest.TestCase):
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """
    def test_attribute(self):
        with _jit_disabled():
            out_disabled = subprocess.check_output([
                sys.executable,
                "-c",
                _program_string])
        out_enabled = subprocess.check_output([
            sys.executable,
            "-c",
            _program_string])
        self.assertEqual(out_disabled, out_enabled)


if __name__ == '__main__':
    unittest.main()
