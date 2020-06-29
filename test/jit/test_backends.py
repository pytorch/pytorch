from torch.testing._internal.jit_utils import JitTestCase
import io
import os
import sys

import torch
import torch._C

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")


def to_test_backend(module, method_compile_spec):
    return torch._C._jit_to_test_backend(module, {"forward": method_compile_spec})


def to_test_backend_multi(module, method_compile_spec):
    return torch._C._jit_to_test_backend(module, method_compile_spec)


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h


class TestBackends(JitTestCase):
    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of MyModule.
        self.module = MyModule()
        self.scripted_module = torch.jit.script(MyModule())
        self.lowered_module = to_test_backend_multi(
            self.scripted_module._c, {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}})

    def compare_py_jit_backend(self, name, input):
        """
        This is a helper function for comparing the outputs of self.module (Python), self.scripted_module (JIT)
        and self.lowered_module (backend) when the method named 'name' is invoked using 'input'.
        """
        # Get handles for Python, JIT and backend methods.
        python_method = self.module.__getattribute__(name)
        jit_method = self.scripted_module.__getattr__(name)
        backend_method = self.lowered_module.__getattr__(name)

        # Run methods.
        python_output = python_method(input, input)
        jit_output = jit_method(input, input)
        backend_output = backend_method(input, input)

        # The answers returned by Python, JIT and to_backend should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)

    def test_simple(self):
        """
        This is a simple test that compiles MyModule for the test backend and ensures it produces the correct
        answers for each method.
        """
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.compare_py_jit_backend("accum", input)
        self.compare_py_jit_backend("sub_accum", input)
        self.compare_py_jit_backend("forward", input)

    def test_save_load(self):
        """
        This method tests that a lowered module till produces the same output as a Python module and ScriptModule after
        saving and loading.
        """
        # Save the lowered module.
        buffer = io.BytesIO()
        torch.jit.save(self.lowered_module, buffer)

        # Save the compile spec to compare against the version retrieved after loading.
        pre_compile_spec = self.lowered_module.__getattr__("__method_compile_spec")

        # Load the lowered module.
        buffer.seek(0)
        self.lowered_module = torch.jit.load(buffer)

        # Get the compile spec after loading.
        post_compile_spec = self.lowered_module.__getattr__("__method_compile_spec")

        # Compile specs should match.
        self.assertEqual(pre_compile_spec, post_compile_spec)

        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.compare_py_jit_backend("accum", input)
        self.compare_py_jit_backend("sub_accum", input)
        self.compare_py_jit_backend("forward", input)
