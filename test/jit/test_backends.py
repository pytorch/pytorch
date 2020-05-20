from torch.testing._internal.jit_utils import JitTestCase
import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Add the location of the test Python C extension to PYTHONPATH so that the _jit_to_test_backend
# function can be imported and used.
test_python_ext_dir = os.path.join(os.path.dirname(pytorch_test_dir), "build", "lib")
sys.path.append(test_python_ext_dir)

import libtest_jit_python

def to_test_backend(module, method_compile_spec):
    return libtest_jit_python.to_test_backend(module, {"forward": method_compile_spec})


def to_test_backend_multi(module, method_compile_spec):
    return libtest_jit_python.to_test_backend(module, method_compile_spec)


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
    def test_simple(self):
        module = MyModule()
        scripted_module = torch.jit.script(MyModule())

        # Test compile.
        lowered_module = to_test_backend_multi(
            scripted_module._c, {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}})

        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        def compare_py_jit_backend(name, input):
            # Get handles for Python, JIT and backend methods.
            python_method = module.__getattribute__(name)
            jit_method = scripted_module.__getattr__(name)
            backend_method = lowered_module._get_method(name)

            # Run methods.
            python_output = python_method(input, input)
            jit_output = jit_method(input, input)
            backend_output = backend_method(input, input)

            # The answers returned by Python, JIT and to_backend should all match.
            self.assertEqual(python_output, backend_output)
            self.assertEqual(jit_output, backend_output)

        # Test all three module methods.
        compare_py_jit_backend("accum", input)
        compare_py_jit_backend("sub_accum", input)
        compare_py_jit_backend("forward", input)

        # TODO: Test save and load.
