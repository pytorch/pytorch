from torch.testing._internal.jit_utils import JitTestCase
import os
import sys

import torch
import torch._C
from torch.testing._internal.common_utils import TEST_WITH_ROCM, skipIfRocm

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


def to_test_backend(module, method_compile_spec):
    return torch._C._jit_to_backend("test_backend", module, {"forward": method_compile_spec})


def to_test_backend_multi(module, method_compile_spec):
    return torch._C._jit_to_backend("test_backend", module, method_compile_spec)


class BasicModule(torch.nn.Module):
    """
    A simple Module used to test to_backend lowering machinery.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h


class JitBackendTestCase(JitTestCase):
    """
    A common base class for JIT backend tests that contains common utility
    functions for output comparison and serialization/deserialization.
    """

    def setUp(self):
        super().setUp()
        # Subclasses are expected to set up three variables in their setUp methods:
        # module - a regular, Python version of the module being tested
        # scripted_module - a scripted version of module
        # lowered_modle - a version of module lowered to a backend

    def check_function(self, function_name, input):
        """
        Check that the function named 'function_name' produces the same output using
        Python, regular JIT and the backend for the given 'input'.
        """
        # Get handles for Python, JIT and backend methods.
        python_method = self.module.__getattribute__(function_name)
        jit_method = self.scripted_module.__getattr__(function_name)
        backend_method = self.lowered_module.__getattr__(function_name)

        # Run methods.
        python_output = python_method(input, input)
        jit_output = jit_method(input, input)
        backend_output = backend_method(input, input)

        # The answers returned by Python, JIT and to_backend should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)

    def save_load(self):
        """
        Save and load the lowered module.
        """
        self.lowered_module = self.getExportImportCopy(self.lowered_module)


class BasicModuleTest(JitBackendTestCase):
    """
    Tests for BasicModule.
    """

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()
        self.scripted_module = torch.jit.script(BasicModule())
        self.lowered_module = to_test_backend_multi(
            self.scripted_module._c,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.check_function("accum", input)
        self.check_function("sub_accum", input)
        self.check_function("forward", input)

    @skipIfRocm
    def test_save_load(self):
        # Lowered module should produce the same outputs.
        self.test_execution()

        # Save the compile spec to compare against the version retrieved after loading.
        pre_compile_spec = self.lowered_module.__getattr__("__method_compile_spec")

        # Save and load the lowered module.
        self.save_load()

        # Get the compile spec after loading.
        post_compile_spec = self.lowered_module.__getattr__("__method_compile_spec")

        # Compile specs should match.
        self.assertEqual(pre_compile_spec, post_compile_spec)

        # Loaded module should produce the same outputs.
        self.test_execution()


class NestedModuleTest(JitBackendTestCase):
    """
    Tests for NestedModule that check that a module lowered to a backend can be used
    as a submodule.
    """
    class NestedModule(torch.nn.Module):
        """
        A Module with one submodule that is used to test that lowered Modules
        can be used as submodules.
        """

        def __init__(self, submodule):
            super().__init__()
            self.submodule = submodule

        def forward(self, x, h):
            return self.submodule.forward(x, h)

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of NestedModule.
        # Both modules in self.module are regular Python modules.
        self.module = NestedModuleTest.NestedModule(BasicModule())
        # Both modules in self.scripted_module are ScriptModules.
        self.scripted_module = torch.jit.script(NestedModuleTest.NestedModule(BasicModule()))
        lowered_module = to_test_backend_multi(
            self.scripted_module._c, {"forward": {"": ""}}
        )
        # self.lowered_module is a ScriptModule, but its submodule is a lowered module.
        self.lowered_module = torch.jit.script(NestedModuleTest.NestedModule(lowered_module))

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test forward.
        self.check_function("forward", input)

    def test_save_load(self):
        # Lowered module should produce the same outputs.
        self.test_execution()

        # Save and load the lowered module.
        self.save_load()

        # Loaded module should produce the same outputs.
        self.test_execution()


class TestBackends(JitTestCase):
    """
    This class wraps and invokes all subclasses of JitBackendTestCase so that each one
    does not have to be individually imported in test_jit.py.
    """

    def __init__(self, name):
        super().__init__(name)
        self.basic_module_test = BasicModuleTest(name)
        self.nested_module_test = NestedModuleTest(name)

    def setUp(self):
        super().setUp()
        if not TEST_WITH_ROCM:
            self.basic_module_test.setUp()
            self.nested_module_test.setUp()

    @skipIfRocm
    def test_execution(self):
        self.basic_module_test.test_execution()
        self.nested_module_test.test_execution()

    @skipIfRocm
    def test_save_load(self):
        self.basic_module_test.test_save_load()
        self.nested_module_test.test_save_load()
