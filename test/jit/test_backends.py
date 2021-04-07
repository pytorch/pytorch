from torch.testing._internal.jit_utils import JitTestCase
import io
import os
import sys
import unittest

import torch
import torch._C
from torch.testing import FileCheck
from pathlib import Path

from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    TEST_WITH_ROCM,
    skipIfRocm,
)
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


def to_test_backend_selective(module, method_compile_spec, submodules):
    def _to_test_backend(module):
        return to_test_backend(module, method_compile_spec)

    return torch._C._jit_to_backend_selective(module, _to_test_backend, submodules)


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


# This is ignored in IS_WINDOWS or IS_MACOS cases. Hence we need the one in TestBackends.
@unittest.skipIf(TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
                 "Non-portable load_library call used in test")
class JitBackendTestCase(JitTestCase):
    """
    A common base class for JIT backend tests that contains common utility
    functions for output comparison and serialization/deserialization.
    """

    def setUp(self):
        super().setUp()
        torch_root = Path(__file__).resolve().parent.parent.parent
        p = torch_root / 'build' / 'lib' / 'libjitbackend_test.so'
        torch.ops.load_library(str(p))
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
        python_output = python_method(*input)
        jit_output = jit_method(*input)
        backend_output = backend_method(*input)

        # The answers returned by Python, JIT and to_backend should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)

    def save_load(self):
        """
        Save and load the lowered module.
        """
        self.lowered_module = self.getExportImportCopy(self.lowered_module)

    def test_execution(self):
        """
        Stub for correctness tests.
        """
        pass

    def test_save_load(self):
        """
        Stub for serialization tests.
        """
        pass

    def test_errors(self):
        """
        Stub for testing error checking.
        """
        pass


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
            self.scripted_module,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.check_function("accum", (input, input))
        self.check_function("sub_accum", (input, input))
        self.check_function("forward", (input, input))

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


class BasicModuleUnavailableTest(JitBackendTestCase):
    """
    Tests for BasicModule with a backend that is not available.
    Fundamentally:
      * _jit_to_backend is successful.
      * Execution fails with an exception.
      * Saving is successful.
      * Loading fails with an exception.
    """

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()
        self.scripted_module = torch.jit.script(BasicModule())
        self.lowered_module = torch._C._jit_to_backend(
            "test_backend_unavailable",
            self.scripted_module,
            {"forward": {"": ""}},
        )

    def test_execution(self):
        # Test execution with backend fails because the backend that is not available.
        input = torch.randn(5)

        # Test exception is thrown.
        with self.assertRaisesRegex(Exception, r"Backend is not available."):
            backend_method = self.lowered_module.__getattr__("forward")
            backend_output = backend_method(*(input, input))

    @skipIfRocm
    def test_save_load(self):
        # Test that saving the lowered module is OK but loading fails because the backend is not available.
        buffer = io.BytesIO()
        torch.jit.save(self.lowered_module, buffer)
        buffer.seek(0)
        with self.assertRaisesRegex(Exception, r"Backend is not available."):
            imported = torch.jit.load(buffer)


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

        # First, script another instance of NestedModule with share_types=False so that it can be
        # selectively lowered without modifying the type of self.scripted_module.
        lowered_module = to_test_backend_multi(
            torch.jit.script(BasicModule()),
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )
        # self.lowered_module is a ScriptModule, but its submodule is a lowered module.
        self.lowered_module = torch.jit.script(NestedModuleTest.NestedModule(lowered_module))

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test forward.
        self.check_function("forward", (input, input))

    def test_save_load(self):
        # Lowered module should produce the same outputs.
        self.test_execution()

        # Save and load the lowered module.
        self.save_load()

        # Loaded module should produce the same outputs.
        self.test_execution()


class SelectiveLoweringTest(JitBackendTestCase):
    """
    Tests for the selective lowering API.
    """
    class OuterModule(torch.nn.Module):
        def __init__(self, sub1, sub2, other):
            super().__init__()
            self.sub1 = sub1
            self.sub2 = sub2
            self.other = other

        def forward(self, x, y):
            # Call the module that will be lowered directly to test
            # type remapping in modules that are not its parent.
            a, b = self.sub1.submodule.forward(x, y)
            c, d = self.sub2.forward(x, y)
            e, f = self.other.forward(x, y)
            return a + c + e, b + d + f

    class MiddleModule(torch.nn.Module):
        def __init__(self, submodule):
            super().__init__()
            self.submodule = submodule

        def forward(self, x, y):
            return self.submodule.forward(x, y)

    def setUp(self):
        super().setUp()
        OuterModule = SelectiveLoweringTest.OuterModule
        MiddleModule = SelectiveLoweringTest.MiddleModule

        def script_without_type_sharing(mod):
            return torch.jit._recursive.create_script_module(mod, torch.jit._recursive.infer_methods_to_compile, share_types=False)
        # Create Python, JIT and backend versions of a hierarchy that looks like this:
        #                 --------- OuterModule --------
        #                 |              |              |
        #           MiddleModule    MiddleModule   MiddleModule
        #                |               |              |
        #           BasicModule     BasicModule    BasicModule
        #
        # Two BasicModules will be lowered and the third will not.
        self.module = OuterModule(MiddleModule(BasicModule()), MiddleModule(BasicModule()), MiddleModule(BasicModule()))
        self.scripted_module = script_without_type_sharing(OuterModule(MiddleModule(
            BasicModule()), MiddleModule(BasicModule()), MiddleModule(BasicModule())))
        self.lowered_module = script_without_type_sharing(OuterModule(MiddleModule(
            BasicModule()), MiddleModule(BasicModule()), MiddleModule(BasicModule())))
        self.lowered_module = to_test_backend_selective(self.lowered_module, {"forward": ""}, [
                                                        "sub1.submodule", "sub2.submodule"])

    def test_execution(self):
        input = torch.randn(5)
        self.check_function("forward", (input, input))

        self.test_selective_lowering_type_remap()

    def test_save_load(self):
        self.test_execution()
        self.save_load()
        self.test_execution()

        self.test_selective_lowering_type_remap()

    def test_selective_lowering_type_remap(self):
        """
        Check that type remapping and replacement occurred during selective lowering.
        """
        # Check that self.lowered_module was not lowered, but that it does contain test_backendLoweredModule due to it
        # calling the lowered module directly.
        FileCheck() \
            .check("OuterModule") \
            .check("BasicModule") \
            .run(self.scripted_module.graph)
        FileCheck() \
            .check("OuterModule") \
            .check_not("__torch__.torch.classes.__backends__.test_backend") \
            .check("test_backendLoweredModule") \
            .run(self.lowered_module.graph)

        # Check that self.lowered_module.sub1/sub2 were not lowered but that BasicModule has been replaced in their graphs.
        FileCheck() \
            .check("MiddleModule") \
            .check("BasicModule") \
            .check_not("test_backendLoweredModule") \
            .run(self.scripted_module.sub1.graph)
        FileCheck() \
            .check("MiddleModule") \
            .check_not("__torch__.torch.classes.__backends__.test_backend") \
            .check("test_backendLoweredModule") \
            .check_not("BasicModule") \
            .run(self.lowered_module.sub1.graph)

        FileCheck() \
            .check("MiddleModule") \
            .check("BasicModule") \
            .check_not("test_backendLoweredModule") \
            .run(self.scripted_module.sub2.graph)
        FileCheck() \
            .check("MiddleModule") \
            .check_not("__torch__.torch.classes.__backends__.test_backend") \
            .check("test_backendLoweredModule") \
            .check_not("BasicModule") \
            .run(self.lowered_module.sub2.graph)

        # Check that self.lowered_module.sub1/sub2.submodule were lowered. Its graph should mention
        # __torch__.torch.classes.__backends__.test_backend, the TorchBind class for executing functions
        # on the test JIT backend.
        FileCheck() \
            .check("test_backendLoweredModule") \
            .check_not("BasicModule") \
            .check("__torch__.torch.classes.__backends__.test_backend") \
            .run(self.lowered_module.sub1.submodule.graph)

        FileCheck() \
            .check("test_backendLoweredModule") \
            .check_not("BasicModule") \
            .check("__torch__.torch.classes.__backends__.test_backend") \
            .run(self.lowered_module.sub2.submodule.graph)

        # Check that self.other and self.other.submodule have been left untouched by the selective lowering process.
        FileCheck() \
            .check("MiddleModule") \
            .check("BasicModule") \
            .check_not("__torch__.torch.classes.__backends__.test_backend") \
            .check_not("test_backendLoweredModule") \
            .run(self.scripted_module.other.graph)
        FileCheck() \
            .check("BasicModule") \
            .check_not("__torch__.torch.classes.__backends__.test_backend") \
            .check_not("test_backendLoweredModule") \
            .run(self.scripted_module.other.submodule.graph)

    def test_errors(self):
        """
        Check errors associated with selective lowering.
        """
        # Check error messages thrown when attempting to lower something that is not a ScriptModule.
        with self.assertRaisesRegex(RuntimeError, r"Object .* is not a ScriptModule"):
            to_test_backend_selective(torch.nn.ReLU(), {"forward": ""}, ["submodule"])

        MiddleModule = SelectiveLoweringTest.MiddleModule
        mod = MiddleModule(BasicModule())
        mod.new_attr = 3

        with self.assertRaisesRegex(RuntimeError, r"Attribute named new_attr is not a Module"):
            to_test_backend_selective(torch.jit.script(mod), {"forward": ""}, ["new_attr"])

        # Check error message thrown when module hierarchy doesn't have unique types.
        OuterModule = SelectiveLoweringTest.OuterModule
        mod = OuterModule(MiddleModule(BasicModule()), MiddleModule(BasicModule()), MiddleModule(BasicModule()))

        with self.assertRaisesRegex(RuntimeError, r"Selective lowering is only supported for module hierarchies with unique types"):
            to_test_backend_selective(torch.jit.script(mod), {"forward": ""}, ["sub1.submodule"])


# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
                 "Non-portable load_library call used in test")
class TestBackends(JitTestCase):
    """
    This class wraps and invokes all subclasses of JitBackendTestCase so that each one
    does not have to be individually imported in test_jit.py.
    """

    def __init__(self, name):
        super().__init__(name)
        self.basic_module_test = BasicModuleTest(name)
        self.basic_module_unavailable_test = BasicModuleUnavailableTest(name)
        self.nested_module_test = NestedModuleTest(name)
        self.selective_lowering_test = SelectiveLoweringTest(name)

    def setUp(self):
        super().setUp()
        if not TEST_WITH_ROCM:
            self.basic_module_test.setUp()
            self.basic_module_unavailable_test.setUp()
            self.nested_module_test.setUp()
            self.selective_lowering_test.setUp()

    @skipIfRocm
    def test_execution(self):
        self.basic_module_test.test_execution()
        self.basic_module_unavailable_test.test_execution()
        self.nested_module_test.test_execution()
        self.selective_lowering_test.test_execution()

    @skipIfRocm
    def test_save_load(self):
        self.basic_module_test.test_save_load()
        self.basic_module_unavailable_test.test_save_load()
        self.nested_module_test.test_save_load()
        self.selective_lowering_test.test_save_load()

    @skipIfRocm
    def test_errors(self):
        self.selective_lowering_test.test_errors()
