# Owner(s): ["oncall: jit"]

import io
import os
import sys
import unittest

import torch
import torch._C
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing import FileCheck

from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    skipIfRocm,
    TEST_WITH_ROCM,
)
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


def to_test_backend(module, method_compile_spec):
    return torch._C._jit_to_backend(
        "test_backend", module, {"forward": method_compile_spec}
    )


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

    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h


# This is ignored in IS_WINDOWS or IS_MACOS cases. Hence we need the one in TestBackends.
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class JitBackendTestCase(JitTestCase):
    """
    A common base class for JIT backend tests that contains common utility
    functions for output comparison and serialization/deserialization.
    """

    def setUp(self):
        super().setUp()
        lib_file_path = find_library_location("libjitbackend_test.so")
        torch.ops.load_library(str(lib_file_path))
        # Subclasses are expected to set up three variables in their setUp methods:
        # module - a regular, Python version of the module being tested
        # scripted_module - a scripted version of module
        # lowered_module - a version of module lowered to a backend

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
        pre_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")

        # Save and load the lowered module.
        self.save_load()

        # Get the compile spec after loading.
        post_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")

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
        with self.assertRaisesRegexWithHighlight(
            Exception,
            r"Backend is not available.",
            'raise Exception("Backend is not available."',
        ):
            backend_method = self.lowered_module.__getattr__("forward")
            backend_output = backend_method(*(input, input))

    @skipIfRocm
    def test_save_load(self):
        # Test that saving the lowered module is OK but loading fails because the backend is not available.
        buffer = io.BytesIO()
        torch.jit.save(self.lowered_module, buffer)
        buffer.seek(0)
        with self.assertRaisesRegexWithHighlight(
            Exception,
            r"Backend is not available.",
            'raise Exception("Backend is not available."',
        ):
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
        self.scripted_module = torch.jit.script(
            NestedModuleTest.NestedModule(BasicModule())
        )

        # First, script another instance of NestedModule with share_types=False so that it can be
        # selectively lowered without modifying the type of self.scripted_module.
        lowered_module = to_test_backend_multi(
            torch.jit.script(BasicModule()),
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )
        # self.lowered_module is a ScriptModule, but its submodule is a lowered module.
        self.lowered_module = torch.jit.script(
            NestedModuleTest.NestedModule(lowered_module)
        )

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
            return torch.jit._recursive.create_script_module(
                mod, torch.jit._recursive.infer_methods_to_compile, share_types=False
            )

        # Create Python, JIT and backend versions of a hierarchy that looks like this:
        #                 --------- OuterModule --------
        #                 |              |              |
        #           MiddleModule    MiddleModule   MiddleModule
        #                |               |              |
        #           BasicModule     BasicModule    BasicModule
        #
        # Two BasicModules will be lowered and the third will not.
        self.module = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )
        self.scripted_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        self.lowered_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        self.lowered_module = to_test_backend_selective(
            self.lowered_module, {"forward": ""}, ["sub1.submodule", "sub2.submodule"]
        )

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
        FileCheck().check("OuterModule").check("BasicModule").run(
            self.scripted_module.graph
        )
        FileCheck().check("OuterModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.graph)

        # Check that self.lowered_module.sub1/sub2 were not lowered but that BasicModule has been replaced in their graphs.
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub1.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub1.graph)

        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub2.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub2.graph)

        # Check that self.lowered_module.sub1/sub2.submodule were lowered. They should have a new attribute
        # __loweredModule__ whose graph should mention __torch__.torch.classes.__backends__.test_backend,
        # the TorchBind class for executing functions on the test JIT backend.
        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub1.submodule.__loweredModule__.graph)

        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub2.submodule.__loweredModule__.graph)

        # Check that self.other and self.other.submodule have been left untouched by the selective lowering process.
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredWrapper.test_backend").run(self.scripted_module.other.graph)
        FileCheck().check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredModule.test_backend").run(
            self.scripted_module.other.submodule.graph
        )

    def test_errors(self):
        """
        Check errors associated with selective lowering.
        """
        # Check error messages thrown when attempting to lower something that is not a ScriptModule.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Object .* is not a ScriptModule", ""
        ):
            to_test_backend_selective(torch.nn.ReLU(), {"forward": ""}, ["submodule"])

        MiddleModule = SelectiveLoweringTest.MiddleModule
        mod = MiddleModule(BasicModule())
        mod.new_attr = 3

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute named new_attr is not a Module", ""
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["new_attr"]
            )

        # Check error message thrown when module hierarchy doesn't have unique types.
        OuterModule = SelectiveLoweringTest.OuterModule
        mod = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Selective lowering is only supported for module hierarchies with unique types",
            "",
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["sub1.submodule"]
            )


# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
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


"""
Unit Tests for backend with compiler
This test case and the existing TestBackends are separate because they cover different aspects.
The actual backend implementation in this test is different.
It has a simple demo compiler to test the end-to-end flow in mobile.
However, this test cannot cover the selective_lowering for now, which is covered in TestBackends.
"""


class BasicModuleAdd(torch.nn.Module):
    """
    A simple add Module used to test to_backend lowering machinery.
    """

    def forward(self, x, h):
        return x + h


# This is ignored in IS_WINDOWS or IS_MACOS cases. Hence we need the one in TestBackends.
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class JitBackendTestCaseWithCompiler(JitTestCase):
    """
    A common base class for JIT backend tests with compilers that contains common utility
    functions for output comparison.
    """

    def setUp(self):
        super().setUp()
        lib_file_path = find_library_location("libbackend_with_compiler.so")
        torch.ops.load_library(str(lib_file_path))
        # Subclasses are expected to set up four variables in their setUp methods:
        # module - a regular, Python version of the module being tested
        # scripted_module - a scripted version of module
        # lowered_module - a version of module lowered to a backend
        # mobile_module - a module with a format that Pytorch Mobile can execute

    def check_forward(self, input):
        """
        Check that the forward function produces the same output using
        Python, regular JIT, the backend, and mobile for the given 'input'.
        """

        # Get outputs from forward.
        python_output = self.module.forward(*input)
        jit_output = self.scripted_module.forward(*input)
        backend_output = self.lowered_module(*input)
        mobile_output = self.mobile_module(*input)

        # The answers returned by Python, JIT, to_backend, and mobile should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)
        self.assertEqual(mobile_output, backend_output)

    def test_execution(self):
        """
        Stub for correctness tests.
        """
        pass

    def test_errors(self):
        """
        Stub for testing error checking.
        """
        pass


class BasicModuleTestWithCompiler(JitBackendTestCaseWithCompiler):
    """
    Tests for BasicModuleAdd.
    """

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of BasicModuleAdd.
        self.module = BasicModuleAdd()
        self.scripted_module = torch.jit.script(BasicModuleAdd())
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        self.lowered_module = torch._C._jit_to_backend(
            "backend_with_compiler_demo", self.scripted_module, compile_spec
        )
        # Create mobile version of BasicModuleAdd
        buffer = io.BytesIO(self.lowered_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.ones(1, dtype=torch.float)
        self.check_forward((input, input))


class ErrorMessagesWithCompiler(JitBackendTestCase):
    """
    Tests for errors that occur with compiler, specifically:
        * an operator is not supported by the backend
    """

    class ModuleNotSupported(torch.nn.Module):
        """
        A module with an operator that is not supported.
        """

        def forward(self, x, h):
            return x * h
            self._loweredmodule.forward()

    def test_errors(self):
        scripted_module_n = torch.jit.script(
            ErrorMessagesWithCompiler.ModuleNotSupported()
        )
        # Test exception is thrown when lowering a module with an unsupported operator
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            # Special escape characters are replaced with '.'
            r"""The node of aten::mul is not supported in this compiler. .*
        def forward.self, x, h.:
            return x . h
                   ~~~~~ <--- HERE
            self._loweredmodule.forward..
""",
            "",
        ):
            lowered_module_n = torch._C._jit_to_backend(
                "backend_with_compiler_demo", scripted_module_n, {"forward": {"": ""}}
            )


class CompModuleTestWithCompiler(JitBackendTestCase):
    """
    Tests for CompModule, which is a module with two lowered submodules
    """

    class BasicModuleSub(torch.nn.Module):
        """
        A simple subtraction Module to be used in CompModule.
        """

        def forward(self, x, h):
            return x - h

    class CompModule(torch.nn.Module):
        """
        A module with two lowered submodules.
        """

        def __init__(self, addmodule, submodule):
            super().__init__()
            self.lowered_add = addmodule
            self.lowered_sub = submodule

        def forward(self, a, b, s):
            c = self.lowered_add.forward(a, b)
            d = self.lowered_sub.forward(a, b)
            y = s * (c * d)
            return y

    def setUp(self):
        super().setUp()
        # Create Python and JIT versions of CompModule with lowered submodules.
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        lowered_add = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(BasicModuleAdd()),
            compile_spec,
        )
        lowered_sub = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(CompModuleTestWithCompiler.BasicModuleSub()),
            {"forward": {"": ""}},
        )
        self.module = CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)
        self.scripted_module = torch.jit.script(
            CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)
        )
        # No backend version of CompModule currently, so this is filler.
        self.lowered_module = self.scripted_module
        # Create a mobile version of CompModule from JIT version
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input1 = torch.ones(1, dtype=torch.float)
        input2 = torch.ones(1, dtype=torch.float)

        # Test forward.
        self.check_function("forward", (input1, input2, input2))


# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class TestBackendsWithCompiler(JitTestCase):
    """
    This class wraps and invokes all subclasses of JitBackendTestCaseWithCompiler
    so that each one does not have to be individually imported in test_jit.py.
    """

    def __init__(self, name):
        super().__init__(name)
        self.basic_module_compiler_test = BasicModuleTestWithCompiler(name)
        self.error_module_compiler_test = ErrorMessagesWithCompiler(name)
        self.comp_module_compiler_test = CompModuleTestWithCompiler(name)

    def setUp(self):
        super().setUp()
        self.basic_module_compiler_test.setUp()
        self.error_module_compiler_test.setUp()
        self.comp_module_compiler_test.setUp()

    def test_execution(self):
        self.basic_module_compiler_test.test_execution()
        self.comp_module_compiler_test.test_execution()

    def test_errors(self):
        self.error_module_compiler_test.test_errors()


class CompModuleTestSameNameWithCompiler(JitBackendTestCase):
    """
    Tests for CompModule, which is a module with two lowered submodules with same module name
    """

    class ModuleAdd(torch.nn.Module):
        """
        A simple Module used to test to_backend lowering machinery.
        """

        def forward(self, x, h):
            return x + h

    class CompModule(torch.nn.Module):
        """
        A module with two lowered submodules.
        """

        def __init__(self):
            super().__init__()
            compile_spec = {
                "forward": {
                    "some_other_option": "True",
                },
            }
            self.add = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )
            self.sub = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )

        def forward(self, a, b, s: int):
            c = self.add.forward(a, b)
            d = self.sub.forward(a, b)
            y = s * (c * d)
            return y

    def setUp(self):
        super().setUp()

        self.module = CompModule()  # noqa: F821
        self.scripted_module = torch.jit.script(self.module)
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        a = torch.ones(1)
        b = 3 * torch.ones(1)
        s = 3
        # Test forward.
        self.check_function("forward", (a, b, s))


class AddedAttributesTest(JitBackendTestCase):
    """
    Tests for adding attributes to a model after lowering.
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

    def test_attribute(self):
        input = [(torch.ones(5),)]
        pre_bundled = self.lowered_module(*input[0])
        # Attach bundled inputs which adds several attributes and functions to the model
        self.lowered_module = (
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                lowered_module, input  # noqa: F821
            )
        )
        post_bundled = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        # Save and load the lowered module.
        self.save_load()
        # Use bundled after save and load to prove its preserved
        post_load = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        self.assertEqual(pre_bundled, post_bundled)
        self.assertEqual(post_bundled, post_load)
