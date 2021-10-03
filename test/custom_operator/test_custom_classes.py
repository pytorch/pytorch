import unittest
import torch
from torch import ops
import torch.jit as jit
import glob
import os

from torch.testing._internal.common_utils import TestCase, run_tests


def get_custom_class_library_path():
    library_filename = glob.glob("build/*custom_class*")
    assert (len(library_filename) == 1)
    library_filename = library_filename[0]
    path = os.path.abspath(library_filename)
    assert os.path.exists(path), path
    return path

def test_equality(f, cmp_key):
    obj1 = f()
    obj2 = jit.script(f)()
    return (cmp_key(obj1), cmp_key(obj2))

class TestCustomOperators(TestCase):
    def setUp(self):
        ops.load_library(get_custom_class_library_path())

    def test_no_return_class(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            return val.info()
        self.assertEqual(*test_equality(f, lambda x: x))

    def test_constructor_with_args(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            return val
        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_function_call_with_args(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment(1)
            return val

        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_function_method_wrong_type(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment("asdf")
            return val

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            jit.script(f)()

    @unittest.skip("We currently don't support passing custom classes to custom methods.")
    def test_input_class_type(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(1, 2)
            val2 = torch.classes._TorchScriptTesting._Foo(2, 3)
            val.combine(val2)
            return val

        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_stack_string(self):
        def f():
            val = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            return val.pop()
        self.assertEqual(*test_equality(f, lambda x: x))

    def test_stack_push_pop(self):
        def f():
            val = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            val2 = torch.classes._TorchScriptTesting._StackString(["111", "222"])
            val.push(val2.pop())
            return val.pop() + val2.pop()
        self.assertEqual(*test_equality(f, lambda x: x))


if __name__ == "__main__":
    run_tests()
