import unittest
import torch
from torch import ops
import torch.jit as jit
import glob
import os
import sys
import pickle

# TODO: we should stop doing relative imports for test utils
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)
from common_utils import TemporaryFileName

# NB: the test classes for this unittest are built into the main libtorch
# library. We might want to separate it at some point, but right now they
# are available from torch.classes._TorchScriptTesting_*

def test_equality(f, cmp_key):
    obj1 = f()
    obj2 = jit.script(f)()
    return (cmp_key(obj1), cmp_key(obj2))

class TestCustomOperators(unittest.TestCase):
    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    if sys.version_info < (3, 5):
        # assertNotRegexpMatches renamed to assertNotRegex in 3.5
        assertNotRegex = unittest.TestCase.assertNotRegexpMatches

    def setUp(self):
        #ops.load_library(get_custom_class_library_path())
        pass

    def test_no_return_class(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(5, 3)
            return val.info()
        self.assertEqual(*test_equality(f, lambda x: x))

    def test_constructor_with_args(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(5, 3)
            return val
        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_function_call_with_args(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(5, 3)
            val.increment(1)
            return val

        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_function_method_wrong_type(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(5, 3)
            val.increment("asdf")
            return val

        with self.assertRaisesRegex(RuntimeError, "Expected"):
            jit.script(f)()

    @unittest.skip("We currently don't support passing custom classes to custom methods.")
    def test_input_class_type(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(1, 2)
            val2 = torch.classes._TorchScriptTesting_Foo(2, 3)
            val.combine(val2)
            return val

        self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_input_class_type_regular_op(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(1, 2)
            torch.ops._TorchScriptTesting.standalone_multiply_mutable(2, val)
            assert val.info() == 8
            return val

        self.assertEqual(8, torch.jit.script(f)().info())
        # Doesn't work yet on python level
        # self.assertEqual(*test_equality(f, lambda x: x.info()))

    def test_stack_string(self):
        def f():
            val = torch.classes._TorchScriptTesting_StackString(["asdf", "bruh"])
            return val.pop()
        self.assertEqual(*test_equality(f, lambda x: x))

    def test_stack_push_pop(self):
        def f():
            val = torch.classes._TorchScriptTesting_StackString(["asdf", "bruh"])
            val2 = torch.classes._TorchScriptTesting_StackString(["111", "222"])
            val.push(val2.pop())
            return val.pop() + val2.pop()
        self.assertEqual(*test_equality(f, lambda x: x))

    @unittest.skip("Serialization with pickle doesn't work yet")
    def test_pickle_serialization(self):
        # Note: so far we produce wrong pickle output and segfault at unpickling. Dump:
        #     0: \x80 PROTO      3
        #     2: c    GLOBAL     'torch._C ScriptObject'
        #    25: q    BINPUT     0
        #    27: )    EMPTY_TUPLE
        #    28: \x81 NEWOBJ
        #    29: q    BINPUT     1
        #    31: X    BINUNICODE 'magickey_123_456'
        #    52: q    BINPUT     2
        #    54: b    BUILD
        #    55: .    STOP
        val = torch.classes._TorchScriptTesting_Foo(123, 456)
        x = pickle.dumps(val)
        val2 = pickle.loads(x)
        self.assertEqual(val.info(), val2.info())

    def test_getstate_setstate(self):
        def f():
            val = torch.classes._TorchScriptTesting_Foo(3, 5)
            s = val.__getstate__()
            # TODO: sort out whether unpickler should call __new__ or __init__
            val2 = torch.classes._TorchScriptTesting_Foo(0, 0)
            val2.__setstate__(s)
            return val.info(), val2.info()
        ret = f()
        self.assertEqual(ret[0], ret[1])

        ret = torch.jit.script(f)()
        self.assertEqual(ret[0], ret[1])

    @unittest.skip("Serialization with torch.save doesn't work yet")
    def test_torch_serialization(self):
        def f(fn):
            val = torch.classes._TorchScriptTesting_Foo(3, 5)
            torch.save(val, fn)
            val2 = torch.load(fn)
            return val.info(), val2.info()
        with TemporaryFileName() as fname:
            ret = f(fname)
            self.assertEqual(ret[0], ret[1])
        with TemporaryFileName() as fname:
            ret = torch.jit.script(f)(fname)
            self.assertEqual(ret[0], ret[1])


if __name__ == "__main__":
    unittest.main()
