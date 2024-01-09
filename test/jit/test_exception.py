# Owner(s): ["oncall: jit"]
from torch.testing._internal.common_utils import TestCase
import torch
from torch import nn

r"""
Test TorchScript exception handling.
"""
class TestException(TestCase):
    def test_pyop_exception_message(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)

            @torch.jit.script_method
            def forward(self, x):
                return self.conv(x)
        foo = Foo()
        # testing that the correct error message propagates
        with self.assertRaisesRegex(RuntimeError, r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d"):
            foo(torch.ones([123]))  # wrong size

    def test_builtin_error_messsage(self):
        with self.assertRaisesRegex(RuntimeError, "Arguments for call are not valid"):
            @torch.jit.script
            def close_match(x):
                return x.masked_fill(True)

        with self.assertRaisesRegex(RuntimeError, "This op may not exist or may not be currently "
                                    "supported in TorchScript"):
            @torch.jit.script
            def unknown_op(x):
                torch.set_anomaly_enabled(True)
                return x

    def test_exceptions(self):
        cu = torch.jit.CompilationUnit('''
            def foo(cond):
                if bool(cond):
                    raise ValueError(3)
                return 1
        ''')

        cu.foo(torch.tensor(0))
        with self.assertRaisesRegex(torch.jit.Error, "3"):
            cu.foo(torch.tensor(1))

        def foo(cond):
            a = 3
            if bool(cond):
                raise ArbitraryError(a, "hi")  # noqa: F821
                if 1 == 2:
                    raise ArbitraryError  # noqa: F821
            return a

        with self.assertRaisesRegex(RuntimeError, "undefined value ArbitraryError"):
            torch.jit.script(foo)

        def exception_as_value():
            a = Exception()
            print(a)

        with self.assertRaisesRegex(RuntimeError, "cannot be used as a value"):
            torch.jit.script(exception_as_value)

        @torch.jit.script
        def foo_no_decl_always_throws():
            raise RuntimeError("Hi")

        # function that has no declared type but always throws set to None
        output_type = next(foo_no_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "NoneType")

        @torch.jit.script
        def foo_decl_always_throws():
            # type: () -> Tensor
            raise Exception("Hi")

        output_type = next(foo_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "Tensor")

        def foo():
            raise 3 + 4

        with self.assertRaisesRegex(RuntimeError, "must derive from BaseException"):
            torch.jit.script(foo)

        # a escapes scope
        @torch.jit.script
        def foo():
            if 1 == 1:
                a = 1
            else:
                if 1 == 1:
                    raise Exception("Hi")
                else:
                    raise Exception("Hi")
            return a
        self.assertEqual(foo(), 1)

        @torch.jit.script
        def tuple_fn():
            raise RuntimeError("hello", "goodbye")

        with self.assertRaisesRegex(torch.jit.Error, "hello, goodbye"):
            tuple_fn()

        @torch.jit.script
        def no_message():
            raise RuntimeError

        with self.assertRaisesRegex(torch.jit.Error, "RuntimeError"):
            no_message()

    def test_assertions(self):
        cu = torch.jit.CompilationUnit('''
            def foo(cond):
                assert bool(cond), "hi"
                return 0
        ''')

        cu.foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            cu.foo(torch.tensor(0))

        @torch.jit.script
        def foo(cond):
            assert bool(cond), "hi"

        foo(torch.tensor(1))
        # we don't currently validate the name of the exception
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            foo(torch.tensor(0))

    def test_python_op_exception(self):
        @torch.jit.ignore
        def python_op(x):
            raise Exception("bad!")

        @torch.jit.script
        def fn(x):
            return python_op(x)

        with self.assertRaisesRegex(RuntimeError, "operation failed in the TorchScript interpreter"):
            fn(torch.tensor(4))

    def test_dict_expansion_raises_error(self):
        def fn(self):
            d = {"foo": 1, "bar": 2, "baz": 3}
            return {**d}

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError,
                                    "Dict expansion "):
            torch.jit.script(fn)

    def test_custom_python_exception(self):
        class MyValueError(ValueError):
            pass

        @torch.jit.script
        def fn():
            raise MyValueError("test custom exception")

        with self.assertRaisesRegex(torch.jit.Error, "jit.test_exception.MyValueError: test custom exception"):
            fn()

    def test_custom_python_exception_defined_elsewhere(self):
        from jit.myexception import MyKeyError

        @torch.jit.script
        def fn():
            raise MyKeyError("This is a user defined key error")
        with self.assertRaisesRegex(torch.jit.Error, "jit.myexception.MyKeyError: This is a user defined key error"):
            fn()
