# Owner(s): ["module: dynamo"]
"""Tests for nb_bool / generic_bool: bool() via PyObject_IsTrue in Dynamo."""

import collections
import enum

import torch


class _Color(enum.Enum):
    RED = 1
    BLUE = 2


import torch.nn
from torch.testing._internal.common_utils import make_dynamo_test, run_tests, TestCase


class NbBoolTests(TestCase):
    # --- Scalar constants (ConstantVariable path) ---

    @make_dynamo_test
    def test_int(self):
        self.assertEqual(bool(0), False)
        self.assertEqual(bool(1), True)
        self.assertEqual(bool(-1), True)

    @make_dynamo_test
    def test_float(self):
        self.assertEqual(bool(0.0), False)
        self.assertEqual(bool(-0.0), False)
        self.assertEqual(bool(3.14), True)

    @make_dynamo_test
    def test_none(self):
        self.assertEqual(bool(None), False)

    @make_dynamo_test
    def test_str(self):
        self.assertEqual(bool(""), False)
        self.assertEqual(bool("nonempty"), True)

    @make_dynamo_test
    def test_bytes(self):
        self.assertEqual(bool(b""), False)
        self.assertEqual(bool(b"hello"), True)

    @make_dynamo_test
    def test_bool(self):
        self.assertEqual(False, False)
        self.assertEqual(True, True)

    @make_dynamo_test
    def test_complex_zero(self):
        self.assertEqual(bool(0j), False)

    @make_dynamo_test
    def test_complex_nonzero(self):
        self.assertEqual(bool(1 + 2j), True)

    @make_dynamo_test
    def test_complex_real_nonzero_imag_zero(self):
        self.assertEqual(bool(1 + 0j), True)

    @make_dynamo_test
    def test_complex_real_zero_imag_nonzero(self):
        self.assertEqual(bool(0 + 1j), True)

    # --- Containers (length fallback / _bool_from_length path) ---

    @make_dynamo_test
    def test_empty_list(self):
        self.assertEqual(bool([]), False)

    @make_dynamo_test
    def test_nonempty_list(self):
        self.assertEqual(bool([1, 2, 3]), True)

    @make_dynamo_test
    def test_empty_dict(self):
        self.assertEqual(bool({}), False)

    @make_dynamo_test
    def test_nonempty_dict(self):
        self.assertEqual(bool({"a": 1}), True)

    @make_dynamo_test
    def test_empty_tuple(self):
        self.assertEqual(bool(()), False)

    @make_dynamo_test
    def test_nonempty_tuple(self):
        self.assertEqual(bool((1,)), True)

    @make_dynamo_test
    def test_empty_set(self):
        self.assertEqual(bool(set()), False)

    @make_dynamo_test
    def test_nonempty_set(self):
        self.assertEqual(bool({1, 2}), True)

    @make_dynamo_test
    def test_empty_frozenset(self):
        self.assertEqual(bool(frozenset()), False)

    @make_dynamo_test
    def test_nonempty_frozenset(self):
        self.assertEqual(bool(frozenset({1})), True)

    @make_dynamo_test
    def test_empty_range(self):
        self.assertEqual(bool(range(0)), False)

    @make_dynamo_test
    def test_nonempty_range(self):
        self.assertEqual(bool(range(5)), True)

    # --- dict subclasses ---

    @make_dynamo_test
    def test_empty_defaultdict(self):
        d = collections.defaultdict(int)
        self.assertEqual(bool(d), False)

    @make_dynamo_test
    def test_nonempty_defaultdict(self):
        d = collections.defaultdict(int, {"x": 1})
        self.assertEqual(bool(d), True)

    @make_dynamo_test
    def test_empty_counter(self):
        c = collections.Counter()
        self.assertEqual(bool(c), False)

    @make_dynamo_test
    def test_nonempty_counter(self):
        c = collections.Counter("abc")
        self.assertEqual(bool(c), True)

    # --- Enum (UserDefinedClassVariable / ConstantVariable path) ---

    @make_dynamo_test
    def test_enum_member(self):
        self.assertEqual(bool(_Color.RED), True)
        self.assertEqual(bool(_Color.BLUE), True)

    # --- UserDefinedObjectVariable tests (torch.compile path) ---

    def test_user_defined_with_bool(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def __bool__(self):
                return self.val > 0

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(5)), compiled(x, MyObj(5)))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(-1)), compiled(x, MyObj(-1)))

    def test_user_defined_with_len(self):
        class Container:
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

        def fn(x, c):
            return x + 1 if bool(c) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([1, 2])), compiled(x, Container([1, 2])))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([])), compiled(x, Container([])))

    def test_user_defined_no_bool_no_len(self):
        class Plain:
            pass

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, Plain()), compiled(x, Plain()))

    def test_user_defined_bool_returns_non_bool_raises(self):
        class BadBool:
            def __bool__(self):
                return 1  # noqa: PLE0305

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        with self.assertRaises(TypeError):
            bool(BadBool())
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(torch.randn(4), BadBool())

    # --- Metaclass with __bool__ (UserDefinedClassVariable path) ---

    def test_metaclass_bool(self):
        class Foo(type):
            def __bool__(cls):
                return False

        class A(metaclass=Foo):
            pass

        class Bar(type):
            pass

        class B(metaclass=Bar):
            pass

        def fn(x):
            # A's metaclass defines __bool__ returning False; B's does not (truthy).
            return x + bool(A) + bool(B)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- nn.Module (NNModuleVariable path) ---

    def test_nn_module_nonempty(self):
        mod = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def fn(x):
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_nn_module_empty(self):
        mod = torch.nn.ModuleList()

        def fn(x):
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- Tensor (TensorVariable path) ---

    def test_tensor_nonzero(self):
        def fn(x):
            t = torch.tensor(1)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_tensor_zero(self):
        def fn(x):
            t = torch.tensor(0)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


if __name__ == "__main__":
    run_tests()
