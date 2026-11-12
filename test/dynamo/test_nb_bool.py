# Owner(s): ["module: dynamo"]
"""Tests for nb_bool / generic_is_true: bool() via PyObject_IsTrue in Dynamo."""

import collections
import enum
import sys
import unittest

import torch


class _Color(enum.Enum):
    RED = 1
    BLUE = 2


import torch.nn
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    make_dynamo_test,
)


@torch._dynamo.config.patch("enable_trace_unittest", True)
@instantiate_parametrized_tests
class NbBoolTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    @make_dynamo_test
    def test_empty_deque(self):
        self.assertEqual(bool(collections.deque()), False)

    @make_dynamo_test
    def test_nonempty_deque(self):
        self.assertEqual(bool(collections.deque([1, 2])), True)

    @unittest.skipIf(sys.version_info >= (3, 11), "deque lost nb_bool in 3.11")
    @make_dynamo_test
    def test_deque_dunder_bool(self):
        self.assertEqual(collections.deque().__bool__(), False)
        self.assertEqual(collections.deque([1, 2]).__bool__(), True)

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

    @parametrize("case", ["str", "float", "none", "raises", "blocked", "negative", "overflow"])
    @parametrize("truth_test", ["bool", "if", "not"])
    def test_user_defined_len_error_propagates(self, case, truth_test):
        class BadLen:
            def __len__(self):
                if case == "raises":
                    raise TypeError("length failed")
                return {
                    "str": "x",
                    "float": 1.5,
                    "none": None,
                    "negative": -1,
                    "overflow": sys.maxsize + 1,
                }[case]

        if case == "blocked":
            BadLen.__len__ = None

        def fn(x, obj):
            try:
                if truth_test == "bool":
                    result = bool(obj)
                elif truth_test == "not":
                    result = not obj
                else:
                    result = 1 if obj else 0
            except (TypeError, ValueError, OverflowError) as exc:
                return x + 1, type(exc), str(exc)
            return x - 1, result, "missed exception"

        obj = BadLen()
        error_type = {
            "negative": ValueError,
            "overflow": OverflowError,
        }.get(case, TypeError)
        pattern = {
            "str": "cannot be interpreted as an integer",
            "float": "cannot be interpreted as an integer",
            "none": "cannot be interpreted as an integer",
            "raises": "length failed",
            "blocked": "not callable",
            "negative": "__len__.*should return >= 0",
            "overflow": "cannot fit.*index-sized integer",
        }[case]
        with self.assertRaisesRegex(error_type, pattern):
            bool(obj)
        x = torch.ones(1)
        expected = fn(x, obj)
        self.assertIs(expected[1], error_type)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, obj), expected)

    @parametrize("value", [0, 2, False, True])
    def test_user_defined_len_valid_truthiness(self, value):
        class Sized:
            def __len__(self):
                return value

        def fn(x, obj):
            return x + bool(obj), not obj, 1 if obj else 0

        x = torch.ones(1)
        obj = Sized()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, obj), fn(x, obj))

    def test_user_defined_bool_precedes_invalid_len(self):
        class HasBool:
            def __bool__(self):
                return False

            def __len__(self):
                raise TypeError("__len__ must not be called")

        def fn(x, obj):
            return x + bool(obj), not obj, 1 if obj else 0

        x = torch.ones(1)
        obj = HasBool()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, obj), fn(x, obj))

    def test_user_defined_len_accepts_index(self):
        class Length:
            def __index__(self):
                return 2

        class Sized:
            def __len__(self):
                return Length()

        def fn(x, obj):
            return x + bool(obj), not obj

        x = torch.ones(1)
        obj = Sized()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, obj), fn(x, obj))

    @parametrize("obj_kind", ["slice", "list_iter", "tuple_iter", "range_iter", "dict_iter", "function"])
    def test_no_length_protocol_stays_truthy(self, obj_kind):
        def fn(x, obj):
            return x + bool(obj), not obj

        obj = {
            "slice": lambda: slice(0),
            "list_iter": lambda: iter([]),
            "tuple_iter": lambda: iter(()),
            "range_iter": lambda: iter(range(0)),
            "dict_iter": lambda: iter({}),
            "function": lambda: fn,
        }[obj_kind]()
        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, obj), fn(x, obj))

    @parametrize("container_kind", ["list", "tuple", "dict", "empty_list", "range"])
    def test_nonconstant_container_truthiness(self, container_kind):
        def fn(x):
            if container_kind == "list":
                obj = [x]
            elif container_kind == "tuple":
                obj = (x,)
            elif container_kind == "dict":
                obj = {"k": x}
            elif container_kind == "empty_list":
                obj = [x][:0]
            else:
                obj = range(x.size(0))
            return x + bool(obj), 1 if obj else 0

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    def test_user_defined_symbolic_len_stays_dynamic(self):
        class Sized:
            def __init__(self, tensor):
                self.tensor = tensor

            def __len__(self):
                return self.tensor.size(0)

        def fn(x, obj):
            return x + bool(obj), 1 if obj else 0

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
        for size in (2, 4):
            x = torch.ones(size)
            obj = Sized(x)
            self.assertEqual(compiled(x, obj), fn(x, obj))
        self.assertEqual(counter.frame_count, 1)

    def test_user_defined_bool_returns_non_bool_raises(self):
        class BadBool:
            def __bool__(self):
                return 1

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        with self.assertRaises(TypeError):
            bool(BadBool())
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(torch.randn(4), BadBool())

    def test_if_cond_user_defined_object_returns_self(self):
        class MyObj:
            def __bool__(self):
                return self

        def fn(a, obj):
            try:
                if obj:
                    return a + 1
                return a - 1
            except TypeError as e:
                return str(e)

        x = torch.rand(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, MyObj())
        self.assertTrue(type(out) is str)
        self.assertEqual(out, fn(x, MyObj()))

    def test_bool_user_defined_object_raises_typeerror(self):
        class Baz(int):
            def __bool__(self):
                return self

        def fn(obj):
            try:
                bool(obj)
            except TypeError as e:
                return str(e)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        out = opt_fn(Baz())
        self.assertTrue(type(out) is str)
        self.assertEqual(fn(Baz()), out)

    # --- Blocked slot: __bool__ = None ---

    def test_user_defined_bool_none_raises(self):
        class NoBool:
            __bool__ = None

        obj = NoBool()

        def fn(x):
            with self.assertRaises(TypeError):
                str(bool(obj))
            return x.sin()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

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

    # --- int-typed VTs with no runtime object ---

    def test_fake_id_bool(self):
        # id() of an object minted during tracing yields a FakeIdVariable,
        # whose python_type is int and so fills nb_bool.
        def fn(x):
            tmp = [1, 2, 3]
            return x + 1, bool(id(tmp))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    def test_data_ptr_bool_graph_breaks(self):
        # A data pointer is only known at runtime, so its truth value cannot be
        # decided at trace time (torch.empty(0).data_ptr() is 0).
        def fn(x):
            return bool(x.data_ptr())

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Data pointer truth value"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

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
