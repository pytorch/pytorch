# Owner(s): ["module: dynamo"]

import functools
import math
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter


class IdTests(torch._dynamo.test_case.TestCase):
    def _assert_id_equals(self, obj):
        """Assert Dynamo's id(obj) matches eager Python's id(obj)."""
        expected = id(obj)

        def fn(_, o):
            return id(o)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(0), obj
        )
        self.assertEqual(result, expected)

    def _assert_id_graph_breaks(self, fn):
        """Assert fn graph-breaks under fullgraph=True due to FakeIdVariable."""
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    # =====================================================================
    # Category 1: Sourceful objects — id() matches eager
    # =====================================================================

    def test_id_user_function(self):
        def target():
            pass

        self._assert_id_equals(target)

    def test_id_lambda(self):
        self._assert_id_equals(lambda: None)

    def test_id_user_object(self):
        class MyObj:
            pass

        self._assert_id_equals(MyObj())

    def test_id_module_import(self):
        self._assert_id_equals(math)

    def test_id_builtin_len(self):
        self._assert_id_equals(len)

    def test_id_builtin_type_int(self):
        self._assert_id_equals(int)

    def test_id_builtin_type_list(self):
        self._assert_id_equals(list)

    def test_id_tensor(self):
        self._assert_id_equals(torch.tensor([1.0, 2.0]))

    def test_id_nn_module(self):
        self._assert_id_equals(torch.nn.Linear(2, 3))

    def test_id_user_defined_class(self):
        class MyClass:
            pass

        self._assert_id_equals(MyClass)

    def test_id_none(self):
        def fn(_):
            return id(None)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(None))

    def test_id_true(self):
        def fn(_):
            return id(True)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(True))

    def test_id_false(self):
        def fn(_):
            return id(False)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(False))

    def test_id_functools_partial(self):
        p = functools.partial(lambda x, y: x + y, 1)
        self._assert_id_equals(p)

    def test_id_sourceful_dict(self):
        self._assert_id_equals({"a": 1, "b": 2})

    def test_id_sourceful_list(self):
        self._assert_id_equals([1, 2, 3])

    def test_id_sourceful_tuple(self):
        self._assert_id_equals((1, 2))

    def test_id_sourceful_int(self):
        self._assert_id_equals(42)

    def test_id_sourceful_string(self):
        self._assert_id_equals("hello")

    # =====================================================================
    # Category 2: Sourceful objects — ID_MATCH guard when consumed
    # =====================================================================

    def _assert_id_recompiles(self, fn, args1, args2):
        """Assert fn(x, *args1) compiles, then fn(x, *args2) recompiles."""
        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(torch.tensor(0), *args1)
        self.assertEqual(cnt.frame_count, 1)
        opt_fn(torch.tensor(0), *args2)
        self.assertEqual(cnt.frame_count, 2)

    def test_id_guards_graph_input(self):
        class MyObj:
            pass

        def fn(x, obj):
            return x + id(obj)

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_constant_op(self):
        class MyObj:
            pass

        def fn(x, obj):
            k = id(obj) * 2
            return x + k

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_return(self):
        class MyObj:
            pass

        def fn(x, obj):
            return x + 1.0, id(obj)

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_on_branch(self):
        class MyObj:
            pass

        def fn(x, obj):
            if id(obj) % 2 == 0:
                return x + 1
            return x + 2

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_on_comparison(self):
        class MyObj:
            pass

        obj1 = MyObj()

        def fn(x, a, b):
            if id(a) == id(b):
                return x + 1
            return x + 2

        self._assert_id_recompiles(fn, [obj1, obj1], [obj1, MyObj()])

    def test_id_guards_side_effect(self):
        class MyObj:
            pass

        log = []

        def fn(x, obj):
            log.append(id(obj))
            return x + 1.0

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_class_guard_uses_class_match(self):
        class MyClass1:
            pass

        class MyClass2:
            pass

        def fn(x, cls):
            return x + id(cls) // 100000

        self._assert_id_recompiles(fn, [MyClass1], [MyClass2])

    def test_id_nn_module_guard(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                if id(self) == ref_id:
                    return torch.mul(x, 1.0)
                return torch.mul(x, 0)

        m1 = M()
        m2 = M()

        def fn(x, m):
            return m(x, id(m1))

        self._assert_id_recompiles(fn, [m1], [m2])

    def test_id_tensor_guard(self):
        t1 = torch.ones(2)
        t2 = torch.zeros(2)

        def fn(x, t):
            if id(t) == id(t1):
                return torch.mul(x, t1)
            return torch.mul(x, t2)

        self._assert_id_recompiles(fn, [t1], [t2])

    def test_id_guarded_object(self):
        class UserDefinedObject:
            pass

        obj1 = UserDefinedObject()

        def fn(x, obj):
            if id(obj) == id(obj1):
                return torch.mul(x, 1.0)
            return torch.mul(x, 0)

        self._assert_id_recompiles(fn, [obj1], [UserDefinedObject()])

    def test_id_of_nn_module_op_count(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                if id(self) == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)

        cnts = CompileCounter()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, id(m))
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = CompileCounter()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, id(m) + 1)
        self.assertEqual(cnts.op_count, 1)

    # TODO: dict-key-only usage currently overguards (installs ID_MATCH eagerly).
    # Once LazyConstantVariable (PR #170644) lands, this test should pass with
    # frame_count == 1 (no recompile), and the expectedFailure can be removed.
    # Also add recompile tests for dict key usage once lazy constants are landed.
    @torch._dynamo.config.patch(error_on_recompile=False)
    @unittest.expectedFailure
    def test_id_dict_key_only_no_guard(self):
        class MyObj:
            pass

        obj1 = MyObj()
        obj2 = MyObj()
        cnt = CompileCounter()

        def fn(x, obj):
            d = {id(obj): 1}
            return x + d[id(obj)]

        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(torch.tensor(0), obj1)
        self.assertEqual(cnt.frame_count, 1)
        opt_fn(torch.tensor(0), obj2)
        # Dict-key-only usage should NOT recompile, but currently does
        self.assertEqual(cnt.frame_count, 1)

    # =====================================================================
    # Category 3: Sourceless objects — graph break when id escapes
    # =====================================================================

    def test_id_sourceless_return_graph_breaks(self):
        self._assert_id_graph_breaks(lambda x: (x + 1.0, id([1, 2, 3])))

    def test_id_sourceless_branch_graph_breaks(self):
        def fn(x):
            if id([1, 2, 3]) % 2 == 0:
                return x + 1.0
            return x + 2.0

        self._assert_id_graph_breaks(fn)

    def test_id_sourceless_tensor_op_graph_breaks(self):
        self._assert_id_graph_breaks(lambda x: x + id([1, 2, 3]))

    def test_id_sourceless_side_effect_graph_breaks(self):
        log = []

        def fn(x):
            obj = [1, 2, 3]
            log.append(id(obj))
            return x + 1.0

        self._assert_id_graph_breaks(fn)

    def test_id_sourceless_comparison(self):
        def fn(x):
            a = [1, 2]
            b = [3, 4]
            if id(a) == id(b):
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    # =====================================================================
    # Category 4: id() as dict key should not graph break
    # =====================================================================

    def _assert_id_dict_key_works(self, fn, *args):
        """Assert fn compiles with fullgraph=True and matches eager."""
        result = torch.compile(fn, backend="eager", fullgraph=True)(*args)
        self.assertEqual(result, fn(*args))

    def test_id_sourceless_int_as_key(self):
        def fn(x):
            d = {id(1000000): True}
            if id(1000000) in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_string_as_key(self):
        def fn(x):
            d = {id("hello"): True}
            if id("hello") in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_dict_as_key(self):
        def fn(x):
            d = {"a": 1, "b": 2}
            memo = {id(d): True}
            if id(d) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_partial_as_key(self):
        def fn(x):
            p = functools.partial(lambda a, b: a + b, 1)
            d = {id(p): True}
            if id(p) in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_list_as_key(self):
        def fn(x):
            lst = [1.0, 2.0]
            memo = {id(lst): True}
            if id(lst) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_function_as_dict_key(self):
        def nothing():
            pass

        def fn(x, f):
            d = {id(f): 3}
            return x * d[id(f)]

        self._assert_id_dict_key_works(fn, torch.randn(4), nothing)

    def test_id_functools_partial_as_dict_key(self):
        def gn(a, b):
            return a + b

        partial_gn = functools.partial(gn, a=3)

        def fn(x):
            d = {id(partial_gn): 5}
            return partial_gn(b=x) * d[id(partial_gn)]

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceful_dict_as_key(self):
        MY_DICT = {"a": 1, "b": 2}

        def fn(x):
            memo = {id(MY_DICT): True}
            if id(MY_DICT) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceful_list_as_key(self):
        MY_LIST = [1.0, 2.0]

        def fn(x):
            memo = {id(MY_LIST): True}
            if id(MY_LIST) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    # =====================================================================
    # Category 5: Error handling
    # =====================================================================

    def test_id_wrong_arg_count(self):
        def fn0(x):
            return id()

        def fn2(x):
            return id(x, x)

        for fn in [fn0, fn2]:
            torch._dynamo.reset()

            def wrapper(x):
                try:
                    return fn(x)
                except TypeError:
                    return x + 1.0

            x = torch.randn(4)
            result = torch.compile(wrapper, backend="eager", fullgraph=True)(x)
            self.assertEqual(result, wrapper(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
