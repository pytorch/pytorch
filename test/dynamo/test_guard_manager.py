# Owner(s): ["module: dynamo"]
import functools
import json
import os
import subprocess
import sys
import textwrap
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch._dynamo.convert_frame import GlobalStateGuard
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch.testing._internal.common_utils import set_default_dtype


RootGuardManager = guards.RootGuardManager
DictGuardManager = guards.DictGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
OBJECT_ALIASING = guards.OBJECT_ALIASING
install_object_aliasing_guard = guards.install_object_aliasing_guard
NO_TENSOR_ALIASING = guards.NO_TENSOR_ALIASING
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard


x = torch.tensor(4)
weakref_x = weakref.ref(x)

default_mgr_enum = torch._dynamo.guards.GuardManagerType.GUARD_MANAGER


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


global_pair = Pair(torch.randn(4), 1)


def id_type(x):
    return id(type(x))


def equals_match(x, expected):
    return x == expected


def equals_match_verbose_code_parts(expected):
    return [f"x == {expected}"]


def ge_match(x, expected):
    return x >= expected


def ge_match_verbose_code_parts(expected):
    return f"expected >= {expected}"


def less_match(x, expected):
    return x < expected


def less_match_verbose_code_parts(expected):
    return [f"expected < {expected}"]


class GuardManagerTests(torch._dynamo.test_case.TestCase):
    def test_global_state_guard(self):
        guard = guards.GLOBAL_STATE(["global_state_check"])
        self.assertTrue(guard(None))
        with set_default_dtype(torch.double):
            self.assertFalse(guard(None))
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: default_dtype '],
num_guards_executed=0)
""",
            )
        self.assertTrue(guard(None))
        self.assertTrue(guard.check_verbose(None).result)
        _orig = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(not _orig)
            self.assertFalse(guard(None))
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: deterministic_algorithms '],
num_guards_executed=0)
""",
            )
        finally:
            torch.use_deterministic_algorithms(_orig)
        self.assertTrue(guard(None))
        self.assertTrue(guard.check_verbose(None).result)

    def test_global_state_reason(self):
        with torch.enable_grad():
            guards = GlobalStateGuard()
        with torch.no_grad():
            self.assertIs(guards.check(), False)
            self.assertEqual(guards.reason(), "grad_mode ")

    def test_python_lambda_leaf_guard(self):
        const_guard = guards.LAMBDA_GUARD(
            functools.partial(equals_match, expected=5),
            equals_match_verbose_code_parts(5),
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_type_guard(self):
        foo = 4
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == int"])

        self.assertTrue(guard(5))
        self.assertTrue(guard(4))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == dict"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)

        guard = guards.TYPE_MATCH(id_type(foo), ["type(x) == Foo"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

    def test_id_guard(self):
        foo = 4
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.ID_MATCH(id(foo), ["id(x) == id(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))

    def test_equals_guard(self):
        foo = 4
        guard = guards.EQUALS_MATCH(foo, ["x == 4"])

        self.assertTrue(guard(4))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        # tuple
        foo = (1, 2, 3)
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard((1, 2, 3)))
        self.assertFalse(guard((1, 2, 3, 4)))
        self.assertFalse(guard({}))

        # list
        foo = [1, 2, 3]
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard([1, 2, 3]))
        self.assertFalse(guard([1, 2, 3, 4]))

        # type
        foo = int
        guard = guards.EQUALS_MATCH(foo, ["x == foo"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard(int))
        self.assertFalse(guard(float))

    def test_default_device_guard(self):
        foo = 1
        guard = guards.DEFAULT_DEVICE(["cpu device"])
        self.assertTrue(guard(foo))

        try:
            torch.set_default_device("cuda")
            self.assertFalse(guard(foo))
        finally:
            torch.set_default_device(None)

    def test_data_ptr_match_guard(self):
        foo = torch.tensor([1, 2, 3])
        guard = guards.DATA_PTR_MATCH(foo, ["x.data_ptr() == foo.data_ptr()"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard(torch.tensor([1, 2, 3])))

    def test_length_check_guard(self):
        foo = [1, 2, 3]
        guard = guards.LENGTH_CHECK(len(foo), ["len(x) == len(foo)"])
        self.assertTrue(guard(foo))
        self.assertFalse(guard([]))

    def test_no_hasattr_guard(self):
        class Bar:
            def __init__(self) -> None:
                self.bar = 2

        bar = Bar()

        class Foo:
            def __init__(self) -> None:
                self.foo = 2

        foo = Foo()

        guard = guards.NO_HASATTR("foo", ["hasattr(x, 'foo') == False"])
        self.assertTrue(guard(bar))
        self.assertFalse(guard(foo))

    def test_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        f_locals = Foo(a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        install_object_aliasing_guard(x_guard_mgr, y_guard_mgr, ["x is y"])

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertTrue(isinstance(x_guards[0], OBJECT_ALIASING))
        self.assertTrue(isinstance(y_guards[0], OBJECT_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0])

        f_locals_unaliased = Foo(torch.randn(3, 4), torch.randn(3, 4))
        self.assertEqual(len(x_guard_mgr.get_leaf_guards()), 1)
        self.assertEqual(len(y_guard_mgr.get_leaf_guards()), 1)
        self.assertTrue(guard_manager.check(f_locals))

        self.assertFalse(guard_manager.check(f_locals_unaliased))

    def test_dict_version_guard(self):
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_VERSION(foo, ["x.version == foo.version"])

        self.assertTrue(guard(foo))
        self.assertFalse(guard(dict(foo)))
        foo["a"] = 2
        self.assertFalse(guard(foo))
        self.assertFalse(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({}))

    def test_dynamic_indices_guard(self):
        guard1 = guards.DYNAMIC_INDICES(set(), ["x.size(0) == y.size(0)"])
        guard2 = guards.DYNAMIC_INDICES(set({0, 1}), ["x.size(0) == y.size(0)"])

        x = torch.randn(4)
        self.assertTrue(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({0})
        self.assertFalse(guard1(x))
        self.assertTrue(guard2(x))

        x._dynamo_dynamic_indices = set({2})
        self.assertFalse(guard1(x))
        self.assertFalse(guard2(x))

    def test_tensor_match_guard(self):
        guard_manager = RootGuardManager()
        x = torch.randn(4, 4)
        size = list(x.size())
        stride = list(x.stride())
        guard_manager.add_tensor_match_guard(x, size, stride, "x", ["check_tensor(x)"])
        self.assertTrue(guard_manager.check(x))
        self.assertTrue(guard_manager.check_verbose(x).result)
        self.assertTrue(guard_manager.check(torch.randn(4, 4)))
        self.assertTrue(guard_manager.check_verbose(torch.randn(4, 4)).result)
        self.assertFalse(guard_manager.check(x.t_()))

        x = torch.randn(4, 4)
        x.t_()
        debug_info = guard_manager.check_verbose(x)
        print(debug_info.verbose_code_parts[0])
        self.assertTrue(
            "tensor 'x' stride mismatch" in debug_info.verbose_code_parts[0]
        )

    def test_no_tensor_aliasing_guard(self):
        guard_manager = RootGuardManager()

        a = torch.randn(3, 4)

        class Foo:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        f_locals = Foo(a, a, a)

        x_guard_mgr = guard_manager.getattr_manager("x", "", a, default_mgr_enum)
        y_guard_mgr = guard_manager.getattr_manager("y", "", a, default_mgr_enum)
        z_guard_mgr = guard_manager.getattr_manager("z", "", a, default_mgr_enum)
        install_no_tensor_aliasing_guard(
            [x_guard_mgr, y_guard_mgr, z_guard_mgr],
            ["x", "y", "z"],
            ["no_aliasing(x, y, z)"],
        )

        # Check structure
        x_guards = x_guard_mgr.get_leaf_guards()
        y_guards = y_guard_mgr.get_leaf_guards()
        z_guards = z_guard_mgr.get_leaf_guards()
        self.assertEqual(len(x_guards), 1)
        self.assertEqual(len(y_guards), 1)
        self.assertEqual(len(z_guards), 1)
        self.assertTrue(isinstance(x_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(y_guards[0], NO_TENSOR_ALIASING))
        self.assertTrue(isinstance(z_guards[0], NO_TENSOR_ALIASING))
        # Check that the two guards are the same object
        self.assertTrue(x_guards[0] is y_guards[0] is z_guards[0])
        self.assertFalse(guard_manager.check(f_locals))
        self.assertFalse(guard_manager.check_verbose(f_locals).result)

        f_locals_unaliased = Foo(
            torch.randn(3, 4),
            torch.randn(3, 4),
            torch.randn(3, 4),
        )
        self.assertTrue(guard_manager.check(f_locals_unaliased))
        self.assertTrue(guard_manager.check_verbose(f_locals_unaliased).result)
        # Check that hash map is cleared.
        self.assertTrue(guard_manager.check(f_locals_unaliased))

        f_locals_unaliased = Foo(
            a,
            torch.randn(3, 4),
            a,
        )
        self.assertFalse(guard_manager.check(f_locals_unaliased))
        self.assertFalse(guard_manager.check_verbose(f_locals_unaliased).result)

    def test_weakref_alive_guard(self):
        x = torch.rand(3, 4)
        weakref_x = weakref.ref(x)

        guard = guards.NOT_NONE(["weakref_x is not None"])
        self.assertTrue(guard(weakref_x()))
        del x
        self.assertFalse(guard(weakref_x()))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_call_function_no_args_guard(self):
        x = torch.cuda.current_device()
        guard = guards.EQUALS_MATCH(x, [0])
        self.assertTrue(guard(0))
        self.assertFalse(guard(1))
        self.assertFalse(guard(2))

    def test_guard_manager_leaf_guard(self):
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"])
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_verbose_code_parts(expected=5),
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_verbose_code_parts(expected=10),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 3)
        self.assertEqual(len(guard_manager.get_accessors()), 0)
        self.assertTrue(guard_manager.check(6))
        self.assertFalse(guard_manager.check(4))
        self.assertFalse(guard_manager.check("foo"))

    def test_attr_guard_manager(self):
        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getattr_manager("x", "x", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_verbose_code_parts(foo.x),
        )
        guard_manager.getattr_manager("y", "y", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_verbose_code_parts(foo.y),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetAttrGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    attr="x",
                    source="x",
                    example_value=None,
                    guard_manager_enum=default_mgr_enum,
                ).get_leaf_guards()
            ),
            1,
        )
        self.assertEqual(
            len(
                guard_manager.getattr_manager(
                    "y", "y", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check(Foo(3, 4)))
        self.assertFalse(guard_manager.check("foo"))

    def test_item_guard_manager(self):
        foo = [1, 2]
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guard_manager.getitem_manager(0, "", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[0]),
            equals_match_verbose_code_parts(foo[0]),
        )
        guard_manager.getitem_manager(1, "", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[1]),
            equals_match_verbose_code_parts(foo[1]),
        )
        self.assertEqual(len(guard_manager.get_leaf_guards()), 1)
        # 2 child managers, one for x and one for y
        self.assertEqual(len(guard_manager.get_accessors()), 2)
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        self.assertTrue(
            isinstance(guard_manager.get_accessors()[1], GetItemGuardAccessor)
        )
        # Check leaf guards on child managers
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    0, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )
        self.assertEqual(
            len(
                guard_manager.getitem_manager(
                    1, "", None, default_mgr_enum
                ).get_leaf_guards()
            ),
            1,
        )

        self.assertTrue(guard_manager.check(foo))
        self.assertFalse(guard_manager.check([3, 4]))
        self.assertFalse(guard_manager.check("foo"))

    def test_framelocals_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guards_manager.framelocals_manager(
            ("a", 0), "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"])
        guards_manager.framelocals_manager(
            ("b", 1), "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"])

        self.assertTrue(guards_manager.check(foo))
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_framelocals_guard_e2e(self):
        def fn(x, y, z):
            return x + y + z[0]

        opt_fn = torch.compile(fn, backend="eager")

        ref = opt_fn(torch.ones(3), 2, {0: 1, 2: 3})
        with torch._dynamo.set_stance("fail_on_recompile"):
            res = opt_fn(torch.ones(3), 2, {0: 1, 2: 3})
        self.assertEqual(ref, res)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)
        guard_str = str(c1[0].guard_manager)
        self.assertIn(
            "source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)",
            guard_str,
        )
        self.assertIn(
            "source=L['y'], accessed_by=FrameLocalsGuardAccessor(key='y', framelocals_idx=1)",
            guard_str,
        )
        self.assertIn(
            "source=L['z'], accessed_by=FrameLocalsGuardAccessor(key='z', framelocals_idx=2)",
            guard_str,
        )

    @torch._dynamo.config.patch(enable_cpp_framelocals_guard_eval=False)
    def test_framelocals_guard_config_flag(self):
        def fn(x):
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        ref = opt_fn(torch.ones(3))
        with torch._dynamo.set_stance("fail_on_recompile"):
            res = opt_fn(torch.ones(3))
        self.assertEqual(ref, res)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)
        guard_str = str(c1[0].guard_manager)
        self.assertIn(
            "source=L['x'], accessed_by=DictGetItemGuardAccessor('x')",
            guard_str,
        )

    def test_dict_getitem_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"])
        guards_manager.dict_getitem_manager(
            "a", "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"])
        guards_manager.dict_getitem_manager(
            "b", "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"])

        self.assertTrue(guards_manager.check(foo))
        self.assertFalse(guards_manager.check({"a": 1, "b": 3}))

    def test_globals(self):
        global global_pair, Pair
        guard_manager = RootGuardManager()
        gpair_mgr = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        ).getitem_manager("global_pair", "", global_pair, default_mgr_enum)

        gpair_mgr.add_lambda_guard(
            lambda x: isinstance(x, Pair)
            and isinstance(x.x, torch.Tensor)
            and isinstance(x.y, int),
            "global guard fail",
        )

        self.assertTrue(guard_manager.check(global_pair))
        global_pair.y = "foo"
        self.assertFalse(guard_manager.check(global_pair))

    def test_type_manager(self):
        guard_manager = RootGuardManager()

        class A:
            a = 4

        class B(A):
            def mul(self, x):
                super().mul(x)

        foo = B()
        f_locals = {"foo": foo}

        # len(type(foo).__mro__) == 2
        foo_mgr = guard_manager.getitem_manager("foo", "", foo, default_mgr_enum)
        type_manager = foo_mgr.type_manager("", type(foo), default_mgr_enum)
        self.assertTrue(isinstance(foo_mgr.get_accessors()[0], TypeGuardAccessor))
        mro_manager = type_manager.getattr_manager(
            "__mro__", "", type(foo).__mro__, default_mgr_enum
        )
        self.assertTrue(
            isinstance(type_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        mro_manager.add_length_check_guard(
            3,
            "Expected len(type(foo).__mro__) == 3",
        )

        # type(foo).__mro__[0].a = 4
        item_manager = mro_manager.getitem_manager(
            1, "", type(foo).__mro__[1], default_mgr_enum
        )
        self.assertTrue(
            isinstance(mro_manager.get_accessors()[0], GetItemGuardAccessor)
        )
        attr_manager = item_manager.getattr_manager(
            "a", "", type(foo).__mro__[0].a, default_mgr_enum
        )
        self.assertTrue(
            isinstance(item_manager.get_accessors()[0], GetAttrGuardAccessor)
        )
        attr_manager.add_lambda_guard(
            lambda x: x == 4,
            "Expected value 4",
        )

        self.assertTrue(guard_manager.check(f_locals))

    def test_tuple_iterator_getitem(self):
        a = (1, 2, 3, 4, 5, 6)
        foo = iter(a)
        next(foo)  # foo points at index=1

        guard_manager = RootGuardManager()
        # Check a[3] which is tuple_iterator_getitem(foo, 2)
        guard_manager.add_tuple_iterator_length_guard(
            5, id_type(iter(())), ["len == 5"]
        )
        guard_manager.tuple_iterator_getitem_manager(
            2, "", foo, default_mgr_enum
        ).add_equals_match_guard(a[3], ["x==4"])

        # Check that type match works
        self.assertFalse(guard_manager.check(False))

        self.assertTrue(guard_manager.check(foo))

        # Check that index error fails gracefully
        b = (1, 2)
        b_foo = iter(b)
        self.assertFalse(guard_manager.check(b_foo))

    def test_global_weakref(self):
        guard_manager = RootGuardManager()
        globals_manager = guard_manager.globals_dict_manager(
            globals(), "", None, default_mgr_enum
        )
        weakref_manager = globals_manager.global_weakref_manager(
            "weakref_x", "", None, default_mgr_enum
        )

        weakref_manager.add_lambda_guard(
            lambda x: isinstance(x, torch.Tensor),
            "global weakref fail",
        )

        self.assertTrue(guard_manager.check(None))
        global x
        del x
        self.assertFalse(guard_manager.check(None))

    def test_lambda_manager(self):
        a = (1, 1, 3, 4, 5, 6)

        guard_manager = RootGuardManager()

        # Check that we can use the same accessor
        foo_mgr = guard_manager.lambda_manager(
            lambda x: x[2], "", None, default_mgr_enum
        )
        foo_mgr.add_lambda_guard(
            lambda x: x == 3,
            "Expected value 3",
        )
        self.assertTrue(guard_manager.check(a))

        # test that exception works
        guard_manager = RootGuardManager()

        def fn(x):
            raise AssertionError("Test")
            return x

        foo_mgr = guard_manager.lambda_manager(fn, "", None, default_mgr_enum)

        self.assertFalse(guard_manager.check(None))
        debug_info = guard_manager.check_verbose(None)
        self.assertFalse(debug_info.result)
        self.assertTrue("Test" in debug_info.verbose_code_parts[0])

    def test_dict_contains_guard(self):
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_CONTAINS(True, "a", ["has a"])

        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertFalse(guard({}))

        guard = guards.DICT_CONTAINS(False, "c", ["not has c"])
        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertTrue(guard({}))

    def test_dict_guard_manager(self):
        root = RootGuardManager()

        def nothing():
            pass

        f_locals = {
            "d": {"a": 1, nothing: {"z": 3}, 100: torch.randn(4)},
        }

        # its a getitem_manager just for f_locals. But the child guard manager
        # should be a DictGuardManager.
        dict_mgr = root.getitem_manager(
            "d",
            "",
            f_locals["d"],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(dict_mgr, DictGuardManager))

        self.assertTrue(root.check(f_locals))

        # Check that no one can add a leaf guard
        with self.assertRaises(RuntimeError):
            dict_mgr.add_id_match_guard(id_type(f_locals), "id match")

        # Check that no one can add an arbitrary accessor
        with self.assertRaises(RuntimeError):
            dict_mgr.getitem_manager("a", "", f_locals["d"]["a"])

        # Check that it fails with different length dict
        f_locals_prime = {
            "d": {"a": 1, "b": 2},
        }
        self.assertFalse(root.check(f_locals_prime))

        # Add key-value manager ("a" : 1)
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(0, "", "a", default_mgr_enum).add_equals_match_guard(
            "a",
            ["dict.keys()[0] == a"],
        )
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_value_manager(0, "", 1, default_mgr_enum).add_equals_match_guard(
            1, ["d[0] == 1"]
        )
        self.assertTrue(root.check(f_locals))

        # Add key-value manager (nothing : {"z" : 3})
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(1, "", nothing, default_mgr_enum).add_lambda_guard(
            lambda x: x is nothing, ["x is nothing"]
        )
        self.assertTrue(root.check(f_locals))
        value_mgr = dict_mgr.get_value_manager(
            1,
            "",
            f_locals["d"][nothing],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        self.assertTrue(isinstance(value_mgr, DictGuardManager))
        self.assertTrue(root.check(f_locals))

        # Check structure
        # Check that we are only guarding on two keys. This is common in
        # LazyVariableTracker.
        self.assertEqual(len(dict_mgr.get_key_value_managers()), 2)

        f_locals["d"]["a"] = 2
        self.assertFalse(root.check(f_locals))
        self.assertFalse(root.check_verbose(f_locals).result)

        f_locals["d"]["a"] = 1
        self.assertTrue(root.check(f_locals))

        f_locals["d"].pop(100)
        # fails because of len check
        self.assertFalse(root.check(f_locals))

    def test_clone(self):
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals):
            root = guard_wrapper.root

            # Check full cloning works as expected
            cloned_root = root.clone_manager(lambda x: True)
            self.assertTrue(cloned_root.check(f_locals))
            f_locals["foo"] = [3, 4]
            self.assertFalse(cloned_root.check(f_locals))
            f_locals["foo"] = [2, 3]

            # Skip guarding on foo
            cloned_root = root.clone_manager(lambda x: "foo" not in x.get_source())
            f_locals["foo"] = [3, 4]
            # Original root should fail, but new root should pass because of
            # absence of guards on foo.
            self.assertFalse(root.check(f_locals))
            self.assertTrue(cloned_root.check(f_locals))

        class Bar:
            x = 4
            y = torch.randn(4)

        foo = [2, 3]
        bar = Bar()

        def fn(x, foo, bar):
            return x + foo[0] + bar.x * bar.y

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(x, foo, bar)

    def test_diff_guard_manager(self):
        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook
        counter = 0

        def hook(guard_wrapper, f_locals):
            nonlocal counter
            root = guard_wrapper.root
            diff_guard_root = guard_wrapper.diff_guard_root

            # Check full cloning works as expected
            self.assertTrue(root.check(f_locals))
            self.assertTrue(diff_guard_root.check(f_locals))

            # Check that tensor guards run well
            old_tensor = f_locals["bar"].y
            f_locals["bar"].y = torch.randn(5)
            self.assertFalse(root.check(f_locals))
            self.assertFalse(diff_guard_root.check(f_locals))
            f_locals["bar"].y = old_tensor

            # Original root should fail on foo changes, but diff_guard_root
            # should pass because it does not have foo guards on counter = 0. On
            # counter = 1, it should pass because we have caused a recompile
            # because of foo, causing it to recompile on foo.
            f_locals["foo"] = [3, 3]
            self.assertFalse(root.check(f_locals))
            if counter == 0:
                self.assertTrue(diff_guard_root.check(f_locals))
            else:
                self.assertFalse(diff_guard_root.check(f_locals))
            counter += 1

        class Bar:
            x = 4
            y = torch.randn(4)

        bar = Bar()

        def fn(x, foo, bar):
            return x + foo[0] + bar.x * bar.y

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            foo = (12.0, 13)
            opt_fn(x, foo, bar)

            foo = (10.0, 11)
            opt_fn(x, foo, bar)

    def test_guard_lookup_stats_api_records_cache_hits(self):
        def fn(x, y, scale: int):
            return (x + y) * scale

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        y = torch.randn(4)

        opt_fn(x, y, 3)
        guards.reset_guard_lookup_stats()
        before = guards.get_guard_lookup_stats()
        self.assertEqual(before["lookup_count"], 0)

        opt_fn(x, y, 3)

        after = guards.get_guard_lookup_stats()
        self.assertGreater(after["lookup_count"], 0)
        self.assertGreater(after["lookup_total_ns"], 0)
        self.assertGreater(after["slow_guard_ns"], 0)
        self.assertGreater(after["root_guard_count"], 0)
        self.assertGreater(after["root_guard_total_ns"], 0)
        self.assertFalse(after["unsafe_mock_guard_bypass_enabled"])
        self.assertEqual(after["unsafe_mock_guard_bypass_count"], 0)

    def test_guard_lookup_stats_records_accessor_types(self):
        class Foo:
            def __init__(self, x):
                self.x = x

        foo = Foo(1)

        guards.reset_guard_lookup_stats()

        attr_root = RootGuardManager()
        attr_root.getattr_manager("x", "x", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=1),
            equals_match_verbose_code_parts(1),
        )
        self.assertTrue(attr_root.check(foo))

        item_root = RootGuardManager()
        item_root.getitem_manager(0, "", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=1),
            equals_match_verbose_code_parts(1),
        )
        self.assertTrue(item_root.check([1]))

        type_root = RootGuardManager()
        type_root.getitem_manager("foo", "", foo, default_mgr_enum).type_manager(
            "", type(foo), default_mgr_enum
        ).add_lambda_guard(lambda x: x is type(foo), ["type(foo)"])
        self.assertTrue(type_root.check({"foo": foo}))

        generic_dict_root = RootGuardManager()
        generic_dict_root.get_generic_dict_manager(
            "foo.__dict__", foo.__dict__, default_mgr_enum
        ).add_lambda_guard(lambda x: x["x"] == 1, ["foo.__dict__['x'] == 1"])
        self.assertTrue(generic_dict_root.check(foo))

        dict_root = RootGuardManager()
        f_locals = {"d": {"a": 1}}
        dict_mgr = dict_root.getitem_manager(
            "d",
            "",
            f_locals["d"],
            torch._dynamo.guards.GuardManagerType.DICT_GUARD_MANAGER,
        )
        dict_mgr.get_key_manager(0, "", "a", default_mgr_enum).add_equals_match_guard(
            "a",
            ["dict.keys()[0] == a"],
        )
        dict_mgr.get_value_manager(0, "", 1, default_mgr_enum).add_equals_match_guard(
            1,
            ["d[0] == 1"],
        )
        self.assertTrue(dict_root.check(f_locals))

        accessor_stats = guards.get_guard_lookup_stats()[
            "root_guard_accessor_type_stats"
        ]
        for name in (
            "GetAttrGuardAccessor",
            "GetItemGuardAccessor",
            "TypeGuardAccessor",
            "DictGetItemGuardAccessor",
        ):
            self.assertIn(name, accessor_stats)
            self.assertGreater(accessor_stats[name]["count"], 0)
            self.assertGreater(accessor_stats[name]["inclusive_ns"], 0)

        detail_stats = guards.get_guard_lookup_stats()[
            "root_guard_accessor_detail_topk"
        ]
        for name in (
            "GetAttrGuardAccessor",
            "GetItemGuardAccessor",
            "GetGenericDictGuardAccessor",
            "TypeGuardAccessor",
            "DictGetItemGuardAccessor",
        ):
            matches = [
                stats
                for key, stats in detail_stats.items()
                if key.startswith(f"{name}:")
            ]
            self.assertTrue(matches)
            self.assertGreater(matches[0]["count"], 0)
            self.assertGreater(matches[0]["inclusive_ns"], 0)
            self.assertIn("self_ns", matches[0])
            self.assertIn("child_ns", matches[0])

    def test_unsafe_mock_guard_bypass_skips_slow_guard(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

def fn(x, y, scale: int):
    return (x + y) * scale

opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
x = torch.randn(4)
y = torch.randn(4)

opt_fn(x, y, 3)
guards.reset_guard_lookup_stats()
for _ in range(3):
    opt_fn(x, y, 3)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "lookup_count": stats["lookup_count"],
    "slow_guard_ns": stats["slow_guard_ns"],
    "unsafe_mock_guard_bypass_enabled": stats[
        "unsafe_mock_guard_bypass_enabled"
    ],
    "unsafe_mock_guard_bypass_count": stats[
        "unsafe_mock_guard_bypass_count"
    ],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_UNSAFE_MOCK_GUARD_BYPASS"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["lookup_count"], 0)
        self.assertTrue(stats["unsafe_mock_guard_bypass_enabled"])
        self.assertGreater(stats["unsafe_mock_guard_bypass_count"], 0)
        self.assertEqual(stats["slow_guard_ns"], 0)

    def test_guard_subtree_probe_is_off_by_default(self):
        guards.reset_guard_lookup_stats()
        stats = guards.get_guard_lookup_stats()

        self.assertEqual(stats["guard_subtree_probe_mode"], "off")
        self.assertEqual(stats["guard_subtree_probe_attempt"], 0)
        self.assertEqual(stats["guard_subtree_probe_entry_match"], 0)
        self.assertNotIn("guard_subtree_probe_top_paths", stats)

    def test_guard_subtree_probe_summary_records_shadow_entries(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    def forward(self, x):
        return self.seq(x)

model = Mod()
opt_model = torch.compile(model, backend="eager", fullgraph=True)
x = torch.randn(2, 4)

opt_model(x)
guards.reset_guard_lookup_stats()
for _ in range(3):
    opt_model(x)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "mode": stats["guard_subtree_probe_mode"],
    "attempt": stats["guard_subtree_probe_attempt"],
    "entry_match": stats["guard_subtree_probe_entry_match"],
    "shadow_ok": stats["guard_subtree_probe_shadow_ok"],
    "child_check_ns": stats["guard_subtree_probe_child_check_ns"],
    "has_top_paths": "guard_subtree_probe_top_paths" in stats,
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_SUBTREE_PROBE"] = "summary"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertEqual(stats["mode"], "summary")
        self.assertGreater(stats["attempt"], 0)
        self.assertGreater(stats["entry_match"], 0)
        self.assertGreater(stats["shadow_ok"], 0)
        self.assertGreater(stats["child_check_ns"], 0)
        self.assertFalse(stats["has_top_paths"])

    def test_guard_subtree_probe_detail_records_top_paths(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    def forward(self, x):
        return self.seq(x)

model = Mod()
opt_model = torch.compile(model, backend="eager", fullgraph=True)
x = torch.randn(2, 4)

opt_model(x)
guards.reset_guard_lookup_stats()
for _ in range(3):
    opt_model(x)

stats = guards.get_guard_lookup_stats()
top_paths = stats["guard_subtree_probe_top_paths"]
print(json.dumps({
    "mode": stats["guard_subtree_probe_mode"],
    "attempt": stats["guard_subtree_probe_attempt"],
    "entry_match": stats["guard_subtree_probe_entry_match"],
    "shadow_ok": stats["guard_subtree_probe_shadow_ok"],
    "top_path_count": len(top_paths),
    "top_path_values": list(top_paths.values()),
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_SUBTREE_PROBE"] = "detail"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertEqual(stats["mode"], "detail")
        self.assertGreater(stats["attempt"], 0)
        self.assertGreater(stats["entry_match"], 0)
        self.assertGreater(stats["shadow_ok"], 0)
        self.assertGreater(stats["top_path_count"], 0)
        self.assertTrue(
            any(value["attempt"] > 0 for value in stats["top_path_values"])
        )

    def test_guard_fast_plan_subtree_memo_records_hits(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
modules_mgr.add_dict_length_check_guard(
    len(model._modules), ["len(L['self']._modules) == 1"]
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "candidate": stats["guard_fastplan_candidate"],
    "shadow_pass": stats["guard_fastplan_shadow_pass"],
    "enable": stats["guard_fastplan_enable"],
    "hit": stats["guard_fastplan_hit"],
    "miss": stats["guard_fastplan_miss"],
    "disabled": stats["guard_fastplan_disabled"],
    "token_count_sum": stats["guard_fastplan_token_count_sum"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["candidate"], 0)
        self.assertGreater(stats["shadow_pass"], 0)
        self.assertGreater(stats["enable"], 0)
        self.assertGreater(stats["hit"], 0)
        self.assertEqual(stats["miss"], 0)
        self.assertEqual(stats["disabled"], 0)
        self.assertGreater(stats["token_count_sum"], 0)

    def test_guard_fast_plan_records_disabled_reasons(self):
        script = r"""
import functools
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

def equals_match(x, expected):
    return x == expected

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
modules_mgr.add_lambda_guard(
    functools.partial(equals_match, expected=model._modules),
    ["L['self']._modules == original"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
assert root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "candidate": stats["guard_fastplan_candidate"],
    "disabled": stats["guard_fastplan_disabled"],
    "reasons": stats["guard_fastplan_disabled_reasons"],
    "top_paths": stats["guard_fastplan_disabled_top_paths"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["candidate"], 0)
        self.assertGreater(stats["disabled"], 0)
        self.assertEqual(
            stats["reasons"].get("unsupported_leaf:LAMBDA_GUARD"), 1
        )
        self.assertIn("L['self']._modules", stats["top_paths"])
        self.assertEqual(
            stats["top_paths"]["L['self']._modules"]["reason"],
            "unsupported_leaf:LAMBDA_GUARD",
        )

    def test_guard_fast_plan_nested_modules_records_hits(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clean = Child()

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
clean_mgr = modules_mgr.getitem_manager(
    "clean",
    "L['self']._modules['clean']",
    model.clean,
    GuardManagerType.GUARD_MANAGER,
)
clean_dict_mgr = clean_mgr.get_generic_dict_manager(
    "L['self']._modules['clean'].__dict__",
    model.clean.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr = clean_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['clean']._modules",
    model.clean._modules,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr.add_dict_length_check_guard(
    len(model.clean._modules),
    ["len(L['self']._modules['clean']._modules) == 1"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "candidate_nested": stats["guard_fastplan_candidate_nested"],
    "shadow_pass_nested": stats["guard_fastplan_shadow_pass_nested"],
    "enable_nested": stats["guard_fastplan_enable_nested"],
    "hit_nested": stats["guard_fastplan_hit_nested"],
    "miss": stats["guard_fastplan_miss"],
    "disabled_nested": stats["guard_fastplan_disabled_nested"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["candidate_nested"], 0)
        self.assertGreater(stats["shadow_pass_nested"], 0)
        self.assertGreater(stats["enable_nested"], 0)
        self.assertGreater(stats["hit_nested"], 0)
        self.assertEqual(stats["miss"], 0)
        self.assertEqual(stats["disabled_nested"], 0)

    def test_guard_fast_plan_blocked_sibling_does_not_disable_clean_nested(self):
        script = r"""
import functools
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

def equals_match(x, expected):
    return x == expected

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clean = Child()
        self.blocked = Child()

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
clean_mgr = modules_mgr.getitem_manager(
    "clean",
    "L['self']._modules['clean']",
    model.clean,
    GuardManagerType.GUARD_MANAGER,
)
clean_dict_mgr = clean_mgr.get_generic_dict_manager(
    "L['self']._modules['clean'].__dict__",
    model.clean.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr = clean_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['clean']._modules",
    model.clean._modules,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr.add_dict_length_check_guard(
    len(model.clean._modules),
    ["len(L['self']._modules['clean']._modules) == 1"],
)

blocked_mgr = modules_mgr.getitem_manager(
    "blocked",
    "L['self']._modules['blocked']",
    model.blocked,
    GuardManagerType.GUARD_MANAGER,
)
blocked_dict_mgr = blocked_mgr.get_generic_dict_manager(
    "L['self']._modules['blocked'].__dict__",
    model.blocked.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
blocked_modules_mgr = blocked_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['blocked']._modules",
    model.blocked._modules,
    GuardManagerType.GUARD_MANAGER,
)
blocked_modules_mgr.add_lambda_guard(
    functools.partial(equals_match, expected=model.blocked._modules),
    ["L['self']._modules['blocked']._modules == original"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "candidate_top": stats["guard_fastplan_candidate_top"],
    "candidate_nested": stats["guard_fastplan_candidate_nested"],
    "partial_enable": stats["guard_fastplan_partial_enable"],
    "partial_hit": stats["guard_fastplan_partial_hit"],
    "enable_nested": stats["guard_fastplan_enable_nested"],
    "hit_nested": stats["guard_fastplan_hit_nested"],
    "disabled": stats["guard_fastplan_disabled"],
    "disabled_nested": stats["guard_fastplan_disabled_nested"],
    "reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["candidate_top"], 0)
        self.assertGreater(stats["candidate_nested"], 0)
        self.assertGreater(stats["partial_enable"], 0)
        self.assertGreater(stats["partial_hit"], 0)
        self.assertGreater(stats["enable_nested"], 0)
        self.assertGreater(stats["hit_nested"], 0)
        self.assertGreater(stats["disabled"], 0)
        self.assertGreater(stats["disabled_nested"], 0)
        self.assertGreaterEqual(
            stats["reasons"].get("unsupported_leaf:LAMBDA_GUARD", 0), 1
        )

    def test_guard_fast_plan_partial_hit_checks_unsupported_child(self):
        script = r"""
import functools
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

def is_original_dict(x, expected):
    return x is expected

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clean = Child()
        self.blocked = Child()

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = Parent()

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
parent_mgr = modules_mgr.getitem_manager(
    "parent",
    "L['self']._modules['parent']",
    model.parent,
    GuardManagerType.GUARD_MANAGER,
)
parent_dict_mgr = parent_mgr.get_generic_dict_manager(
    "L['self']._modules['parent'].__dict__",
    model.parent.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
parent_modules_mgr = parent_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['parent']._modules",
    model.parent._modules,
    GuardManagerType.GUARD_MANAGER,
)

clean_mgr = parent_modules_mgr.getitem_manager(
    "clean",
    "L['self']._modules['parent']._modules['clean']",
    model.parent.clean,
    GuardManagerType.GUARD_MANAGER,
)
clean_dict_mgr = clean_mgr.get_generic_dict_manager(
    "L['self']._modules['parent']._modules['clean'].__dict__",
    model.parent.clean.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr = clean_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['parent']._modules['clean']._modules",
    model.parent.clean._modules,
    GuardManagerType.GUARD_MANAGER,
)
clean_modules_mgr.add_dict_length_check_guard(
    len(model.parent.clean._modules),
    ["len(L['self']._modules['parent']._modules['clean']._modules) == 1"],
)

blocked_mgr = parent_modules_mgr.getitem_manager(
    "blocked",
    "L['self']._modules['parent']._modules['blocked']",
    model.parent.blocked,
    GuardManagerType.GUARD_MANAGER,
)
blocked_dict_mgr = blocked_mgr.get_generic_dict_manager(
    "L['self']._modules['parent']._modules['blocked'].__dict__",
    model.parent.blocked.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
blocked_modules_mgr = blocked_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules['parent']._modules['blocked']._modules",
    model.parent.blocked._modules,
    GuardManagerType.GUARD_MANAGER,
)
blocked_modules_mgr.add_lambda_guard(
    functools.partial(is_original_dict, expected=model.parent.blocked._modules),
    ["blocked._modules is original"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

model.parent.blocked._modules = {}
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "partial_enable": stats["guard_fastplan_partial_enable"],
    "partial_hit": stats["guard_fastplan_partial_hit"],
    "miss": stats["guard_fastplan_miss"],
    "reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["partial_enable"], 0)
        self.assertGreater(stats["partial_hit"], 0)
        self.assertEqual(stats["miss"], 0)
        self.assertGreaterEqual(
            stats["reasons"].get("unsupported_leaf:LAMBDA_GUARD", 0), 1
        )

    def test_guard_fast_plan_tensor_match_token_hits_and_checks_metadata(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

model = torch.nn.Sequential(torch.nn.Linear(4, 4))
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
linear_mgr = modules_mgr.getitem_manager(
    "0",
    "L['self']._modules['0']",
    model[0],
    GuardManagerType.GUARD_MANAGER,
)
linear_dict_mgr = linear_mgr.get_generic_dict_manager(
    "L['self']._modules['0'].__dict__",
    model[0].__dict__,
    GuardManagerType.GUARD_MANAGER,
)
parameters_mgr = linear_dict_mgr.getitem_manager(
    "_parameters",
    "L['self']._modules['0']._parameters",
    model[0]._parameters,
    GuardManagerType.GUARD_MANAGER,
)
weight_mgr = parameters_mgr.getitem_manager(
    "weight",
    "L['self']._modules['0']._parameters['weight']",
    model[0].weight,
    GuardManagerType.GUARD_MANAGER,
)
weight_mgr.add_tensor_match_guard(
    model[0].weight,
    list(model[0].weight.size()),
    list(model[0].weight.stride()),
    "L['self']._modules['0']._parameters['weight']",
    ["check_tensor(weight)"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

with torch.no_grad():
    model[0].weight.add_(1)
assert root.check(f_locals)

model[0].weight.requires_grad_(False)
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "tensor_shadow": stats["guard_fastplan_tensor_token_shadow"],
    "tensor_hit": stats["guard_fastplan_tensor_token_hit"],
    "tensor_miss": stats["guard_fastplan_tensor_token_miss"],
    "miss_reasons": stats["guard_fastplan_tensor_token_miss_reasons"],
    "fastplan_miss": stats["guard_fastplan_miss"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["tensor_shadow"], 0)
        self.assertGreater(stats["tensor_hit"], 0)
        self.assertGreater(stats["tensor_miss"], 0)
        self.assertGreater(stats["fastplan_miss"], 0)
        self.assertGreater(
            stats["miss_reasons"].get("requires_grad", 0), 0
        )
        self.assertEqual(
            stats["disabled_reasons"].get("unsupported_leaf:TENSOR_MATCH", 0),
            0,
        )

    def test_guard_fast_plan_tensor_match_token_skips_attr_sources(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other_tensor = torch.ones(2)

model = Mod()
container = torch.nn.Module()
container.child = model
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", container, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", container.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    container._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    model,
    GuardManagerType.GUARD_MANAGER,
)
cached_mgr = child_mgr.getattr_manager(
    "other_tensor",
    "L['self']._modules['child'].other_tensor",
    model.other_tensor,
    GuardManagerType.GUARD_MANAGER,
)
cached_mgr.add_tensor_match_guard(
    model.other_tensor,
    list(model.other_tensor.size()),
    list(model.other_tensor.stride()),
    "L['self']._modules['child'].other_tensor",
    ["check_tensor(cached)"],
)

f_locals = {"self": container}
guards.reset_guard_lookup_stats()
for _ in range(4):
    assert root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "tensor_shadow": stats["guard_fastplan_tensor_token_shadow"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertEqual(stats["tensor_shadow"], 0)
        self.assertGreater(
            stats["disabled_reasons"].get("unsupported_leaf:TENSOR_MATCH", 0),
            0,
        )

    def test_guard_fast_plan_tensor_match_token_supports_cached_tensor_attr(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_tensor = torch.ones(2)

model = Mod()
container = torch.nn.Module()
container.child = model
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", container, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", container.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    container._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    model,
    GuardManagerType.GUARD_MANAGER,
)
cached_mgr = child_mgr.getattr_manager(
    "_cached_tensor",
    "L['self']._modules['child']._cached_tensor",
    model._cached_tensor,
    GuardManagerType.GUARD_MANAGER,
)
cached_mgr.add_tensor_match_guard(
    model._cached_tensor,
    list(model._cached_tensor.size()),
    list(model._cached_tensor.stride()),
    "L['self']._modules['child']._cached_tensor",
    ["check_tensor(cached)"],
)

f_locals = {"self": container}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

model._cached_tensor.resize_(3)
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "tensor_shadow": stats["guard_fastplan_tensor_token_shadow"],
    "tensor_hit": stats["guard_fastplan_tensor_token_hit"],
    "tensor_miss": stats["guard_fastplan_tensor_token_miss"],
    "miss_reasons": stats["guard_fastplan_tensor_token_miss_reasons"],
    "fastplan_miss": stats["guard_fastplan_miss"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["tensor_shadow"], 0)
        self.assertGreater(stats["tensor_hit"], 0)
        self.assertGreater(stats["tensor_miss"], 0)
        self.assertGreater(stats["fastplan_miss"], 0)
        self.assertGreater(stats["miss_reasons"].get("size", 0), 0)
        self.assertEqual(
            stats["disabled_reasons"].get("unsupported_leaf:TENSOR_MATCH", 0),
            0,
        )

    def test_guard_fast_plan_tensor_match_token_supports_scale_attr(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.ones(2)

model = Mod()
container = torch.nn.Module()
container.child = model
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", container, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", container.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    container._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    model,
    GuardManagerType.GUARD_MANAGER,
)
scale_mgr = child_mgr.getattr_manager(
    "scale",
    "L['self']._modules['child'].scale",
    model.scale,
    GuardManagerType.GUARD_MANAGER,
)
scale_mgr.add_tensor_match_guard(
    model.scale,
    list(model.scale.size()),
    list(model.scale.stride()),
    "L['self']._modules['child'].scale",
    ["check_tensor(scale)"],
)

f_locals = {"self": container}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

model.scale.resize_(3)
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "tensor_shadow": stats["guard_fastplan_tensor_token_shadow"],
    "tensor_hit": stats["guard_fastplan_tensor_token_hit"],
    "tensor_miss": stats["guard_fastplan_tensor_token_miss"],
    "miss_reasons": stats["guard_fastplan_tensor_token_miss_reasons"],
    "fastplan_miss": stats["guard_fastplan_miss"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["tensor_shadow"], 0)
        self.assertGreater(stats["tensor_hit"], 0)
        self.assertGreater(stats["tensor_miss"], 0)
        self.assertGreater(stats["fastplan_miss"], 0)
        self.assertGreater(stats["miss_reasons"].get("size", 0), 0)
        self.assertEqual(
            stats["disabled_reasons"].get("unsupported_leaf:TENSOR_MATCH", 0),
            0,
        )

    def test_guard_fast_plan_supports_no_tensor_aliasing_token(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.ones(2)
        self.b = torch.zeros(2)

container = torch.nn.Module()
container.child = Child()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", container, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", container.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    container._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    container.child,
    GuardManagerType.GUARD_MANAGER,
)
a_mgr = child_mgr.getattr_manager(
    "a",
    "L['self']._modules['child'].a",
    container.child.a,
    GuardManagerType.GUARD_MANAGER,
)
b_mgr = child_mgr.getattr_manager(
    "b",
    "L['self']._modules['child'].b",
    container.child.b,
    GuardManagerType.GUARD_MANAGER,
)
guards.install_no_tensor_aliasing_guard(
    [a_mgr, b_mgr],
    ["a", "b"],
    ["no_aliasing(a, b)"],
)

f_locals = {"self": container}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

container.child.b = container.child.a
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "fastplan_hit": stats["guard_fastplan_hit"],
    "fastplan_miss": stats["guard_fastplan_miss"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["fastplan_hit"], 0)
        self.assertGreater(stats["fastplan_miss"], 0)
        self.assertEqual(
            stats["disabled_reasons"].get(
                "unsupported_leaf:NO_TENSOR_ALIASING", 0
            ),
            0,
        )

    def test_guard_fast_plan_supports_object_aliasing_token(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.ones(2)
        self.b = self.a

container = torch.nn.Module()
container.child = Child()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", container, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", container.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    container._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    container.child,
    GuardManagerType.GUARD_MANAGER,
)
a_mgr = child_mgr.getattr_manager(
    "a",
    "L['self']._modules['child'].a",
    container.child.a,
    GuardManagerType.GUARD_MANAGER,
)
b_mgr = child_mgr.getattr_manager(
    "b",
    "L['self']._modules['child'].b",
    container.child.b,
    GuardManagerType.GUARD_MANAGER,
)
guards.install_object_aliasing_guard(
    a_mgr,
    b_mgr,
    ["aliasing(a, b)"],
)

f_locals = {"self": container}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

container.child.b = torch.zeros(2)
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "fastplan_hit": stats["guard_fastplan_hit"],
    "fastplan_miss": stats["guard_fastplan_miss"],
    "disabled_reasons": stats["guard_fastplan_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["fastplan_hit"], 0)
        self.assertGreater(stats["fastplan_miss"], 0)
        self.assertEqual(
            stats["disabled_reasons"].get(
                "unsupported_leaf:OBJECT_ALIASING", 0
            ),
            0,
        )

    def test_guard_last_success_shadow_records_lookup_attempts(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

def fn(x):
    return x + 1

compiled = torch.compile(fn, backend="eager")
x = torch.ones(4)
guards.reset_guard_lookup_stats()
for _ in range(5):
    out = compiled(x)
    assert torch.equal(out, x + 1)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "lookup_count": stats["lookup_count"],
    "attempt": stats["guard_last_success_shadow_attempt"],
    "success": stats["guard_last_success_shadow_success"],
    "incomplete": stats["guard_last_success_shadow_incomplete"],
    "support_check_ns": stats["guard_last_success_support_check_ns"],
    "token_cap_count_sum": stats["guard_last_success_token_cap_count_sum"],
    "token_cap_count_max": stats["guard_last_success_token_cap_count_max"],
    "compare_mismatch": stats["guard_last_success_compare_mismatch"],
    "mismatch_reasons": stats["guard_last_success_mismatch_reasons"],
    "mismatch_token_kinds": stats["guard_last_success_mismatch_token_kinds"],
    "disabled_reasons": stats["guard_last_success_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["lookup_count"], 0)
        self.assertGreater(stats["attempt"], 0)
        self.assertGreater(stats["incomplete"] + stats["success"], 0)
        self.assertGreaterEqual(stats["support_check_ns"], 0)
        self.assertGreaterEqual(stats["token_cap_count_sum"], 0)
        self.assertGreaterEqual(stats["token_cap_count_max"], 0)
        self.assertGreaterEqual(stats["compare_mismatch"], 0)
        self.assertIsInstance(stats["mismatch_reasons"], dict)
        self.assertIsInstance(stats["mismatch_token_kinds"], dict)

    def test_guard_last_success_supports_default_device_token(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

def fn(x):
    return x + 1

compiled = torch.compile(fn, backend="eager")
x = torch.ones(4)
guards.reset_guard_lookup_stats()
for _ in range(5):
    out = compiled(x)
    assert torch.equal(out, x + 1)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "attempt": stats["guard_last_success_shadow_attempt"],
    "success": stats["guard_last_success_shadow_success"],
    "disabled_reasons": stats["guard_last_success_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["attempt"], 0)
        self.assertGreaterEqual(stats["success"], 0)
        self.assertEqual(
            stats["disabled_reasons"].get("unsupported_leaf:DEFAULT_DEVICE", 0),
            0,
        )

    def test_guard_last_success_supports_global_state_token(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

def fn(x):
    return x + 1

compiled = torch.compile(fn, backend="eager")
x = torch.ones(4)
guards.reset_guard_lookup_stats()
for _ in range(5):
    out = compiled(x)
    assert torch.equal(out, x + 1)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "attempt": stats["guard_last_success_shadow_attempt"],
    "success": stats["guard_last_success_shadow_success"],
    "disabled_reasons": stats["guard_last_success_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["attempt"], 0)
        self.assertGreaterEqual(stats["success"], 0)
        self.assertEqual(
            stats["disabled_reasons"].get("unsupported_leaf:GLOBAL_STATE", 0),
            0,
        )

    def test_guard_last_success_supports_torch_function_mode_stack_token(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards

def fn(x):
    return x + 1

compiled = torch.compile(fn, backend="eager")
x = torch.ones(4)
guards.reset_guard_lookup_stats()
for _ in range(5):
    out = compiled(x)
    assert torch.equal(out, x + 1)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "attempt": stats["guard_last_success_shadow_attempt"],
    "success": stats["guard_last_success_shadow_success"],
    "disabled_reasons": stats["guard_last_success_disabled_reasons"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertGreater(stats["attempt"], 0)
        self.assertGreaterEqual(stats["success"], 0)
        self.assertEqual(
            stats["disabled_reasons"].get(
                "unsupported_leaf:TORCH_FUNCTION_MODE_STACK", 0
            ),
            0,
        )

    def test_guard_fast_plan_subtree_memo_miss_disables_and_falls_back(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
modules_mgr.add_dict_length_check_guard(
    len(model._modules), ["len(L['self']._modules) == 1"]
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

model.extra = torch.nn.ReLU()
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "hit": stats["guard_fastplan_hit"],
    "miss": stats["guard_fastplan_miss"],
    "disabled": stats["guard_fastplan_disabled"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["hit"], 0)
        self.assertGreater(stats["miss"], 0)
        self.assertGreater(stats["disabled"], 0)

    def test_guard_fast_plan_subtree_memo_detects_list_item_change(self):
        script = r"""
import json
import torch
from torch._C._dynamo import guards
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType

class Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.items = [1]

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = Child()

model = Mod()
root = RootGuardManager()
self_mgr = root.getitem_manager(
    "self", "L['self']", model, GuardManagerType.GUARD_MANAGER
)
self_dict_mgr = self_mgr.get_generic_dict_manager(
    "L['self'].__dict__", model.__dict__, GuardManagerType.GUARD_MANAGER
)
modules_mgr = self_dict_mgr.getitem_manager(
    "_modules",
    "L['self']._modules",
    model._modules,
    GuardManagerType.GUARD_MANAGER,
)
child_mgr = modules_mgr.getitem_manager(
    "child",
    "L['self']._modules['child']",
    model.child,
    GuardManagerType.GUARD_MANAGER,
)
child_dict_mgr = child_mgr.get_generic_dict_manager(
    "L['self']._modules['child'].__dict__",
    model.child.__dict__,
    GuardManagerType.GUARD_MANAGER,
)
items_mgr = child_dict_mgr.getitem_manager(
    "items",
    "L['self']._modules['child'].items",
    model.child.items,
    GuardManagerType.GUARD_MANAGER,
)
item_mgr = items_mgr.getitem_manager(
    0,
    "L['self']._modules['child'].items[0]",
    model.child.items[0],
    GuardManagerType.GUARD_MANAGER,
)
item_mgr.add_equals_match_guard(
    model.child.items[0],
    ["L['self']._modules['child'].items[0] == 1"],
)

f_locals = {"self": model}
guards.reset_guard_lookup_stats()
for _ in range(6):
    assert root.check(f_locals)

model.child.items[0] = 2
assert not root.check(f_locals)

stats = guards.get_guard_lookup_stats()
print(json.dumps({
    "enabled": stats["guard_fastplan_enabled"],
    "hit": stats["guard_fastplan_hit"],
    "miss": stats["guard_fastplan_miss"],
    "disabled": stats["guard_fastplan_disabled"],
}))
"""
        env = os.environ.copy()
        env["TORCHDYNAMO_GUARD_FAST_PLAN"] = "1"
        out = subprocess.check_output(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )
        stats = json.loads(out.splitlines()[-1])

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["hit"], 0)
        self.assertGreater(stats["miss"], 0)
        self.assertGreater(stats["disabled"], 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
