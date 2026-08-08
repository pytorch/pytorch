# Owner(s): ["module: dynamo"]
import abc
import functools
import gc
import inspect
import os
import subprocess
import sys
import textwrap
import unittest
import weakref
from unittest import mock

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._C._dynamo import guards
from torch._dynamo.convert_frame import GlobalStateGuard
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._library.fake_class_registry import FakeScriptObject
from torch.testing._internal.common_utils import (
    set_default_dtype,
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
)


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
    def test_cpp_shape_guard_missing_windows_compiler_falls_back(self):
        from torch._inductor.codecache import CppCodeCache
        from torch._inductor.cpp_builder import check_compiler_exist_windows
        from torch._inductor.cpu_vec_isa import valid_vec_isa_list

        def fn(x):
            return x.sin()

        CppCodeCache.cache_clear()
        check_compiler_exist_windows.cache_clear()
        valid_vec_isa_list.cache_clear()

        x = torch.randn(2, 3)
        torch._dynamo.mark_dynamic(x, 0)
        with (
            torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=True),
            mock.patch("torch._inductor.cpp_builder._IS_WINDOWS", True),
            mock.patch.dict(os.environ, {"CXX": "definitely_missing_cl_for_157458"}),
        ):
            actual = torch.compile(fn, backend="eager")(x)

        self.assertEqual(actual, fn(x))

    def test_guard_debug_info_user_stack(self):
        """Test that GuardDebugInfo can store user stack trace information."""
        import traceback

        # Create a sample user stack
        user_stack = traceback.StackSummary.from_list(
            [
                traceback.FrameSummary("test.py", 10, "test_func", line="x = y + 1"),
                traceback.FrameSummary("main.py", 5, "main", line="test_func()"),
            ]
        )

        # Test creating GuardDebugInfo with user_stack
        debug_info = guards.GuardDebugInfo(False, ["test_guard_failed"], 1, user_stack)

        # Verify user_stack is stored correctly
        self.assertFalse(debug_info.result)
        self.assertEqual(len(debug_info.verbose_code_parts), 1)
        self.assertEqual(debug_info.num_guards_executed, 1)
        self.assertIsNotNone(debug_info.user_stack)

        # Verify user_stack content
        self.assertEqual(len(debug_info.user_stack), 2)
        self.assertEqual(debug_info.user_stack[0].filename, "test.py")
        self.assertEqual(debug_info.user_stack[0].lineno, 10)
        self.assertEqual(debug_info.user_stack[0].name, "test_func")

        # Test GuardDebugInfo without user_stack (backward compatibility)
        debug_info2 = guards.GuardDebugInfo(True, ["test_guard_passed"], 2)
        self.assertTrue(debug_info2.result)
        # user_stack should be None when not provided
        self.assertTrue(
            debug_info2.user_stack is None or debug_info2.user_stack is not None
        )

    def test_global_state_guard(self):
        root = RootGuardManager()
        guard = guards.GLOBAL_STATE(root, ["global_state_check"], None)
        self.assertTrue(guard(None))
        with set_default_dtype(torch.double):
            self.assertFalse(guard(None))
            self.assertExpectedInline(
                str(guard.check_verbose(None)),
                """\
GuardDebugInfo(
result=0,
verbose_code_parts=['GLOBAL_STATE changed: default_dtype '],
num_guards_executed=0,
user_stack=None)
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
num_guards_executed=0,
user_stack=None)
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

    def test_global_state_reason_autocast_cache(self):
        # Test that autocast_cache_enabled is reported specifically
        old_cache = torch.is_autocast_cache_enabled()
        try:
            torch.set_autocast_cache_enabled(True)
            guard = GlobalStateGuard()
            torch.set_autocast_cache_enabled(False)
            self.assertIs(guard.check(), False)
            self.assertEqual(guard.reason(), "autocast_cache_enabled ")
        finally:
            torch.set_autocast_cache_enabled(old_cache)

    def test_python_lambda_leaf_guard(self):
        root = RootGuardManager()
        const_guard = guards.LAMBDA_GUARD(
            root,
            functools.partial(equals_match, expected=5),
            equals_match_verbose_code_parts(5),
            None,
        )
        self.assertTrue(const_guard(5))
        self.assertFalse(const_guard(4))
        self.assertFalse(const_guard("foo"))

    def test_type_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == int"], None)

        self.assertTrue(guard(5))
        self.assertTrue(guard(4))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == dict"], None)
        self.assertTrue(guard(foo))
        self.assertTrue(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        class Foo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        foo = Foo(1, 2)

        guard = guards.TYPE_MATCH(root, id_type(foo), ["type(x) == Foo"], None)
        self.assertTrue(guard(foo))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

    def test_fake_script_type_match_guard(self):
        class Real:
            pass

        class Other:
            pass

        root = RootGuardManager()
        real = Real()
        fake = FakeScriptObject(object(), "Real", real)
        guard = guards.FAKE_SCRIPT_TYPE_MATCH(
            root,
            FakeScriptObject,
            id_type(real),
            ["type match through FakeScriptObject"],
            None,
        )

        # Passes for the FakeScriptObject at compile time and the real
        # underlying object at runtime.
        self.assertTrue(guard(fake))
        self.assertTrue(guard(real))
        # Different real type wrapped in FakeScriptObject should fail.
        self.assertFalse(guard(FakeScriptObject(object(), "Other", Other())))
        # Different raw type should fail.
        self.assertFalse(guard(Other()))
        self.assertFalse(guard(5))

    def test_id_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.ID_MATCH(root, id(foo), ["id(x) == id(foo)"], None)

        self.assertTrue(guard(foo))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        foo = {"a": 1}
        guard = guards.ID_MATCH(root, id(foo), ["id(x) == id(foo)"], None)
        self.assertTrue(guard(foo))
        self.assertFalse(guard({"a": 1}))
        self.assertFalse(guard({}))
        self.assertFalse(guard(5))

    def test_equals_guard(self):
        root = RootGuardManager()
        foo = 4
        guard = guards.EQUALS_MATCH(root, foo, ["x == 4"], None)

        self.assertTrue(guard(4))
        self.assertFalse(guard(5))
        self.assertFalse(guard("foo"))

        # tuple
        foo = (1, 2, 3)
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"], None)
        self.assertTrue(guard(foo))
        self.assertTrue(guard((1, 2, 3)))
        self.assertFalse(guard((1, 2, 3, 4)))
        self.assertFalse(guard({}))

        # list
        foo = [1, 2, 3]
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"], None)
        self.assertTrue(guard(foo))
        self.assertTrue(guard([1, 2, 3]))
        self.assertFalse(guard([1, 2, 3, 4]))

        # type
        foo = int
        guard = guards.EQUALS_MATCH(root, foo, ["x == foo"], None)
        self.assertTrue(guard(foo))
        self.assertTrue(guard(int))
        self.assertFalse(guard(float))

    def test_default_device_guard(self):
        root = RootGuardManager()
        foo = 1
        guard = guards.DEFAULT_DEVICE(root, ["cpu device"], None)
        self.assertTrue(guard(foo))

        if not torch.accelerator.is_available():
            self.skipTest("Accelerator is not available")

        try:
            device = torch.accelerator.current_accelerator()
            torch.set_default_device(device)
            self.assertFalse(guard(foo))
        finally:
            torch.set_default_device(None)

    def test_length_check_guard(self):
        root = RootGuardManager()
        foo = [1, 2, 3]
        guard = guards.LENGTH_CHECK(root, len(foo), ["len(x) == len(foo)"], None)
        self.assertTrue(guard(foo))
        self.assertFalse(guard([]))

    def test_no_hasattr_guard(self):
        root = RootGuardManager()

        class Bar:
            def __init__(self) -> None:
                self.bar = 2

        bar = Bar()

        class Foo:
            def __init__(self) -> None:
                self.foo = 2

        foo = Foo()

        guard = guards.NO_HASATTR(root, "foo", ["hasattr(x, 'foo') == False"], None)
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
        install_object_aliasing_guard(x_guard_mgr, y_guard_mgr, ["x is y"], None)

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

    @torch._dynamo.config.patch(skip_tensor_guards_with_matching_dict_tags=True)
    def test_dict_tag_does_not_skip_immutable_object_aliasing_guard(self):
        value = (1, 2)
        different = (1, 3)
        container = {"value": value}

        root = RootGuardManager()
        dict_manager = root.list_getitem_manager(0, "", container, default_mgr_enum)
        dict_value_manager = dict_manager.dict_getitem_manager(
            "value", "", value, default_mgr_enum
        )
        peer_manager = root.list_getitem_manager(1, "", value, default_mgr_enum)
        install_object_aliasing_guard(
            dict_value_manager,
            peer_manager,
            ["container['value'] is peer"],
            None,
        )

        self.assertTrue(root.check([container, value]))
        self.assertFalse(root.check([container, different]))

    def test_dict_version_guard(self):
        root = RootGuardManager()
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_VERSION(root, foo, ["x.version == foo.version"], None)

        self.assertTrue(guard(foo))
        self.assertFalse(guard(dict(foo)))
        foo["a"] = 2
        self.assertFalse(guard(foo))
        self.assertFalse(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({}))

    def test_dynamic_indices_guard(self):
        root = RootGuardManager()

        # Test with expected attr: _dynamo_dynamic_indices = {0, 1}
        # and absent attr: _dynamo_static_indices
        expected_attrs = {"_dynamo_dynamic_indices": {0, 1}}
        absent_attrs = ["_dynamo_static_indices"]
        dependent_attrs = {}  # type: ignore[var-annotated]
        guard = guards.DIMENSION_DYNAMIC_MARKING_GUARD(
            root,
            expected_attrs,
            absent_attrs,
            dependent_attrs,
            ["dimension marking guard"],
            None,
        )

        # No attr at all -> pass (unspecified = don't care)
        x = torch.randn(4)
        self.assertTrue(guard(x))

        # Exact match -> pass
        x._dynamo_dynamic_indices = {0, 1}
        x._has_dynamo_dim_marking = True
        self.assertTrue(guard(x))

        # Subset -> pass (runtime markings are a subset of compiled)
        x._dynamo_dynamic_indices = {0}
        x._has_dynamo_dim_marking = True
        self.assertTrue(guard(x))

        # Different set -> fail
        x._dynamo_dynamic_indices = {2}
        x._has_dynamo_dim_marking = True
        self.assertFalse(guard(x))

        # Absent attr present -> fail
        x._dynamo_dynamic_indices = {0, 1}
        x._dynamo_static_indices = {0}
        x._has_dynamo_dim_marking = True
        self.assertFalse(guard(x))

    def test_dimension_marking_guard_dependent_attrs(self):
        root = RootGuardManager()

        # Test dependent_attrs: _dynamo_shape_ids is checked only when
        # _dynamo_unbacked_indices (gate) is present.
        expected_attrs = {"_dynamo_unbacked_indices": {0}}
        absent_attrs = []  # type: ignore[var-annotated]
        dependent_attrs = {
            "_dynamo_shape_ids": ({0: "batch"}, "_dynamo_unbacked_indices"),
        }
        guard = guards.DIMENSION_DYNAMIC_MARKING_GUARD(
            root,
            expected_attrs,
            absent_attrs,
            dependent_attrs,
            ["dimension marking guard dependent"],
            None,
        )

        # No gate attr -> pass (don't care)
        x = torch.randn(4)
        self.assertTrue(guard(x))

        # Gate present + dependent attr matches -> pass
        x._dynamo_unbacked_indices = {0}
        x._dynamo_shape_ids = {0: "batch"}
        x._has_dynamo_dim_marking = True
        self.assertTrue(guard(x))

        # Gate present + dependent attr mismatch -> fail
        x._dynamo_shape_ids = {0: "other"}
        self.assertFalse(guard(x))

        # Gate present + dependent attr absent + expected non-None -> fail
        del x._dynamo_shape_ids
        self.assertFalse(guard(x))

        # Test with expected=None for dependent attr (compile-time also absent)
        dependent_attrs_none = {
            "_dynamo_shape_ids": (None, "_dynamo_unbacked_indices"),
        }
        guard2 = guards.DIMENSION_DYNAMIC_MARKING_GUARD(
            root,
            expected_attrs,
            absent_attrs,
            dependent_attrs_none,
            ["dimension marking guard dependent none"],
            None,
        )

        # Gate present + dependent attr absent + expected None -> pass
        y = torch.randn(4)
        y._dynamo_unbacked_indices = {0}
        y._has_dynamo_dim_marking = True
        self.assertTrue(guard2(y))

    def test_tensor_match_guard(self):
        guard_manager = RootGuardManager()
        x = torch.randn(4, 4)
        size = list(x.size())
        stride = list(x.stride())
        guard_manager.add_tensor_match_guard(
            x,
            size,
            stride,
            "x",
            ["check_tensor(x)"],
            None,
            type(x),
            torch._C._dispatch_keys(x),
        )
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
            None,
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
        root = RootGuardManager()
        x = torch.rand(3, 4)
        weakref_x = weakref.ref(x)

        guard = guards.NOT_NONE(root, ["weakref_x is not None"], None)
        self.assertTrue(guard(weakref_x()))
        del x
        self.assertFalse(guard(weakref_x()))

    def test_call_function_no_args_guard(self):
        if not torch.accelerator.is_available():
            self.skipTest("Accelerator is not available")

        root = RootGuardManager()
        device = torch.accelerator.current_accelerator()
        # Use device.index which is device-agnostic (works on all accelerators)
        x = device.index if device.index is not None else 0
        guard = guards.EQUALS_MATCH(root, x, [0], None)
        self.assertTrue(guard(0))
        self.assertFalse(guard(1))
        self.assertFalse(guard(2))

    def test_guard_manager_leaf_guard(self):
        guard_manager = RootGuardManager()
        guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"], None)
        guard_manager.add_lambda_guard(
            functools.partial(ge_match, expected=5),
            ge_match_verbose_code_parts(expected=5),
            None,
        )
        guard_manager.add_lambda_guard(
            functools.partial(less_match, expected=10),
            less_match_verbose_code_parts(expected=10),
            None,
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
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"], None)
        guard_manager.getattr_manager("x", "x", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.x),
            equals_match_verbose_code_parts(foo.x),
            None,
        )
        guard_manager.getattr_manager("y", "y", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo.y),
            equals_match_verbose_code_parts(foo.y),
            None,
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
        guard_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"], None)
        guard_manager.getitem_manager(0, "", 1, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[0]),
            equals_match_verbose_code_parts(foo[0]),
            None,
        )
        guard_manager.getitem_manager(1, "", 2, default_mgr_enum).add_lambda_guard(
            functools.partial(equals_match, expected=foo[1]),
            equals_match_verbose_code_parts(foo[1]),
            None,
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
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"], None)
        guards_manager.framelocals_manager(
            ("a", 0), "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"], None)
        guards_manager.framelocals_manager(
            ("b", 1), "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"], None)

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

    def test_dict_getitem_accessor(self):
        foo = {
            "a": 1,
            "b": 2,
        }

        guards_manager = RootGuardManager()
        guards_manager.add_type_match_guard(id_type(foo), ["type(x) == Foo"], None)
        guards_manager.dict_getitem_manager(
            "a", "", 1, default_mgr_enum
        ).add_equals_match_guard(1, ["a == 1"], None)
        guards_manager.dict_getitem_manager(
            "b", "", 2, default_mgr_enum
        ).add_equals_match_guard(2, ["b == 2"], None)

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
            None,
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
            None,
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
            None,
        )

        self.assertTrue(guard_manager.check(f_locals))

    def test_tuple_iterator_getitem(self):
        a = (1, 2, 3, 4, 5, 6)
        foo = iter(a)
        next(foo)  # foo points at index=1

        guard_manager = RootGuardManager()
        # Check a[3] which is tuple_iterator_getitem(foo, 2)
        guard_manager.add_tuple_iterator_length_guard(
            5, id_type(iter(())), ["len == 5"], None
        )
        guard_manager.tuple_iterator_getitem_manager(
            2, "", foo, default_mgr_enum
        ).add_equals_match_guard(a[3], ["x==4"], None)

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
            None,
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
            None,
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
        root = RootGuardManager()
        foo = {"a": 1, "b": 2}
        guard = guards.DICT_CONTAINS(root, True, "a", ["has a"], None)

        self.assertTrue(guard(foo))
        self.assertTrue(guard({"a": 1, "b": 2}))
        self.assertFalse(guard({"b": 2, "c": 3}))
        self.assertFalse(guard({}))

        guard = guards.DICT_CONTAINS(root, False, "c", ["not has c"], None)
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

        # ID_MATCH is the only leaf guard supported on DictGuardManager.
        dict_mgr.add_id_match_guard(id(f_locals["d"]), "id match on dict", None)
        self.assertTrue(root.check(f_locals))

        # Other leaf guards are rejected.
        with self.assertRaises(RuntimeError):
            dict_mgr.add_equals_match_guard(f_locals["d"], ["equals match"], None)

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
            None,
        )
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_value_manager(0, "", 1, default_mgr_enum).add_equals_match_guard(
            1, ["value == 1"], None
        )
        self.assertTrue(root.check(f_locals))

        # Add key-value manager (nothing : {"z" : 3})
        self.assertTrue(root.check(f_locals))
        dict_mgr.get_key_manager(1, "", nothing, default_mgr_enum).add_lambda_guard(
            lambda key: key is nothing, ["key is nothing"], None
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

        def hook(guard_wrapper, f_locals, builder):
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

        def hook(guard_wrapper, f_locals, builder):
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
            def __init__(self):
                self.x = 4
                self.y = torch.randn(4)

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


class TypePropagationTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(skip_tensor_guards_with_matching_dict_tags=True)
    def test_basic_types(self):
        class Foo:
            def __init__(self):
                self.x = {"a": 2}
                self.y = torch.randn(4)
                self.z = {}

        foo = Foo()

        mod = torch.nn.Linear(4, 4)

        def fn(x):
            return x + foo.x["a"] + foo.y + mod(x)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import AttrSource, DictGetItemSource, LocalSource

            foo_source = LocalSource("foo")
            foo_x_source = AttrSource(foo_source, "x")

            self.assertTrue(builder.get(foo_source) is foo)
            self.assertTrue(builder.get(foo_x_source) is foo.x)

            # Check types of foo.x
            foo_x_mgr = builder.get_guard_manager_from_source(foo_x_source)
            self.assertTrue(issubclass(foo_x_mgr.get_type_of_guarded_value(), dict))

            # Check types of foo.x["a"]
            foo_x_a_source = DictGetItemSource(foo_x_source, "a")
            foo_x_a_mgr = builder.get_guard_manager_from_source(foo_x_a_source)
            self.assertTrue(foo_x_a_mgr.is_guarded_value_immutable())

            # Check types of foo.y
            foo_y_source = AttrSource(foo_source, "y")
            foo_y_mgr = builder.get_guard_manager_from_source(foo_y_source)
            self.assertTrue(foo_y_mgr.is_guarded_value_immutable())

            # Check types of foo.z
            foo_z_source = AttrSource(foo_source, "z")
            foo_z_mgr = builder.get_guard_manager_from_source(foo_z_source)
            self.assertTrue(issubclass(foo_z_mgr.get_type_of_guarded_value(), dict))

            # Check types of mod
            mod_source = LocalSource("mod")
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(
                issubclass(mod_mgr.get_type_of_guarded_value(), torch.nn.Module)
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class DuplicateGuardTest(torch._dynamo.test_case.TestCase):
    def test_duplicate_guard(self):
        class Foo:
            def __init__(self):
                self.x = 4
                self.bar = 4

        foo = Foo()

        def fn(x):
            if hasattr(foo, "y"):
                x = torch.sin(x)
            if hasattr(foo, "y"):
                x = torch.sin(x)

            if hasattr(foo, "bar"):
                x = torch.cos(x)
            if hasattr(foo, "bar"):
                x = torch.cos(x)
            return x + foo.x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            guard_str = str(guard_wrapper)
            self.assertEqual(guard_str.count("NO_HASATTR"), 1)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class RecursiveDictTagTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._prev = torch._dynamo.config.use_recursive_dict_tags_for_guards
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.use_recursive_dict_tags_for_guards = self._prev


class TagSafetyChecks(RecursiveDictTagTests):
    def setUp(self):
        super().setUp()
        self._prev = torch._dynamo.config.use_recursive_dict_tags_for_guards
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.use_recursive_dict_tags_for_guards = self._prev

    def test_immutable_tag_safe(self):
        class Bar:
            pass

        class Foo:
            def __init__(self):
                self.a = Bar()
                self.b = torch.randn(4)
                self.c = 3
                self.d = (3, 4)
                self.e = (3, Bar())

        foo = Foo()

        def fn(x):
            if foo.a:
                x = torch.sin(x)
            x = x * foo.b + foo.c + foo.d[0] + foo.d[1] + foo.e[0]
            if foo.e[1]:
                x = torch.sin(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import AttrSource, LocalSource

            foo_source = LocalSource("foo")
            foo_mgr = builder.get_guard_manager_from_source(foo_source)
            for accessor in foo_mgr.get_accessors():
                if isinstance(accessor, GetAttrGuardAccessor):
                    self.assertTrue(
                        accessor.get_attr_name() in ("a", "b", "c", "d", "e")
                    )

            # Check types of foo.a
            foo_a_source = AttrSource(foo_source, "a")
            foo_a_mgr = builder.get_guard_manager_from_source(foo_a_source)
            self.assertFalse(foo_a_mgr.is_tag_safe())
            self.assertFalse(foo_a_mgr.is_tag_safe_root())

            # Check types of foo.b
            foo_b_source = AttrSource(foo_source, "b")
            foo_b_mgr = builder.get_guard_manager_from_source(foo_b_source)
            if torch._dynamo.config.skip_tensor_guards_with_matching_dict_tags:
                self.assertTrue(foo_b_mgr.is_tag_safe())
            else:
                self.assertFalse(foo_b_mgr.is_tag_safe())

            self.assertFalse(foo_b_mgr.is_tag_safe_root())

            # Check types of foo.c
            foo_c_source = AttrSource(foo_source, "c")
            foo_c_mgr = builder.get_guard_manager_from_source(foo_c_source)
            self.assertTrue(foo_c_mgr.is_tag_safe())
            self.assertFalse(foo_c_mgr.is_tag_safe_root())

            # Check types of foo.d
            foo_d_source = AttrSource(foo_source, "d")
            foo_d_mgr = builder.get_guard_manager_from_source(foo_d_source)
            self.assertTrue(foo_d_mgr.is_tag_safe())
            self.assertFalse(foo_d_mgr.is_tag_safe_root())

            # Check types of foo.e
            foo_e_source = AttrSource(foo_source, "e")
            foo_e_mgr = builder.get_guard_manager_from_source(foo_e_source)
            self.assertFalse(foo_e_mgr.is_tag_safe())
            self.assertFalse(foo_e_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_dict_tag_safe(self):
        class Foo:
            def __init__(self):
                self.a = 4

        foo = Foo()
        terminal_dict = {
            "a": 1,
        }

        tag_safe_dict = {
            "const": 1,
            "tup": (2, 3),
            "nested_dict": terminal_dict,
        }

        tag_unsafe_dict = {
            "const": 1,
            "foo": foo,
        }

        outer_dict = {
            "safe": tag_safe_dict,
            "unsafe": tag_unsafe_dict,
            "terminal_dict": {"a": 1},
        }

        def fn(x):
            x = x + outer_dict["safe"]["const"]

            x = x + outer_dict["safe"]["tup"][0]
            x = x + outer_dict["safe"]["tup"][1]

            x = x + outer_dict["safe"]["nested_dict"]["a"]

            x = x + outer_dict["unsafe"]["const"]

            x = x + outer_dict["unsafe"]["foo"].a

            if outer_dict["terminal_dict"]:
                x = torch.sin(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import DictGetItemSource, LocalSource

            outer_source = LocalSource("outer_dict")

            # Check tagness of outer dict
            outer_mgr = builder.get_guard_manager_from_source(outer_source)
            self.assertFalse(outer_mgr.is_tag_safe())
            self.assertFalse(outer_mgr.is_tag_safe_root())

            # Check tagness of outer["safe"]
            outer_safe_source = DictGetItemSource(outer_source, "safe")
            outer_safe_mgr = builder.get_guard_manager_from_source(outer_safe_source)
            self.assertTrue(outer_safe_mgr.is_tag_safe())
            self.assertFalse(outer_safe_mgr.is_tag_safe_root())

            # Check tagness of outer["unsafe"]
            outer_unsafe_source = DictGetItemSource(outer_source, "unsafe")
            outer_unsafe_mgr = builder.get_guard_manager_from_source(
                outer_unsafe_source
            )
            self.assertFalse(outer_unsafe_mgr.is_tag_safe())
            self.assertFalse(outer_unsafe_mgr.is_tag_safe_root())

            # Check tagness of outer["terminal_dict"]
            outer_terminal_source = DictGetItemSource(outer_source, "terminal_dict")
            outer_terminal_mgr = builder.get_guard_manager_from_source(
                outer_terminal_source
            )
            self.assertTrue(outer_terminal_mgr.is_tag_safe())
            self.assertFalse(outer_terminal_mgr.is_tag_safe_root())

            # Check tagness of outer["safe"]["nested_dict"]
            outer_safe_nested_source = DictGetItemSource(
                outer_safe_source, "nested_dict"
            )
            outer_safe_nested_mgr = builder.get_guard_manager_from_source(
                outer_safe_nested_source
            )
            self.assertTrue(outer_safe_nested_mgr.is_tag_safe())
            # This should not be marked as a root
            self.assertFalse(outer_safe_nested_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_nn_module_tag_safe(self):
        class Foo(torch.nn.Module):
            c = 2

            def __init__(self):
                super().__init__()
                self.a = 4

            def check(self, x):
                return True

            def forward(self, x):
                inspect.signature(self.check).parameters.items()
                return x + self.a + self.c

        foo = Foo()

        class Env(metaclass=abc.ABCMeta):  # noqa: B024
            pass

        class Baz(torch.nn.Module, Env):
            def __init__(self):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                if "Foo" in str(type(self).__mro__):
                    x = torch.sin(x)
                return self.foo(x)

        baz = Baz()

        def fn(x):
            x = x + baz(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            baz_source = LocalSource("baz")

            # Check tagness of baz
            baz_mgr = builder.get_guard_manager_from_source(baz_source)
            self.assertTrue(baz_mgr.is_tag_safe())
            self.assertTrue(baz_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))

    def test_nn_module_tag_overridden_getattr_safe(self):
        class Baz(torch.nn.Module, metaclass=abc.ABCMeta):
            def __init__(self):
                super().__init__()
                self.norm = 2

            def __getattr__(self, key):
                if key == "a":
                    return 5
                return super().__getattr__(key)

            def forward(self, x):
                return x + self.a + self.norm

        baz = Baz()

        def fn(x):
            x = x + baz(x)
            return x

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def hook(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            baz_source = LocalSource("baz")

            # Check tagness of baz
            baz_mgr = builder.get_guard_manager_from_source(baz_source)
            self.assertTrue(baz_mgr.is_tag_safe())
            self.assertTrue(baz_mgr.is_tag_safe_root())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(hook):
            opt_fn(torch.randn(4, 4))


class RecursiveDictGuardTests(RecursiveDictTagTests):
    def test_disabling(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 4

            def forward(self, x):
                return x + self.a

        mod = Mod()
        mod_to_fail = Mod()

        def fn(x):
            return mod(x)

        x = torch.randn(4, 4)

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        def basic_hook_test(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard pass but dict matching fail, this should add new cached entry
            self.assertTrue(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard fail, this should disable dict tag optimization as well
            mod_to_fail.a = 5
            self.assertFalse(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(basic_hook_test):
            opt_fn(x)

        # Test that dict tag matching failure leads to disable of dict tag optimization
        torch.compiler.reset()
        mod = Mod()
        mod_to_fail = Mod()

        def disable_on_dict_tag_match_failure(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Change the mod attr to cause dict tag matching to fail, this still
            # get the guard pass. This should disable the dict tag optimization.
            mod.a = 5
            mod.a = 4
            self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with install_guard_manager_testing_hook(disable_on_dict_tag_match_failure):
            opt_fn(x)

        # Test that max size limit breach disables the dict tag optimization
        torch.compiler.reset()
        mod = Mod()
        mod_to_fail = Mod()

        def max_size_test(guard_wrapper, f_locals, builder):
            from torch._dynamo.source import LocalSource

            mod_source = LocalSource("mod")

            # Check tagness of mod
            mod_mgr = builder.get_guard_manager_from_source(mod_source)
            self.assertTrue(mod_mgr.is_tag_safe())
            self.assertTrue(mod_mgr.is_tag_safe_root())
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            for _ in range(10):
                self.assertTrue(guard_wrapper.check({"mod": mod, "x": x}))
            self.assertFalse(mod_mgr.is_recursive_dict_tag_matching_disabled())

            # Let the guard pass but dict matching fail, since cache size is set
            # to 1, this would cause dict tag optimization to be disabled.
            self.assertTrue(guard_wrapper.check({"mod": mod_to_fail, "x": x}))
            self.assertTrue(mod_mgr.is_recursive_dict_tag_matching_disabled())

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with torch._dynamo.config.patch(
            max_saved_pointers_for_recursive_dict_tags_check=1
        ):
            with install_guard_manager_testing_hook(max_size_test):
                opt_fn(x)


class SourceCloneTests(torch._dynamo.test_case.TestCase):
    def test_clone_identity_transform(self):
        """Identity transform should produce a source with the same name."""
        from torch._dynamo.source import AttrSource, GetItemSource, LocalSource

        local = LocalSource("x")
        attr = AttrSource(local, "weight")
        item = GetItemSource(attr, 0)

        for source in [local, attr, item]:
            cloned = source.clone(lambda x: x)
            self.assertEqual(cloned.name, source.name)

    def test_clone_no_transform(self):
        from torch._dynamo.source import AttrSource, LocalSource

        local = LocalSource("x")
        attr = AttrSource(local, "weight")

        self.assertIs(local.clone(), local)
        cloned_attr = attr.clone()
        self.assertEqual(cloned_attr.name, attr.name)

    def test_clone_parameterized_deep_chain(self):
        """Replace leaf sources deep in a chain via a find->replace dictionary."""
        from torch._dynamo.source import (
            AttrSource,
            ConstDictKeySource,
            DictGetItemSource,
            GetItemSource,
            LocalSource,
        )

        # Build: L['x'].layers[0].weight
        local = LocalSource("x")
        attr1 = AttrSource(local, "layers")
        item = GetItemSource(attr1, 0)
        attr2 = AttrSource(item, "weight")

        replacements = {local: LocalSource("y")}

        def transform(s):
            return replacements.get(s, s)

        cloned = attr2.clone(transform)
        self.assertEqual(cloned.name, "L['y'].layers[0].weight")

        # Build: L['d'][list(dict.keys(L['d']))[0]]  (DictGetItemSource with Source key)
        local_d = LocalSource("d")
        key = ConstDictKeySource(local_d, 0)
        dict_src = DictGetItemSource(local_d, key)

        replacements = {local_d: LocalSource("other")}
        cloned = dict_src.clone(transform)
        self.assertEqual(cloned.name, "L['other'][list(dict.keys(L['other']))[0]]")

    def test_clone_dict_get_item_source_with_constant_key(self):
        from torch._dynamo.source import DictGetItemSource, LocalSource

        local = LocalSource("d")
        source = DictGetItemSource(local, "key")
        cloned = source.clone()
        self.assertEqual(cloned.name, source.name)

        replacement = LocalSource("other_d")

        def replace_local(s):
            if isinstance(s, LocalSource) and s.local_name == "d":
                return replacement
            return s

        cloned = source.clone(replace_local)
        self.assertEqual(cloned.name, "L['other_d']['key']")

    def test_clone_dict_get_item_source_with_source_key(self):
        from torch._dynamo.source import (
            ConstDictKeySource,
            DictGetItemSource,
            LocalSource,
        )

        local = LocalSource("d")
        key_source = ConstDictKeySource(local, 0)
        source = DictGetItemSource(local, key_source)

        cloned = source.clone(lambda x: x)
        self.assertEqual(cloned.name, source.name)

    def test_clone_dict_subclass_get_item_source(self):
        from torch._dynamo.source import DictSubclassGetItemSource, LocalSource

        local = LocalSource("d")
        source = DictSubclassGetItemSource(local, "key")

        cloned = source.clone()
        self.assertEqual(cloned.name, source.name)

        cloned = source.clone(lambda x: x)
        self.assertEqual(cloned.name, source.name)

    def test_clone_get_item_source(self):
        from torch._dynamo.source import GetItemSource, LocalSource

        local = LocalSource("lst")
        source = GetItemSource(local, 3)

        cloned = source.clone()
        self.assertEqual(cloned.name, source.name)

        cloned = source.clone(lambda x: x)
        self.assertEqual(cloned.name, source.name)


class GuardCheckSpecTests(torch._dynamo.test_case.TestCase):
    """Tests for the GuardCheckSpec get_metadata_fn/eval_fn handlers on GuardBuilder."""

    def _get_handler(self, name):
        from torch._dynamo.guards import GUARD_VALUE_DISPATCH

        return GUARD_VALUE_DISPATCH[name]

    def _make_guard(self, create_fn):
        from torch._dynamo.source import LocalSource

        return LocalSource("x").make_guard(create_fn)

    def test_type_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.TYPE_MATCH)
        handler = self._get_handler("TYPE_MATCH")

        expected = handler.get_metadata_fn(guard, 42)
        self.assertIs(expected, int)
        self.assertTrue(handler.eval_fn(100, expected))
        self.assertFalse(handler.eval_fn("hello", expected))

    def test_constant_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.CONSTANT_MATCH)
        handler = self._get_handler("CONSTANT_MATCH")

        expected = handler.get_metadata_fn(guard, 42)
        self.assertEqual(expected, 42)
        self.assertTrue(handler.eval_fn(42, expected))
        self.assertFalse(handler.eval_fn(99, expected))

    def test_equals_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.EQUALS_MATCH)
        handler = self._get_handler("EQUALS_MATCH")

        expected = handler.get_metadata_fn(guard, [1, 2, 3])
        self.assertTrue(handler.eval_fn([1, 2, 3], expected))
        self.assertFalse(handler.eval_fn([1, 2], expected))

    def test_id_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.ID_MATCH)
        handler = self._get_handler("ID_MATCH")

        obj = object()
        expected = handler.get_metadata_fn(guard, obj)
        self.assertIs(expected, obj)
        self.assertTrue(handler.eval_fn(obj, expected))
        self.assertFalse(handler.eval_fn(object(), expected))

    def test_class_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.CLASS_MATCH)
        handler = self._get_handler("CLASS_MATCH")

        expected = handler.get_metadata_fn(guard, dict)
        self.assertIs(expected, dict)
        self.assertTrue(handler.eval_fn(dict, expected))
        self.assertFalse(handler.eval_fn(list, expected))

    def test_module_match(self):
        import types

        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.MODULE_MATCH)
        handler = self._get_handler("MODULE_MATCH")

        mod = types.ModuleType("test_mod")
        expected = handler.get_metadata_fn(guard, mod)
        self.assertIs(expected, mod)
        self.assertTrue(handler.eval_fn(mod, expected))

        other = types.ModuleType("other_mod")
        self.assertFalse(handler.eval_fn(other, expected))

    def test_builtin_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.BUILTIN_MATCH)
        handler = self._get_handler("BUILTIN_MATCH")

        expected = handler.get_metadata_fn(guard, len)
        self.assertIs(expected, len)
        self.assertTrue(handler.eval_fn(len, expected))
        self.assertFalse(handler.eval_fn(print, expected))

    def test_hasattr_present(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(functools.partial(GuardBuilder.HASATTR, attr="weight"))
        handler = self._get_handler("HASATTR")

        class Obj:
            weight = 1.0

        expected = handler.get_metadata_fn(guard, Obj())
        self.assertEqual(expected, ("weight", True))
        self.assertTrue(handler.eval_fn(Obj(), expected))

        class Empty:
            pass

        self.assertFalse(handler.eval_fn(Empty(), expected))

    def test_hasattr_absent(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(functools.partial(GuardBuilder.HASATTR, attr="bias"))
        handler = self._get_handler("HASATTR")

        class Obj:
            weight = 1.0

        obj = Obj()
        expected = handler.get_metadata_fn(guard, obj)
        self.assertEqual(expected, ("bias", False))
        self.assertTrue(handler.eval_fn(obj, expected))
        # Adding the attr should fail the "not hasattr" guard
        obj.bias = 0.0  # type: ignore[attr-defined]
        self.assertFalse(handler.eval_fn(obj, expected))

    def test_sequence_length(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.SEQUENCE_LENGTH)
        handler = self._get_handler("SEQUENCE_LENGTH")

        expected = handler.get_metadata_fn(guard, [1, 2, 3])
        self.assertEqual(expected, 3)
        self.assertTrue(handler.eval_fn([4, 5, 6], expected))
        self.assertTrue(handler.eval_fn((7, 8, 9), expected))
        self.assertFalse(handler.eval_fn([1], expected))

    def test_dict_contains(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(functools.partial(GuardBuilder.DICT_CONTAINS, key="a"))
        handler = self._get_handler("DICT_CONTAINS")

        d = {"a": 1, "b": 2}
        expected = handler.get_metadata_fn(guard, d)
        self.assertEqual(expected, "a")
        self.assertTrue(handler.eval_fn({"a": 99}, expected))
        self.assertFalse(handler.eval_fn({"b": 1}, expected))

    def test_dict_not_contains(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(
            functools.partial(GuardBuilder.DICT_NOT_CONTAINS, key="a")
        )
        handler = self._get_handler("DICT_NOT_CONTAINS")

        d = {"b": 2}
        expected = handler.get_metadata_fn(guard, d)
        self.assertEqual(expected, "a")
        self.assertTrue(handler.eval_fn({"b": 1}, expected))
        self.assertFalse(handler.eval_fn({"a": 1}, expected))

    def test_not_present_in_generic_dict(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(
            functools.partial(GuardBuilder.NOT_PRESENT_IN_GENERIC_DICT, attr="hidden")
        )
        handler = self._get_handler("NOT_PRESENT_IN_GENERIC_DICT")

        class Obj:
            pass

        obj = Obj()
        expected = handler.get_metadata_fn(guard, obj)
        self.assertEqual(expected, "hidden")
        self.assertTrue(handler.eval_fn(obj, expected))
        obj.hidden = 1  # type: ignore[attr-defined]
        self.assertFalse(handler.eval_fn(obj, expected))

    def test_closure_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.CLOSURE_MATCH)
        handler = self._get_handler("CLOSURE_MATCH")

        def fn():
            return 42

        expected = handler.get_metadata_fn(guard, fn)
        self.assertIs(expected, fn.__code__)
        self.assertTrue(handler.eval_fn(fn, expected))

        def other_fn():
            return 99

        self.assertFalse(handler.eval_fn(other_fn, expected))

    def test_tensor_match(self):
        from torch._dynamo.guards import extract_tensor_metadata, GuardBuilder

        guard = self._make_guard(GuardBuilder.TENSOR_MATCH)
        handler = self._get_handler("TENSOR_MATCH")

        t = torch.randn(3, 4)
        expected = handler.get_metadata_fn(guard, t)
        self.assertEqual(expected, extract_tensor_metadata(t))
        self.assertTrue(handler.eval_fn(t, expected))
        self.assertTrue(handler.eval_fn(torch.randn(3, 4), expected))
        self.assertFalse(handler.eval_fn(torch.randn(5, 4), expected))
        self.assertFalse(
            handler.eval_fn(torch.randn(3, 4, dtype=torch.float64), expected)
        )
        self.assertFalse(handler.eval_fn(42, expected))

    def test_empty_nn_module_hooks_dict(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.EMPTY_NN_MODULE_HOOKS_DICT)
        handler = self._get_handler("EMPTY_NN_MODULE_HOOKS_DICT")

        expected = handler.get_metadata_fn(guard, {})
        self.assertIsNone(expected)
        self.assertTrue(handler.eval_fn({}, expected))
        self.assertFalse(handler.eval_fn({"hook": lambda: None}, expected))

    def test_bool_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.BOOL_MATCH)
        handler = self._get_handler("BOOL_MATCH")

        expected = handler.get_metadata_fn(guard, True)
        self.assertTrue(handler.eval_fn(True, expected))
        self.assertFalse(handler.eval_fn(False, expected))

        expected_false = handler.get_metadata_fn(guard, False)
        self.assertTrue(handler.eval_fn(False, expected_false))
        self.assertFalse(handler.eval_fn(True, expected_false))

    def test_none_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.NONE_MATCH)
        handler = self._get_handler("NONE_MATCH")

        expected = handler.get_metadata_fn(guard, None)
        self.assertIsNone(expected)
        self.assertTrue(handler.eval_fn(None, expected))
        self.assertFalse(handler.eval_fn(0, expected))
        self.assertFalse(handler.eval_fn("", expected))

    def test_function_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.FUNCTION_MATCH)
        handler = self._get_handler("FUNCTION_MATCH")

        def my_fn():
            return 42

        expected = handler.get_metadata_fn(guard, my_fn)
        self.assertIs(expected, my_fn)
        self.assertTrue(handler.eval_fn(my_fn, expected))
        self.assertFalse(handler.eval_fn(lambda: 42, expected))
        self.assertFalse(handler.eval_fn(torch.add, expected))

    def test_weakref_alive(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.WEAKREF_ALIVE)
        handler = self._get_handler("WEAKREF_ALIVE")

        # The guard spec receives the dereferenced value (ref()), not the
        # weakref itself. A live referent resolves to the object; a dead
        # one resolves to None.
        class C:
            pass

        obj = C()
        ref = weakref.ref(obj)
        expected = handler.get_metadata_fn(guard, ref())
        self.assertIsNone(expected)
        self.assertTrue(handler.eval_fn(ref(), expected))
        # Delete the referent — weakref() now returns None
        del obj
        self.assertFalse(handler.eval_fn(ref(), expected))

    def test_set_contains(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(functools.partial(GuardBuilder.SET_CONTAINS, key="a"))
        handler = self._get_handler("SET_CONTAINS")

        s = {"a", "b", "c"}
        expected = handler.get_metadata_fn(guard, s)
        self.assertEqual(expected, "a")
        self.assertTrue(handler.eval_fn({"a", "x"}, expected))
        self.assertFalse(handler.eval_fn({"b", "c"}, expected))

    def test_set_not_contains(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(
            functools.partial(GuardBuilder.SET_NOT_CONTAINS, key="z")
        )
        handler = self._get_handler("SET_NOT_CONTAINS")

        s = {"a", "b"}
        expected = handler.get_metadata_fn(guard, s)
        self.assertEqual(expected, "z")
        self.assertTrue(handler.eval_fn({"a", "b"}, expected))
        self.assertFalse(handler.eval_fn({"a", "z"}, expected))

    def test_not_none_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.NOT_NONE_MATCH)
        handler = self._get_handler("NOT_NONE_MATCH")

        expected = handler.get_metadata_fn(guard, torch.randn(2))
        self.assertIsNone(expected)
        self.assertTrue(handler.eval_fn(torch.randn(3), expected))
        self.assertTrue(handler.eval_fn(42, expected))
        self.assertFalse(handler.eval_fn(None, expected))

    def test_dispatch_key_set_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.DISPATCH_KEY_SET_MATCH)
        handler = self._get_handler("DISPATCH_KEY_SET_MATCH")

        dks = torch._C._dispatch_keys(torch.randn(3))
        expected = handler.get_metadata_fn(guard, dks)
        self.assertEqual(expected, dks.raw_repr())
        self.assertTrue(handler.eval_fn(dks, expected))
        # Different tensor with same dispatch keys should match
        dks2 = torch._C._dispatch_keys(torch.randn(5))
        self.assertTrue(handler.eval_fn(dks2, expected))

    def test_tuple_iterator_len(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.TUPLE_ITERATOR_LEN)
        handler = self._get_handler("TUPLE_ITERATOR_LEN")

        it = iter((1, 2, 3))
        expected = handler.get_metadata_fn(guard, it)
        self.assertEqual(expected, 3)
        self.assertTrue(handler.eval_fn(iter((4, 5, 6)), expected))
        self.assertFalse(handler.eval_fn(iter((1,)), expected))

    def test_range_iterator_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.RANGE_ITERATOR_MATCH)
        handler = self._get_handler("RANGE_ITERATOR_MATCH")

        it = iter(range(1, 10, 2))
        expected = handler.get_metadata_fn(guard, it)
        self.assertTrue(handler.eval_fn(iter(range(1, 10, 2)), expected))
        self.assertFalse(handler.eval_fn(iter(range(0, 10, 2)), expected))
        self.assertFalse(handler.eval_fn(iter(range(1, 10, 3)), expected))

    def test_nn_module(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.NN_MODULE)
        handler = self._get_handler("NN_MODULE")

        mod = torch.nn.Linear(3, 4)
        expected = handler.get_metadata_fn(guard, mod)
        self.assertIs(expected, mod)
        self.assertTrue(handler.eval_fn(mod, expected))
        self.assertFalse(handler.eval_fn(torch.nn.Linear(3, 4), expected))

    def test_mapping_keys_check(self):
        import types

        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.MAPPING_KEYS_CHECK)
        handler = self._get_handler("MAPPING_KEYS_CHECK")

        mp = types.MappingProxyType({"a": 1, "b": 2})
        expected = handler.get_metadata_fn(guard, mp)
        self.assertEqual(expected, ["a", "b"])
        self.assertTrue(
            handler.eval_fn(types.MappingProxyType({"a": 10, "b": 20}), expected)
        )
        self.assertFalse(handler.eval_fn(types.MappingProxyType({"x": 1}), expected))

    @unittest.skipIf(
        sys.platform != "linux",
        "Only support mem leak checking on Linux.",
    )
    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN,
        "RSS-based leak detection is unreliable under sanitizers.",
    )
    def test_clone_manager_memory_leak(self):
        def get_mem():
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

        def clone_filter(mgr):
            return True

        root = RootGuardManager()

        # Warmup
        for _ in range(100):
            root.clone_manager(clone_filter)
        gc.collect()

        # Iterate to make the leak larger
        initial_mem = get_mem()
        for _ in range(10000):
            root.clone_manager(clone_filter)
        gc.collect()
        final_mem = get_mem()
        delta = final_mem - initial_mem

        # Only fail if the leak is larger than 1MB.
        self.assertLessEqual(
            delta,
            1 * 1024 * 1024,
            lambda msg: f"{msg}\nMemory leaked: {delta / 1024 / 1024:.2f} MB",
        )

    def test_dict_keys_match(self):
        from torch._dynamo.guards import GuardBuilder

        guard = self._make_guard(GuardBuilder.DICT_KEYS_MATCH)
        handler = self._get_handler("DICT_KEYS_MATCH")

        d = {"a": 1, "b": 2, "c": 3}
        expected = handler.get_metadata_fn(guard, d)
        self.assertEqual(expected, ["a", "b", "c"])
        self.assertTrue(handler.eval_fn({"a": 10, "b": 20, "c": 30}, expected))
        self.assertFalse(handler.eval_fn({"a": 1, "b": 2}, expected))
        self.assertFalse(handler.eval_fn({"x": 1, "y": 2, "z": 3}, expected))


class GuardActualPartialFastPathTests(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _run_guard_lookup_memo_script(script):
        prefix = (
            "import torch\n"
            "torch._dynamo.config.enable_guard_lookup_memo = True\n"
            "from torch._dynamo.eval_frame import _debug_get_cache_entry_list\n"
            "def _only_cache_entry(code):\n"
            "    entries = _debug_get_cache_entry_list(code)\n"
            "    assert len(entries) == 1, len(entries)\n"
            "    return entries[0]\n"
            "def _guard_tree_shapes(entry):\n"
            "    leaves = []\n"
            "    accessors = []\n"
            "    pending = [entry.guard_manager.root]\n"
            "    while pending:\n"
            "        manager = pending.pop()\n"
            "        source = manager.get_source()\n"
            "        leaves.extend(\n"
            "            (source, type(guard).__name__)\n"
            "            for guard in manager.get_leaf_guards()\n"
            "        )\n"
            "        accessors.extend(\n"
            "            (source, type(accessor).__name__)\n"
            "            for accessor in manager.get_accessors()\n"
            "        )\n"
            "        pending.extend(manager.get_child_managers())\n"
            "    return leaves, accessors\n"
            "from torch._dynamo.testing import CompileCounter\n"
            "def _warm_model(model, enabled=True):\n"
            "    counter = CompileCounter()\n"
            "    compiled = torch.compile(model, backend=counter, fullgraph=True)\n"
            "    x = torch.zeros(2)\n"
            "    expected = model(x)\n"
            "    for _ in range(8):\n"
            "        torch.testing.assert_close(compiled(x), expected)\n"
            "    assert counter.frame_count == 1, counter.frame_count\n"
            "    entry = _only_cache_entry(type(model).forward.__code__)\n"
            "    assert entry._debug_fast_guard_enabled is enabled\n"
            "    return compiled, counter, entry, x\n"
        )
        subprocess.run(
            [sys.executable, "-c", prefix + textwrap.dedent(script)],
            cwd=os.getcwd(),
            check=True,
        )

    def test_actual_partial_preserves_module_and_residual_guards(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            global_bias = torch.tensor(3.0)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = 2.0
                    self.offsets = [1.0]

                def forward(self, x):
                    return x * self.scale + self.offsets[0] + global_bias

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )

            x = torch.ones(4)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count

            cache_entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(cache_entries) == 1, len(cache_entries)
            original_entry = cache_entries[0]
            assert original_entry._debug_fast_guard_enabled

            model.scale = 4.0
            torch.testing.assert_close(compiled(x), model(x))
            assert original_entry._debug_fast_guard_enabled

            model.offsets[0] = 5.0
            torch.testing.assert_close(compiled(x), model(x))
            assert original_entry._debug_fast_guard_enabled

            global_bias = torch.tensor(7.0)
            torch.testing.assert_close(compiled(x), model(x))
            assert original_entry._debug_fast_guard_enabled

            model.scale = 6.0
            x = torch.ones(9)
            torch.testing.assert_close(compiled(x), model(x))
            assert original_entry._debug_fast_guard_enabled

            def alias_sensitive(a, b):
                return a + b if a is b else a - b

            compiled_alias = torch.compile(
                alias_sensitive, backend="eager", fullgraph=True
            )
            a = torch.ones(4)
            b = torch.full((4,), 2.0)
            torch.testing.assert_close(compiled_alias(a, a), alias_sensitive(a, a))
            torch.testing.assert_close(compiled_alias(a, b), alias_sensitive(a, b))
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_fail_closed_leaf_and_container_shapes(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class ExactContainerModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = [torch.ones(2)]
                    self.options = (torch.full((2,), 2.0),)

                def forward(self, x):
                    return (
                        x
                        + self.values[0]
                        + len(self.values)
                        + self.options[0]
                        + len(self.options)
                    )

            exact_model = ExactContainerModel()
            exact_counter = CompileCounter()
            exact_compiled = torch.compile(
                exact_model, backend=exact_counter, fullgraph=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(exact_compiled(x), exact_model(x))
            assert exact_counter.frame_count == 1, exact_counter.frame_count
            exact_entry = _only_cache_entry(ExactContainerModel.forward.__code__)
            _, exact_accessors = _guard_tree_shapes(exact_entry)
            assert any(
                kind == "ListGetItemGuardAccessor"
                and source.startswith("L['self']")
                for source, kind in exact_accessors
            ), exact_accessors
            assert any(
                kind == "TupleGetItemGuardAccessor"
                and source.startswith("L['self']")
                for source, kind in exact_accessors
            ), exact_accessors
            assert exact_entry._debug_fast_guard_enabled

            class Holder:
                pass

            class NoHasattrModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = Holder()

                def forward(self, x):
                    if hasattr(self.holder, "scale"):
                        return x + self.holder.scale
                    return x + 1

            no_hasattr_model = NoHasattrModel()
            no_hasattr_counter = CompileCounter()
            no_hasattr_compiled = torch.compile(
                no_hasattr_model, backend=no_hasattr_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(
                    no_hasattr_compiled(x), no_hasattr_model(x)
                )
            assert no_hasattr_counter.frame_count == 1, no_hasattr_counter.frame_count
            no_hasattr_entry = _only_cache_entry(NoHasattrModel.forward.__code__)
            leaves, _ = _guard_tree_shapes(no_hasattr_entry)
            assert any(
                kind == "NO_HASATTR" and source.startswith("L['self']")
                for source, kind in leaves
            ), leaves
            assert not no_hasattr_entry._debug_fast_guard_enabled

            no_hasattr_model.holder.scale = torch.full((2,), 4.0)
            torch.testing.assert_close(
                no_hasattr_compiled(x), no_hasattr_model(x)
            )
            assert no_hasattr_counter.frame_count == 2, no_hasattr_counter.frame_count

            class MutableEqualsModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = [1, 2]

                def forward(self, x):
                    if self.mode == [1, 2]:
                        return x + 1
                    return x - 1

            mutable_model = MutableEqualsModel()
            mutable_counter = CompileCounter()
            mutable_compiled = torch.compile(
                mutable_model, backend=mutable_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(mutable_compiled(x), mutable_model(x))
            assert mutable_counter.frame_count == 1, mutable_counter.frame_count
            mutable_entry = _only_cache_entry(MutableEqualsModel.forward.__code__)
            leaves, _ = _guard_tree_shapes(mutable_entry)
            assert any(
                kind == "EQUALS_MATCH" and source.startswith("L['self']")
                for source, kind in leaves
            ), leaves
            assert not mutable_entry._debug_fast_guard_enabled

            mutable_model.mode.append(3)
            torch.testing.assert_close(mutable_compiled(x), mutable_model(x))
            assert mutable_counter.frame_count == 2, mutable_counter.frame_count

            class FancyList(list):
                pass

            class NonExactListModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = FancyList([1.0])

                def forward(self, x):
                    return x + self.values[0] + len(self.values)

            nonexact_model = NonExactListModel()
            nonexact_counter = CompileCounter()
            nonexact_compiled = torch.compile(
                nonexact_model, backend=nonexact_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(nonexact_compiled(x), nonexact_model(x))
            assert nonexact_counter.frame_count == 1, nonexact_counter.frame_count
            nonexact_entry = _only_cache_entry(NonExactListModel.forward.__code__)
            leaves, accessors = _guard_tree_shapes(nonexact_entry)
            assert any(
                kind == "LENGTH_CHECK" and source.startswith("L['self']")
                for source, kind in leaves
            ), leaves
            assert any(
                kind == "ListGetItemGuardAccessor"
                and source.startswith("L['self']")
                for source, kind in accessors
            ), accessors
            assert not nonexact_entry._debug_fast_guard_enabled

            nonexact_model.values.append(2.0)
            torch.testing.assert_close(nonexact_compiled(x), nonexact_model(x))
            assert nonexact_counter.frame_count == 2, nonexact_counter.frame_count
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_instance_attr_binding_and_type_refresh(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class Holder:
                def __init__(self):
                    self.scale = torch.ones(2)

            class OtherHolder:
                pass

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = Holder()

                def forward(self, x):
                    return self.holder.scale + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _only_cache_entry(Model.forward.__code__)
            assert original_entry._debug_fast_guard_enabled

            Holder._fastguard_unrelated_type_change = None
            try:
                torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 1, counter.frame_count
                assert original_entry._debug_fast_guard_enabled
            finally:
                del Holder._fastguard_unrelated_type_change

            original_scale = model.holder.scale
            model.holder.scale = torch.full((2,), 3.0)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            model.holder.scale = original_scale
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            Holder.scale = property(lambda self: torch.full((2,), 5.0))
            try:
                torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 3, counter.frame_count
                assert original_entry._debug_fast_guard_enabled
            finally:
                del Holder.scale

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 3, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            model.holder.__class__ = OtherHolder
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 4, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            model.holder.__class__ = Holder
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 4, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_default_getattribute_change(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class InitiallyCustomHolder:
                def __init__(self):
                    self.scale = torch.ones(2)

                def __getattribute__(self, name):
                    return object.__getattribute__(self, name)

            class InitiallyCustomModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = InitiallyCustomHolder()

                def forward(self, x):
                    return self.holder.scale + x

            initial_model = InitiallyCustomModel()
            initial_counter = CompileCounter()
            initial_compiled = torch.compile(
                initial_model, backend=initial_counter, fullgraph=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(initial_compiled(x), initial_model(x))
            assert initial_counter.frame_count == 1, initial_counter.frame_count
            initial_entry = _only_cache_entry(
                InitiallyCustomModel.forward.__code__
            )
            assert not initial_entry._debug_fast_guard_enabled

            class InitiallyGetattrHolder:
                def __init__(self):
                    self.scale = torch.ones(2)

                def __getattr__(self, name):
                    raise AttributeError(name)

            class InitiallyGetattrModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = InitiallyGetattrHolder()

                def forward(self, x):
                    return self.holder.scale + x

            getattr_model = InitiallyGetattrModel()
            getattr_counter = CompileCounter()
            getattr_compiled = torch.compile(
                getattr_model, backend=getattr_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(
                    getattr_compiled(x), getattr_model(x)
                )
            assert getattr_counter.frame_count == 1, getattr_counter.frame_count
            getattr_entry = _only_cache_entry(
                InitiallyGetattrModel.forward.__code__
            )
            assert not getattr_entry._debug_fast_guard_enabled

            class Holder:
                def __init__(self):
                    self.scale = torch.ones(2)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = Holder()

                def forward(self, x):
                    return self.holder.scale + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _only_cache_entry(Model.forward.__code__)
            assert original_entry._debug_fast_guard_enabled

            replacement_scale = torch.full((2,), 5.0)

            def replacement_getattribute(self, name):
                if name == "scale":
                    return replacement_scale
                return object.__getattribute__(self, name)

            Holder.__getattribute__ = replacement_getattribute
            try:
                torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 2, counter.frame_count
                assert original_entry._debug_fast_guard_enabled
            finally:
                del Holder.__getattribute__

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_generic_dict_binding_proof(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class Holder:
                def __init__(self):
                    self.scale = torch.ones(2)

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holder = Holder()

                def forward(self, x):
                    return self.holder.__dict__["scale"] + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _only_cache_entry(Model.forward.__code__)
            _, accessors = _guard_tree_shapes(original_entry)
            assert any(
                kind == "GetGenericDictGuardAccessor"
                and source.startswith("L['self']")
                for source, kind in accessors
            ), accessors
            assert original_entry._debug_fast_guard_enabled

            original_dict = model.holder.__dict__
            model.holder.__dict__ = {"scale": torch.full((2,), 4.0)}
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            model.holder.__dict__ = original_dict
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_plan_is_per_cache_entry(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = 1

                def forward(self, x):
                    return x + self.mode + GLOBAL_DICT["used"]

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=False
            )
            inputs = (torch.zeros(2), torch.zeros(3))
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                for x in inputs:
                    torch.testing.assert_close(
                        compiled(x), torch.full_like(x, 2.0)
                    )
            assert counter.frame_count == 2, counter.frame_count
            original_entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(original_entries) == 2, len(original_entries)
            assert all(
                entry._debug_fast_guard_enabled for entry in original_entries
            ), [entry._debug_fast_guard_enabled for entry in original_entries]

            model.mode = 5
            for i, x in enumerate(inputs):
                GLOBAL_DICT["noise"] = [100 + i]
                torch.testing.assert_close(
                    compiled(x), torch.full_like(x, 6.0)
                )
            assert counter.frame_count == 4, counter.frame_count
            assert all(
                entry._debug_fast_guard_enabled for entry in original_entries
            ), [entry._debug_fast_guard_enabled for entry in original_entries]

            model.mode = 1
            for i, x in enumerate(inputs):
                GLOBAL_DICT["noise"] = [200 + i]
                torch.testing.assert_close(
                    compiled(x), torch.full_like(x, 2.0)
                )
            assert counter.frame_count == 4, counter.frame_count
            assert all(
                entry._debug_fast_guard_enabled for entry in original_entries
            ), [entry._debug_fast_guard_enabled for entry in original_entries]
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_cross_slice_alias_relations(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("value", torch.ones(2))

                def forward(self, x):
                    return self.value + x + GLOBAL_DICT["used"]

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            same = model.value
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(same), torch.full((2,), 3.0))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(original_entry) == 1, len(original_entry)
            assert original_entry[0]._debug_fast_guard_enabled

            GLOBAL_DICT["noise"] = [100]
            different = torch.zeros_like(same)
            torch.testing.assert_close(compiled(different), torch.full((2,), 2.0))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            GLOBAL_DICT["noise"] = [200]
            torch.testing.assert_close(compiled(same), torch.full((2,), 3.0))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            class DistinctModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("value", torch.ones(2))

                def forward(self, x):
                    return self.value + x + GLOBAL_DICT["used"]

            distinct_model = DistinctModel()
            distinct_counter = CompileCounter()
            distinct = torch.compile(
                distinct_model, backend=distinct_counter, fullgraph=True
            )
            for i in range(8):
                GLOBAL_DICT["noise"] = [i + 300]
                current = torch.full((2,), float(i + 2))
                torch.testing.assert_close(
                    distinct(current), distinct_model.value + current + 1
                )
            assert distinct_counter.frame_count == 1, distinct_counter.frame_count
            distinct_entry = _debug_get_cache_entry_list(
                DistinctModel.forward.__code__
            )
            assert len(distinct_entry) == 1, len(distinct_entry)
            assert distinct_entry[0]._debug_fast_guard_enabled

            GLOBAL_DICT["noise"] = [400]
            torch.testing.assert_close(
                distinct(distinct_model.value), torch.full((2,), 3.0)
            )
            assert distinct_counter.frame_count == 2, distinct_counter.frame_count
            assert distinct_entry[0]._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_all_residual_alias_relations(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            torch._dynamo.config.use_lamba_guard_for_object_aliasing = False

            def relation_source_groups(entry):
                groups = {}
                pending = [entry.guard_manager.root]
                while pending:
                    manager = pending.pop()
                    source = manager.get_source()
                    for guard in manager.get_leaf_guards():
                        if type(guard).__name__ in {
                            "OBJECT_ALIASING",
                            "NO_TENSOR_ALIASING",
                        }:
                            # Keep the pybind guard wrapper alive as the key.  The
                            # same C++ relation guard is exposed as the same wrapper,
                            # while retaining it prevents Python id reuse.
                            groups.setdefault(guard, []).append(source)
                    pending.extend(manager.get_child_managers())
                return list(groups.values())

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bias = 1.0

                def forward(self, x, y):
                    if x is y:
                        return x + y + self.bias
                    return x - y + self.bias

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            same = torch.ones(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(same, same), model(same, same))
            assert counter.frame_count == 1, counter.frame_count
            original_entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(original_entries) == 1, len(original_entries)
            original_entry = original_entries[0]
            assert original_entry._debug_fast_guard_enabled

            groups = relation_source_groups(original_entry)
            assert any(
                len(group) >= 2
                and all(not source.startswith("L['self']") for source in group)
                for group in groups
            ), groups

            different = torch.zeros_like(same)
            torch.testing.assert_close(
                compiled(same, different), model(same, different)
            )
            assert counter.frame_count == 2, counter.frame_count
            assert len(_debug_get_cache_entry_list(Model.forward.__code__)) == 2
            assert original_entry._debug_fast_guard_enabled

            torch.testing.assert_close(compiled(same, same), model(same, same))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_all_self_alias_relations(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            torch._dynamo.config.use_lamba_guard_for_object_aliasing = False

            def relation_source_groups(entry):
                groups = {}
                pending = [entry.guard_manager.root]
                while pending:
                    manager = pending.pop()
                    source = manager.get_source()
                    for guard in manager.get_leaf_guards():
                        if type(guard).__name__ in {
                            "OBJECT_ALIASING",
                            "NO_TENSOR_ALIASING",
                        }:
                            # Keep the pybind guard wrapper alive as the key.  The
                            # same C++ relation guard is exposed as the same wrapper,
                            # while retaining it prevents Python id reuse.
                            groups.setdefault(guard, []).append(source)
                    pending.extend(manager.get_child_managers())
                return list(groups.values())

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    shared = torch.ones(2)
                    self.register_buffer("left", shared)
                    self.register_buffer("right", shared)

                def forward(self, x):
                    if self.left is self.right:
                        return x + self.left + self.right
                    return x + self.left - self.right

            model = Model()
            original_right = model.right
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(original_entries) == 1, len(original_entries)
            original_entry = original_entries[0]
            assert original_entry._debug_fast_guard_enabled

            groups = relation_source_groups(original_entry)
            assert any(
                len(group) >= 2
                and all(source.startswith("L['self']") for source in group)
                for group in groups
            ), groups

            model.right = torch.zeros(2)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert len(_debug_get_cache_entry_list(Model.forward.__code__)) == 2
            assert original_entry._debug_fast_guard_enabled

            model.right = original_right
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_dict_tagged_alias_relations(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            def run_case(recursive_tags):
                torch._dynamo.reset()
                torch._dynamo.config.skip_tensor_guards_with_matching_dict_tags = True
                torch._dynamo.config.use_recursive_dict_tags_for_guards = recursive_tags
                torch._dynamo.config.use_lamba_guard_for_object_aliasing = False
                torch._dynamo.config.skip_no_tensor_aliasing_guards_on_parameters = False

                alias_dict = {}

                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.register_buffer("value", torch.ones(2))
                        alias_dict["peer"] = self.value

                    def forward(self):
                        if self.value is alias_dict["peer"]:
                            return self.value + 2
                        return self.value - 2

                model = Model()
                counter = CompileCounter()
                compiled = torch.compile(model, backend=counter, fullgraph=True)
                for _ in range(8):
                    torch.testing.assert_close(compiled(), torch.full((2,), 3.0))
                assert counter.frame_count == 1, counter.frame_count
                original_entry = _debug_get_cache_entry_list(Model.forward.__code__)
                assert len(original_entry) == 1, len(original_entry)
                assert original_entry[0]._debug_fast_guard_enabled

                old_value = model.value
                model.value = torch.zeros(2)
                torch.testing.assert_close(compiled(), torch.full((2,), -2.0))
                assert counter.frame_count == 2, counter.frame_count
                assert original_entry[0]._debug_fast_guard_enabled

                model.value = old_value
                torch.testing.assert_close(compiled(), torch.full((2,), 3.0))
                assert counter.frame_count == 2, counter.frame_count
                assert original_entry[0]._debug_fast_guard_enabled

            run_case(False)
            run_case(True)
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_dimension_marking_guard(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            GLOBAL_DICT = {"used": 1, "noise": [0]}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)
                    self.weight = torch.nn.Parameter(torch.ones(2))

                def forward(self, x):
                    return (
                        self._cached_tensor
                        + self.weight
                        + x
                        + GLOBAL_DICT["used"]
                    )

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)
            for i in range(8):
                GLOBAL_DICT["noise"] = [i]
                torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(original_entry) == 1, len(original_entry)
            assert original_entry[0]._debug_fast_guard_enabled

            model._cached_tensor._dynamo_dynamic_indices = {0}
            model._cached_tensor._has_dynamo_dim_marking = True
            GLOBAL_DICT["noise"] = [100]
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            del model._cached_tensor._dynamo_dynamic_indices
            del model._cached_tensor._has_dynamo_dim_marking
            GLOBAL_DICT["noise"] = [200]
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            model.weight._has_dynamo_dim_marking = None
            GLOBAL_DICT["noise"] = [300]
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 3, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            del model.weight._has_dynamo_dim_marking
            GLOBAL_DICT["noise"] = [400]
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 3, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled

            torch.Tensor._has_dynamo_dim_marking = False
            try:
                GLOBAL_DICT["noise"] = [500]
                torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
                assert counter.frame_count == 4, counter.frame_count
                assert original_entry[0]._debug_fast_guard_enabled
            finally:
                del torch.Tensor._has_dynamo_dim_marking

            GLOBAL_DICT["noise"] = [600]
            torch.testing.assert_close(compiled(x), torch.full((2,), 3.0))
            assert counter.frame_count == 4, counter.frame_count
            assert original_entry[0]._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_rejects_existing_dimension_marking(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2)
                    self._cached_tensor._dynamo_dynamic_indices = {0}
                    self._cached_tensor._has_dynamo_dim_marking = True

                def forward(self, x):
                    return self._cached_tensor + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), torch.ones(2))
            assert counter.frame_count == 1, counter.frame_count
            entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(entries) == 1, len(entries)
            assert not entries[0]._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_rejects_tensor_subclass(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            class TensorSubclass(torch.Tensor):
                pass

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._cached_tensor = torch.ones(2).as_subclass(TensorSubclass)

                def forward(self, x):
                    return self._cached_tensor + x

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(entries) == 1, len(entries)
            assert not entries[0]._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_rejects_dimension_marking_descriptor(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            class MissingSentinel:
                def __get__(self, instance, owner):
                    raise AttributeError("sentinel remains absent")

            torch.Tensor._has_dynamo_dim_marking = MissingSentinel()
            try:
                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self._cached_tensor = torch.ones(2)

                    def forward(self, x):
                        return self._cached_tensor + x

                model = Model()
                counter = CompileCounter()
                compiled = torch.compile(
                    model, backend=counter, fullgraph=True, dynamic=True
                )
                x = torch.zeros(2)
                for _ in range(8):
                    torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 1, counter.frame_count
                entries = _debug_get_cache_entry_list(Model.forward.__code__)
                assert len(entries) == 1, len(entries)
                assert not entries[0]._debug_fast_guard_enabled
            finally:
                del torch.Tensor._has_dynamo_dim_marking
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_retains_exact_list_items(self):
        script = """
            import gc
            import weakref
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = [torch.ones(2)]

                def forward(self, x):
                    return x + self.values[0] + len(self.values)

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _only_cache_entry(Model.forward.__code__)
            assert original_entry._debug_fast_guard_enabled

            original_value = model.values[0]
            original_ref = weakref.ref(original_value)
            model.values[0] = torch.full((2,), 4, dtype=torch.int64)
            del original_value
            gc.collect()
            assert original_ref() is not None

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            model.values[0] = original_ref()
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_bounds_unstable_and_large_list_training(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class UnstableListModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = [object()]

                def forward(self, x):
                    return x + len(self.values)

            unstable_model = UnstableListModel()
            unstable_counter = CompileCounter()
            unstable_compiled = torch.compile(
                unstable_model, backend=unstable_counter, fullgraph=True
            )
            x = torch.zeros(2)
            torch.testing.assert_close(
                unstable_compiled(x), unstable_model(x)
            )
            unstable_model.values[0] = object()
            torch.testing.assert_close(
                unstable_compiled(x), unstable_model(x)
            )
            for _ in range(8):
                unstable_model.values[0] = object()
                torch.testing.assert_close(
                    unstable_compiled(x), unstable_model(x)
                )
            assert unstable_counter.frame_count == 1, unstable_counter.frame_count
            unstable_entry = _only_cache_entry(
                UnstableListModel.forward.__code__
            )

            # Once the changing signature exhausts its training budget, later
            # stable calls stay on the original guard path instead of resuming
            # unbounded recording and eventually enabling a plan.
            for _ in range(8):
                torch.testing.assert_close(
                    unstable_compiled(x), unstable_model(x)
                )
            assert unstable_counter.frame_count == 1, unstable_counter.frame_count
            assert not unstable_entry._debug_fast_guard_enabled

            class BoundaryListModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = [object()]

                def forward(self, x):
                    return x + len(self.values)

            boundary_model = BoundaryListModel()
            boundary_counter = CompileCounter()
            boundary_compiled = torch.compile(
                boundary_model, backend=boundary_counter, fullgraph=True
            )
            torch.testing.assert_close(boundary_compiled(x), boundary_model(x))
            boundary_model.values[0] = object()
            torch.testing.assert_close(boundary_compiled(x), boundary_model(x))
            for _ in range(7):
                boundary_model.values[0] = object()
                torch.testing.assert_close(
                    boundary_compiled(x), boundary_model(x)
                )
            for _ in range(3):
                torch.testing.assert_close(
                    boundary_compiled(x), boundary_model(x)
                )
            assert boundary_counter.frame_count == 1, boundary_counter.frame_count
            boundary_entry = _only_cache_entry(
                BoundaryListModel.forward.__code__
            )
            assert boundary_entry._debug_fast_guard_enabled

            class OversizedListModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.values = [object() for _ in range(4097)]

                def forward(self, x):
                    return x + len(self.values)

            oversized_model = OversizedListModel()
            oversized_counter = CompileCounter()
            oversized_compiled = torch.compile(
                oversized_model, backend=oversized_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(
                    oversized_compiled(x), oversized_model(x)
                )
            assert oversized_counter.frame_count == 1, oversized_counter.frame_count
            oversized_entry = _only_cache_entry(
                OversizedListModel.forward.__code__
            )
            assert not oversized_entry._debug_fast_guard_enabled

            class AggregateWithinBudgetModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    marker = object()
                    self.groups = tuple(
                        [marker] * 4096 for _ in range(16)
                    )

                def forward(self, x):
                    return x + sum(len(group) for group in self.groups)

            within_model = AggregateWithinBudgetModel()
            within_counter = CompileCounter()
            within_compiled = torch.compile(
                within_model, backend=within_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(
                    within_compiled(x), within_model(x)
                )
            assert within_counter.frame_count == 1, within_counter.frame_count
            within_entry = _only_cache_entry(
                AggregateWithinBudgetModel.forward.__code__
            )
            assert within_entry._debug_fast_guard_enabled

            class AggregateOverBudgetModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    marker = object()
                    self.groups = tuple(
                        [marker] * 4096 for _ in range(17)
                    )

                def forward(self, x):
                    return x + sum(len(group) for group in self.groups)

            over_model = AggregateOverBudgetModel()
            over_counter = CompileCounter()
            over_compiled = torch.compile(
                over_model, backend=over_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(over_compiled(x), over_model(x))
            assert over_counter.frame_count == 1, over_counter.frame_count
            over_entry = _only_cache_entry(
                AggregateOverBudgetModel.forward.__code__
            )
            assert not over_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_preserves_local_state_transitions(self):
        script = """
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("scale", torch.ones(2))

                def forward(self, x):
                    state = 0
                    if torch.is_grad_enabled():
                        state += 1
                    if torch.is_inference_mode_enabled():
                        state += 2
                    if torch.is_autocast_enabled("cpu"):
                        state += 4
                    return x + self.scale + state

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(model, backend=counter, fullgraph=True)
            x = torch.zeros(2)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count
            original_entry = _only_cache_entry(Model.forward.__code__)
            assert original_entry._debug_fast_guard_enabled

            with torch.no_grad():
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            with torch.inference_mode():
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 3, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            with torch.autocast("cpu"):
                torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 4, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 4, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_static_module_attr_binding(self):
        script = """
            import types

            namespace = types.ModuleType("fastguard_static_module")
            original = torch.ones(2)
            namespace.scale = original

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = namespace

                def forward(self, x):
                    return self.namespace.scale + x

            model = Model()
            compiled, counter, original_entry, x = _warm_model(model)

            namespace.scale = torch.full((2,), 3.0)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            namespace.scale = original
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            dynamic_values = [torch.ones(2)]
            dynamic_namespace = types.ModuleType("fastguard_dynamic_module")

            def module_getattr(name):
                if name == "scale":
                    return dynamic_values[0]
                raise AttributeError(name)

            dynamic_namespace.__getattr__ = module_getattr

            class DynamicModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = dynamic_namespace

                def forward(self, x):
                    return self.namespace.scale + x

            dynamic_model = DynamicModel()
            _warm_model(dynamic_model, enabled=False)
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_static_type_attr_binding(self):
        script = """
            original = torch.ones(2)

            class Namespace:
                scale = original

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = Namespace

                def forward(self, x):
                    return self.namespace.scale + x

            model = Model()
            compiled, counter, original_entry, x = _warm_model(model)

            Namespace.scale = torch.full((2,), 3.0)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            Namespace.scale = original
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            class DescriptorValue:
                def __init__(self, value):
                    self.value = value

            original_value = DescriptorValue(torch.ones(2))
            replacement_value = DescriptorValue(torch.full((2,), 5.0))

            class MutableDescriptorNamespace:
                scale = original_value

            class MutableDescriptorModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = MutableDescriptorNamespace

                def forward(self, x):
                    return self.namespace.scale.value + x

            mutable_descriptor_model = MutableDescriptorModel()
            mutable_compiled, mutable_counter, mutable_entry, mutable_x = (
                _warm_model(mutable_descriptor_model)
            )

            def descriptor_get(self, obj, owner):
                return replacement_value

            DescriptorValue.__get__ = descriptor_get
            try:
                torch.testing.assert_close(
                    mutable_compiled(mutable_x),
                    mutable_descriptor_model(mutable_x),
                )
                assert mutable_counter.frame_count == 2, mutable_counter.frame_count
            finally:
                del DescriptorValue.__get__

            torch.testing.assert_close(
                mutable_compiled(mutable_x), mutable_descriptor_model(mutable_x)
            )
            assert mutable_counter.frame_count == 2, mutable_counter.frame_count
            assert mutable_entry._debug_fast_guard_enabled

            class Descriptor:
                def __init__(self):
                    self.value = torch.ones(2)

                def __get__(self, obj, owner):
                    return self.value

            class DynamicNamespace:
                scale = Descriptor()

            class DynamicModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.namespace = DynamicNamespace

                def forward(self, x):
                    return self.namespace.scale + x

            dynamic_model = DynamicModel()
            _warm_model(dynamic_model, enabled=False)
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_type_method_binding(self):
        script = """
            import types

            class Model(torch.nn.Module):
                def helper(self, x):
                    return x + 1

                def forward(self, x):
                    return self.helper(x)

            model = Model()
            compiled, counter, original_entry, x = _warm_model(model)

            def replacement(self, value):
                return value + 4

            model.helper = types.MethodType(replacement, model)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            del model.helper
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            original_helper = Model.helper
            Model.helper = replacement
            try:
                torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 3, counter.frame_count
            finally:
                Model.helper = original_helper

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 3, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            class ClassMethodModel(torch.nn.Module):
                @classmethod
                def helper(cls, value):
                    return value + 1

                def forward(self, value):
                    return self.helper(value)

            class_method_model = ClassMethodModel()
            _warm_model(class_method_model, enabled=False)
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_function_code_binding(self):
        script = """
            def helper(value):
                return value + 1

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.helper = helper

                def forward(self, x):
                    return self.helper(x)

            model = Model()
            compiled, counter, original_entry, x = _warm_model(model)
            _, accessors = _guard_tree_shapes(original_entry)
            assert any(
                kind == "CodeGuardAccessor" and source.startswith("L['self']")
                for source, kind in accessors
            ), accessors
            original_code = helper.__code__

            def replacement(value):
                return value + 4

            helper.__code__ = replacement.__code__
            try:
                torch.testing.assert_close(compiled(x), model(x))
                assert counter.frame_count == 2, counter.frame_count
            finally:
                helper.__code__ = original_code

            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled
        """
        self._run_guard_lookup_memo_script(script)

    def test_actual_partial_exact_set_equals_token(self):
        script = """
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = {1, 2}

                def forward(self, x):
                    if self.mode == {1, 2}:
                        return x + 1
                    return x - 1

            model = Model()
            compiled, counter, original_entry, x = _warm_model(model)
            leaves, _ = _guard_tree_shapes(original_entry)
            assert any(
                kind == "EQUALS_MATCH" and source.startswith("L['self']")
                for source, kind in leaves
            ), leaves
            model.mode.add(3)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count

            model.mode.remove(3)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 2, counter.frame_count
            assert original_entry._debug_fast_guard_enabled

            class FancySet(set):
                pass

            class FancySetModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = FancySet({1, 2})

                def forward(self, value):
                    if self.mode == {1, 2}:
                        return value + 1
                    return value - 1

            fancy_model = FancySetModel()
            _warm_model(fancy_model, enabled=False)

            class Comparable(int):
                __hash__ = int.__hash__

                def __eq__(self, other):
                    return int.__eq__(self, other)

            expected_mode = {Comparable(1), Comparable(2)}

            class CallbackSetModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mode = {Comparable(1), Comparable(2)}

                def forward(self, value):
                    if self.mode == expected_mode:
                        return value + 1
                    return value - 1

            callback_model = CallbackSetModel()
            _, _, callback_entry, _ = _warm_model(callback_model, enabled=False)
            callback_leaves, _ = _guard_tree_shapes(callback_entry)
            assert any(
                kind == "EQUALS_MATCH" and source.startswith("L['self']")
                for source, kind in callback_leaves
            ), callback_leaves
        """
        self._run_guard_lookup_memo_script(script)

    def test_precompile_entry_owns_actual_partial_receipt(self):
        script = """
            import threading

            from torch._C._dynamo.eval_frame import (
                _debug_get_precompile_entries,
                _load_precompile_entry,
                _reset_precompile_entries,
            )
            from torch._dynamo.testing import CompileCounter

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = 2.0

                def forward(self, x):
                    return x * self.scale

            model = Model()
            counter = CompileCounter()
            compiled = torch.compile(
                model, backend=counter, fullgraph=True, dynamic=True
            )
            x = torch.ones(4)
            torch.testing.assert_close(compiled(x), model(x))
            assert counter.frame_count == 1, counter.frame_count

            code = Model.forward.__code__
            source_entry = _only_cache_entry(code)
            should_block = threading.Event()
            guard_entered = threading.Event()
            guard_release = threading.Event()

            def blocking_guard(_):
                if should_block.is_set():
                    guard_entered.set()
                    assert guard_release.wait(10), "timed out waiting for reset"
                return True

            source_entry.guard_manager.root.add_lambda_guard(
                blocking_guard, ["blocking precompile guard"], None
            )
            _reset_precompile_entries(code)
            _load_precompile_entry(
                code, source_entry.guard_manager, source_entry.code
            )

            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))

            precompile_entries = _debug_get_precompile_entries(code)
            assert len(precompile_entries) == 1, len(precompile_entries)
            assert precompile_entries[0]._debug_fast_guard_enabled

            # Do not let the debug wrapper mask the container/lookup ownership
            # race that this reset is intended to exercise.
            del precompile_entries

            errors = []

            def run_lookup():
                try:
                    torch.testing.assert_close(compiled(x), model(x))
                except BaseException as exc:
                    errors.append(exc)

            should_block.set()
            worker = threading.Thread(target=run_lookup)
            worker.start()
            assert guard_entered.wait(10), "guard lookup did not block"
            _reset_precompile_entries(code)
            guard_release.set()
            worker.join(10)
            assert not worker.is_alive(), "guard lookup did not finish"
            assert not errors, errors
            assert _debug_get_precompile_entries(code) == []
            torch.testing.assert_close(compiled(x), model(x))
        """
        self._run_guard_lookup_memo_script(script)

    def test_guard_lookup_memo_defaults_off(self):
        script = """
            import torch
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list
            from torch._dynamo.testing import CompileCounter

            assert torch._dynamo.config.enable_guard_lookup_memo is False
            torch._dynamo.config.use_lamba_guard_for_object_aliasing = False

            def relation_guard_names(entry):
                names = set()
                pending = [entry.guard_manager.root]
                while pending:
                    manager = pending.pop()
                    names.update(
                        type(guard).__name__
                        for guard in manager.get_leaf_guards()
                    )
                    pending.extend(manager.get_child_managers())
                return names

            class Model(torch.nn.Module):
                def forward(self, x):
                    return x + 1

            model = Model()
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            x = torch.ones(3)
            for _ in range(8):
                torch.testing.assert_close(compiled(x), model(x))

            cache_entries = _debug_get_cache_entry_list(Model.forward.__code__)
            assert len(cache_entries) == 1, len(cache_entries)
            assert not cache_entries[0]._debug_fast_guard_enabled

            class Holder:
                def __init__(self, value):
                    self.scale = torch.full((3,), value)
                    self.bias = torch.full((3,), value + 1)

            class OrdinaryAccessorModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.holders = [Holder(float(i)) for i in range(32)]

                def forward(self, x):
                    result = x
                    for holder in self.holders:
                        result = (
                            result
                            + holder.scale
                            + holder.__dict__["bias"]
                        )
                    return result

            accessor_model = OrdinaryAccessorModel()
            accessor_counter = CompileCounter()
            accessor_compiled = torch.compile(
                accessor_model, backend=accessor_counter, fullgraph=True
            )
            for _ in range(8):
                torch.testing.assert_close(
                    accessor_compiled(x), accessor_model(x)
                )
            assert (
                accessor_counter.frame_count == 1
            ), accessor_counter.frame_count
            accessor_entries = _debug_get_cache_entry_list(
                OrdinaryAccessorModel.forward.__code__
            )
            assert len(accessor_entries) == 1, len(accessor_entries)
            assert not accessor_entries[0]._debug_fast_guard_enabled

            class ObjectAliasModel(torch.nn.Module):
                def forward(self, x, y):
                    if x is y:
                        return x + y
                    return x - y

            object_model = ObjectAliasModel()
            object_counter = CompileCounter()
            object_compiled = torch.compile(
                object_model, backend=object_counter, fullgraph=True
            )
            same = torch.ones(3)
            for _ in range(8):
                torch.testing.assert_close(
                    object_compiled(same, same), object_model(same, same)
                )
            assert object_counter.frame_count == 1, object_counter.frame_count
            object_entries = _debug_get_cache_entry_list(
                ObjectAliasModel.forward.__code__
            )
            assert len(object_entries) == 1, len(object_entries)
            assert "OBJECT_ALIASING" in relation_guard_names(object_entries[0])
            assert not object_entries[0]._debug_fast_guard_enabled

            different = torch.zeros_like(same)
            torch.testing.assert_close(
                object_compiled(same, different), object_model(same, different)
            )
            assert object_counter.frame_count == 2, object_counter.frame_count
            torch.testing.assert_close(
                object_compiled(same, same), object_model(same, same)
            )
            assert object_counter.frame_count == 2, object_counter.frame_count
            assert all(
                not entry._debug_fast_guard_enabled
                for entry in _debug_get_cache_entry_list(
                    ObjectAliasModel.forward.__code__
                )
            )

            class NoTensorAliasModel(torch.nn.Module):
                def forward(self, x, y):
                    return x + 2 * y

            no_tensor_model = NoTensorAliasModel()
            no_tensor_counter = CompileCounter()
            no_tensor_compiled = torch.compile(
                no_tensor_model, backend=no_tensor_counter, fullgraph=True
            )
            left = torch.ones(3)
            right = torch.zeros(3)
            for _ in range(8):
                torch.testing.assert_close(
                    no_tensor_compiled(left, right), no_tensor_model(left, right)
                )
            assert (
                no_tensor_counter.frame_count == 1
            ), no_tensor_counter.frame_count
            no_tensor_entries = _debug_get_cache_entry_list(
                NoTensorAliasModel.forward.__code__
            )
            assert len(no_tensor_entries) == 1, len(no_tensor_entries)
            assert "NO_TENSOR_ALIASING" in relation_guard_names(
                no_tensor_entries[0]
            )
            assert not no_tensor_entries[0]._debug_fast_guard_enabled

            torch.testing.assert_close(
                no_tensor_compiled(left, left), no_tensor_model(left, left)
            )
            assert (
                no_tensor_counter.frame_count == 2
            ), no_tensor_counter.frame_count
            torch.testing.assert_close(
                no_tensor_compiled(left, right), no_tensor_model(left, right)
            )
            assert (
                no_tensor_counter.frame_count == 2
            ), no_tensor_counter.frame_count
            assert all(
                not entry._debug_fast_guard_enabled
                for entry in _debug_get_cache_entry_list(
                    NoTensorAliasModel.forward.__code__
                )
            )
        """
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=os.getcwd(),
            check=True,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
