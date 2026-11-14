# Owner(s): ["module: dynamo"]

import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
import weakref
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import UncapturedHigherOrderOpError, Unsupported
from torch._dynamo.testing import same
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


try:
    from . import utils
except ImportError:
    import utils


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def Foo():
    return Pair(1, 1)


g_counter = 1
g_list = [0, 1, 2]
g_dict = {"a": 0, "b": 1}
g_object = Foo()
g_tensor = torch.zeros(10)


_name: int = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


def reset_name():
    global _name
    _name = 0


class TestGlobals(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_store_global_1(self):
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_2(self):
        def fn(x):
            global g_counter
            val = x + g_counter
            g_counter += 1
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        """Wrap the second call with torch._dynamo as well"""
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertTrue(same(res2 - res1, 2 * torch.ones(10)))

    def test_store_global_new(self):
        def fn(x):
            # Test create a new global
            global g_counter_new
            g_counter_new = x + 1
            return x + g_counter_new

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        self.assertTrue(same(res1, x + x + 1))

    def test_store_global_list(self):
        def fn(x):
            global g_list
            val = x + g_list[1]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_list[1] += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_list_2(self):
        def fn(x):
            global g_list
            val = x + g_list[1]
            g_list = [x + 1 for x in g_list]
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict(self):
        def fn(x):
            global g_dict
            val = x + g_dict["b"]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_dict["b"] += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict_2(self):
        def fn(x):
            global g_dict
            g_dict = {key: value + 1 for key, value in g_dict.items()}
            val = x + g_dict["b"]
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_object(self):
        def fn(x):
            global g_object
            val = x + g_object.y
            g_object.y += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_cross_file(self):
        def fn(x):
            val = x + utils.g_tensor_export
            utils.g_tensor_export = utils.g_tensor_export + 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_inline_1(self):
        # Borrowed from test_python_autograd.py
        class Variable:
            def __init__(self, value: torch.Tensor, name: str | None = None):
                self.value = value
                self.name = name or fresh_name()

        def fn(a, b):
            a = Variable(a)
            b = Variable(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()

    def test_store_global_inline_2(self):
        # Borrowed from test_python_autograd.py
        class Variable:
            def __init__(self, value: torch.Tensor, name: str | None = None):
                self.value = value
                self.name = name or fresh_name()

            @staticmethod
            def constant(value: torch.Tensor, name: str | None = None):
                return Variable(value, name)

        def fn(a, b):
            a = Variable.constant(a)
            b = Variable.constant(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()

    def test_store_global_crossfile_inline(self):
        try:
            from . import mock_store_global_crossfile_inline
        except ImportError:
            import mock_store_global_crossfile_inline

        @torch.compile(backend="eager")
        def fn(x):
            mock_store_global_crossfile_inline.set_flag_true()
            mock_store_global_crossfile_inline.set_flag_false()
            return x + 1

        @torch.compile(backend="eager")
        def fn_set_true(x):
            mock_store_global_crossfile_inline.set_flag_true()
            return x + 1

        fn_set_true(torch.ones(2, 2))
        self.assertTrue(mock_store_global_crossfile_inline.global_flag)
        fn(torch.ones(2, 2))
        self.assertFalse(mock_store_global_crossfile_inline.global_flag)

    def test_unregistered_importlib_module_globals(self):
        module_name = "test_dynamo_unregistered_module_181243"
        self.assertNotIn(module_name, sys.modules)

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w") as f:
                f.write(
                    """
import functools
import torch


def _helper(x, scale):
    return x.sin() * scale


def my_fn(x, scale):
    return _helper(x, scale)


fn = functools.partial(my_fn, scale=2)
"""
                )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertNotIn(module_name, sys.modules)

            x = torch.randn(4, 4)
            compiled = torch.compile(mod.fn, backend="eager", fullgraph=True)
            self.assertTrue(same(compiled(x), mod.fn(x)))

    def test_store_global_in_unregistered_importlib_module(self):
        module_name = "test_dynamo_unregistered_store_global_181243"
        self.assertNotIn(module_name, sys.modules)

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w") as f:
                f.write(
                    """
import functools


flag = 0


def my_fn(x, scale):
    global flag
    flag = scale
    return x + flag


fn = functools.partial(my_fn, scale=2)
"""
                )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertNotIn(module_name, sys.modules)

            x = torch.randn(4, 4)
            with self.assertRaisesRegex(
                Unsupported, "STORE_GLOBAL in non-module globals"
            ):
                torch.compile(mod.fn, backend="eager", fullgraph=True)(x)

            mod.flag = 0
            compiled = torch.compile(mod.fn, backend="eager")
            self.assertTrue(same(compiled(x), x + 2))
            self.assertEqual(mod.flag, 2)


@contextlib.contextmanager
def crossfile_globals(**updates):
    try:
        from . import mock_store_global_crossfile_inline
    except ImportError:
        import mock_store_global_crossfile_inline

    with mock.patch.dict(mock_store_global_crossfile_inline.__dict__, updates):
        yield mock_store_global_crossfile_inline


def _delete_missing_fn(name):
    if name == "len":

        def fn(x):
            global len
            del len
            return x + 1

    else:

        def fn(x):
            global _dg_a
            del _dg_a
            return x + 1

    return fn


def _delete_twice_fn(prestore):
    if prestore:

        def fn(x):
            global _dg_a
            _dg_a = 1
            del _dg_a
            del _dg_a
            return x + 1

    else:

        def fn(x):
            global _dg_a
            del _dg_a
            del _dg_a
            return x + 1

    return fn


def _delete_then_read_fn(name):
    if name == "len":

        def fn(x):
            global len
            del len
            return x + len([1, 2])

    else:

        def fn(x):
            global _dg_a
            del _dg_a
            try:
                y = _dg_a * 2
            except NameError:
                y = x * 0
            return x + y

    return fn


@instantiate_parametrized_tests
class TestDeleteGlobal(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _assert_delete_missing_global(
        self, fn, name, args=(), still_bound=False, namespace=None
    ):
        if namespace is None:
            namespace = globals()
        with self.assertRaisesRegex(
            Unsupported, f"raised exception NameError.*name '{name}' is not defined"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
        if still_bound:
            self.assertIn(name, namespace)
        else:
            self.assertNotIn(name, namespace)
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        with self.assertRaisesRegex(NameError, f"name '{name}' is not defined"):
            torch.compile(fn, backend=cnt, fullgraph=False)(*args)
        self.assertEqual(cnt.frame_count, 0)

    def test_delete_global(self):
        with mock.patch.dict(globals(), {"_dg_a": 10}):

            def fn(x):
                global _dg_a
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    @parametrize("name", ("_dg_a", "len"))
    def test_delete_missing_global(self, name):
        with mock.patch.dict(globals()):
            self._assert_delete_missing_global(
                _delete_missing_fn(name), name, (torch.ones(2, 2),)
            )

    def test_delete_missing_global_caught_by_caller(self):
        with mock.patch.dict(globals()):

            def inner():
                global _dg_a
                del _dg_a

            def fn(x):
                try:
                    inner()
                except NameError:
                    pass
                return x + 1

            x = torch.ones(2, 2)
            for fullgraph in (True, False):
                with self.subTest(fullgraph=fullgraph):
                    torch._dynamo.reset()
                    cnt = torch._dynamo.testing.CompileCounter()
                    opt_fn = torch.compile(fn, backend=cnt, fullgraph=fullgraph)
                    self.assertEqual(opt_fn(x), x + 1)
                    self.assertEqual(cnt.frame_count, 1)

    # policy=error flags the zero-net store+delete as an unreplayed side
    # effect; the shape itself matches eager without the policy.
    def test_delete_global_created_then_deleted(self):
        with mock.patch.dict(globals()):
            with torch._dynamo.config.patch(side_effect_replay_policy="error"):

                def fn(x):
                    global _dg_a
                    _dg_a = 1
                    del _dg_a
                    return x + 1

                x = torch.ones(2, 2)
                with self.assertRaisesRegex(RuntimeError, "found certain side effects"):
                    torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertNotIn("_dg_a", globals())

    @parametrize("preexisting", (True, False))
    def test_delete_global_twice(self, preexisting):
        with mock.patch.dict(globals(), {"_dg_a": 1} if preexisting else {}):
            self._assert_delete_missing_global(
                _delete_twice_fn(not preexisting),
                "_dg_a",
                (torch.ones(2, 2),),
                still_bound=preexisting,
            )
            if preexisting:
                self.assertNotIn("_dg_a", globals())

    # The re-bound name keeps its old __dict__ slot instead of appending
    # where eager re-inserts it.
    @unittest.expectedFailure
    def test_delete_global_then_store_matches_eager_dict_order(self):
        with mock.patch.dict(globals(), {"_dg_a": 1, "_dg_pad": 0}):

            def fn(x):
                global _dg_a
                del _dg_a
                _dg_a = 7
                return x + 1

            x = torch.ones(2, 2)
            torch.compile(fn, backend="eager", fullgraph=True)(x)
            compiled_position = list(globals()).index("_dg_a")
            globals()["_dg_a"] = 1
            fn(x)
            self.assertEqual(compiled_position, list(globals()).index("_dg_a"))

    def test_delete_global_read_write_chain(self):
        with mock.patch.dict(globals(), {"_dg_a": 1}):

            def fn(x):
                global _dg_a
                _dg_a = x + 1
                y = _dg_a * 2
                del _dg_a
                _dg_a = y + 5
                z = _dg_a * 3
                return x + y + z

            x = torch.ones(2, 2)
            expected = fn(x)
            expected_global = globals()["_dg_a"]
            globals()["_dg_a"] = 1

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), expected)
            self.assertEqual(globals()["_dg_a"], expected_global)

    def test_delete_global_created_then_deleted_read(self):
        with mock.patch.dict(globals()):

            def fn(x):
                global _dg_a
                _dg_a = x + 1
                del _dg_a
                try:
                    y = _dg_a * 2
                except NameError:
                    y = x * 0
                return x + y

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x)
            self.assertNotIn("_dg_a", globals())

    @parametrize("name", ("_dg_a", "len"))
    def test_delete_global_then_read(self, name):
        with mock.patch.dict(
            globals(),
            {"_dg_a": 1} if name == "_dg_a" else {"len": lambda _: "shadow"},
        ):
            x = torch.ones(2, 2)
            opt_fn = torch.compile(
                _delete_then_read_fn(name), backend="eager", fullgraph=True
            )
            self.assertEqual(opt_fn(x), x + 2 if name == "len" else x)
            self.assertNotIn(name, globals())

    def test_delete_tensor_global(self):
        with mock.patch.dict(globals()):

            def fn(x):
                global _dg_a
                y = _dg_a + 1
                del _dg_a
                return x + y

            x = torch.ones(2, 2)
            tensor = torch.ones(2, 2)
            globals()["_dg_a"] = tensor
            ref = weakref.ref(tensor)
            del tensor

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + x + 1)
            self.assertNotIn("_dg_a", globals())
            self.assertIsNone(ref(), "deleting the global kept the tensor alive")

    def test_delete_global_crossfile_inline(self):
        with crossfile_globals(delete_value=True) as mod:

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_value", mod.__dict__)

    def test_delete_global_crossfile_missing(self):
        with crossfile_globals() as mod:

            def fn(x):
                mod.delete_missing_value_fn()
                return x + 1

            self._assert_delete_missing_global(
                fn,
                "delete_missing_value",
                (torch.ones(2, 2),),
                namespace=mod.__dict__,
            )

    def test_delete_global_crossfile_then_read(self):
        with crossfile_globals(delete_value=1) as mod:

            def fn(x):
                try:
                    mod.delete_value_fn()
                    y = mod.read_delete_value_fn()
                except NameError:
                    y = x * 0
                return x + y

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x)
            self.assertNotIn("delete_value", mod.__dict__)

    def test_delete_global_crossfile_then_getattr(self):
        with crossfile_globals(delete_value=1) as mod:

            def fn(x):
                mod.delete_value_fn()
                try:
                    return x + mod.delete_value
                except AttributeError:
                    return x - 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x - 1)
            self.assertNotIn("delete_value", mod.__dict__)

    def test_delete_global_crossfile_module_getattr(self):
        try:
            from . import mock_delete_global_pep562 as mod
        except ImportError:
            import mock_delete_global_pep562 as mod

        with mock.patch.dict(mod.__dict__):

            def fn(x):
                mod.create_delete_missing_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            fn(x)
            self.assertNotIn("create_delete_missing_value", mod.__dict__)
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL in module with __getattr__"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("create_delete_missing_value", mod.__dict__)

    def test_delete_global_crossfile_pending_dict_write(self):
        with crossfile_globals() as mod:

            def fn(x):
                mod.__dict__["delete_missing_value"] = 9
                try:
                    mod.delete_missing_value_fn()
                except NameError:
                    return x + 2
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_missing_value", mod.__dict__)
            mod.__dict__.pop("delete_missing_value", None)
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL with a materialized module dict"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("delete_missing_value", mod.__dict__)

    def test_delete_global_crossfile_pending_getattr(self):
        with crossfile_globals(delete_value=1) as mod:

            def fn(x):
                mod.__getattr__ = lambda name: 42
                mod.delete_value_fn()
                try:
                    return x + mod.delete_value
                except AttributeError:
                    return x - 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 42)
            self.assertNotIn("delete_value", mod.__dict__)
            mod.__dict__.pop("__getattr__", None)
            mod.__dict__["delete_value"] = 1
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL in module with __getattr__"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 42)
            self.assertNotIn("delete_value", mod.__dict__)

    # policy=error flags the zero-net store+delete as an unreplayed side
    # effect; the shape itself matches eager without the policy.
    def test_delete_global_crossfile_created_then_deleted(self):
        with crossfile_globals() as mod:
            with torch._dynamo.config.patch(side_effect_replay_policy="error"):

                @torch.compile(backend="eager", fullgraph=True)
                def fn(x):
                    mod.store_then_delete_missing_fn()
                    return x + 1

                x = torch.ones(2, 2)
                with self.assertRaisesRegex(RuntimeError, "found certain side effects"):
                    fn(x)
            self.assertNotIn("store_then_delete_missing_value", mod.__dict__)

    def test_delete_global_crossfile_created_then_deleted_replays_delete(self):
        with crossfile_globals() as mod:

            def fn(x):
                mod.store_then_delete_missing_fn()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("store_then_delete_missing_value", mod.__dict__)

    def test_delete_global_crossfile_read_write_chain(self):
        with crossfile_globals(store_read_delete_store_read_value=1) as mod:

            def fn(x):
                return mod.store_read_delete_store_read_value_fn(x) + x

            x = torch.ones(2, 2)
            expected = fn(x)
            expected_value = mod.store_read_delete_store_read_value
            mod.store_read_delete_store_read_value = 1

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), expected)
            self.assertEqual(mod.store_read_delete_store_read_value, expected_value)

    @parametrize("preexisting", (True, False))
    def test_delete_global_presence_flip(self, preexisting):
        with mock.patch.dict(globals(), {"_dg_a": 1} if preexisting else {}):

            def fn(x):
                global _dg_a
                y = x + 1
                try:
                    del _dg_a
                except NameError:
                    y = y + 1
                return y

            x = torch.ones(2, 2)
            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1 if preexisting else x + 2)
            self.assertEqual(cnt.frame_count, 1)
            if preexisting:
                self.assertNotIn("_dg_a", globals())
                self.assertEqual(opt_fn(x), x + 2)
                self.assertEqual(cnt.frame_count, 2)
            else:
                globals()["_dg_a"] = 1
                self.assertEqual(opt_fn(x), x + 1)
                self.assertNotIn("_dg_a", globals())
                self.assertEqual(cnt.frame_count, 2)

    def test_delete_global_crossfile_presence_flip(self):
        with crossfile_globals() as mod:
            mod.__dict__.pop("flip_value")

            def fn(x):
                y = x + 1
                try:
                    mod.delete_flip_value_fn()
                except NameError:
                    y = y + 1
                return y

            x = torch.ones(2, 2)
            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
            self.assertEqual(opt_fn(x), x + 2)
            self.assertEqual(cnt.frame_count, 1)

            mod.flip_value = 1
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 2)
            self.assertNotIn("flip_value", mod.__dict__)

    def test_delete_global_inside_hop(self):
        def true_fn(y):
            global _dg_hop
            del _dg_hop
            return y + 1

        def false_fn(y):
            return y - 1

        def fn(x):
            global _dg_hop
            _dg_hop = 1
            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

        with mock.patch.dict(globals()):
            with self.assertRaisesRegex(
                UncapturedHigherOrderOpError, "HOP: Unsafe side effect"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2, 2))
            self.assertNotIn("_dg_hop", globals())

        with mock.patch.dict(globals(), {"_dg_hop": 1}):

            def delete_existing_fn(y):
                global _dg_hop
                del _dg_hop
                return y + 1

            def delete_existing(x):
                return torch.cond(x.sum() > 0, delete_existing_fn, false_fn, (x,))

            with self.assertRaisesRegex(
                UncapturedHigherOrderOpError, "HOP: Unsafe side effect"
            ):
                torch.compile(delete_existing, backend="eager", fullgraph=True)(
                    torch.ones(2, 2)
                )
            self.assertIn("_dg_hop", globals())

    @parametrize("preexisting", (True, False))
    def test_delete_global_through_module_object(self, preexisting):
        me = sys.modules[__name__]
        with mock.patch.dict(globals(), {"_dg_a": 1} if preexisting else {}):

            def fn(x):
                global _dg_a
                me._dg_a = 5
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL with a pending module-object write"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_stale_dict_replay(self):
        me = sys.modules[__name__]
        with mock.patch.dict(globals(), {"_dg_a": 1}):

            def fn(x):
                global _dg_a
                me.__dict__["_dg_b"] = 2
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())
            self.assertIn("_dg_b", globals())
            globals().pop("_dg_b")
            globals()["_dg_a"] = 1
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL with a materialized module dict"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            globals().pop("_dg_b", None)
            globals()["_dg_a"] = 1
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())
            self.assertIn("_dg_b", globals())

    def test_delete_global_created_then_store_read(self):
        with mock.patch.dict(globals()):

            def fn(x):
                global _dg_a
                _dg_a = 1
                del _dg_a
                _dg_a = 2
                return x + _dg_a

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 2)
            self.assertEqual(globals()["_dg_a"], 2)
            globals().pop("_dg_a")
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 2)
            self.assertEqual(globals()["_dg_a"], 2)
            torch._dynamo.reset()
            globals().pop("_dg_a")
            opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
            self.assertEqual(opt_fn(x), x + 2)
            self.assertEqual(globals()["_dg_a"], 2)

    def test_delete_missing_global_read_after_caught_failed_del(self):
        with mock.patch.dict(globals()):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError:
                    pass
                return x + _dg_a  # noqa: F821

            x = torch.ones(2, 2)
            with self.assertRaisesRegex(NameError, "name '_dg_a' is not defined"):
                fn(x)
            with self.assertRaisesRegex(Unsupported, "name '_dg_a' is not defined"):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            torch._dynamo.reset()
            with self.assertRaisesRegex(NameError, "name '_dg_a' is not defined"):
                torch.compile(fn, backend="eager", fullgraph=False)(x)

    @parametrize(
        "shape",
        (
            "hasattr",
            "attr",
            "contains",
            "in_list",
            "getattr",
            "dir",
            "getattribute",
        ),
    )
    def test_delete_global_module_read_after(self, shape):
        me = sys.modules[__name__]

        def fn_hasattr(x):
            global _dg_a
            del _dg_a
            return x + (1 if hasattr(me, "_dg_a") else 0)

        def fn_attr(x):
            global _dg_a
            del _dg_a
            try:
                return x + me._dg_a
            except AttributeError:
                return x - 1

        def fn_contains(x):
            global _dg_a
            del _dg_a
            return x + (1 if "_dg_a" in me.__dict__ else 0)

        def fn_getattr(x):
            global _dg_a
            del _dg_a
            try:
                # The builtin getattr call is the shape under test.
                return x + getattr(me, "_dg_a")  # noqa: B009
            except AttributeError:
                return x - 1

        def fn_in_list(x):
            global _dg_a
            del _dg_a
            return x + (1 if "_dg_a" in list(me.__dict__) else 0)

        def fn_dir(x):
            global _dg_a
            del _dg_a
            return x + (1 if "_dg_a" in dir(me) else 0)

        def fn_getattribute(x):
            global _dg_a
            del _dg_a
            try:
                return x + me.__getattribute__("_dg_a")
            except AttributeError:
                return x - 1

        fn, expected, msg = {
            "hasattr": (fn_hasattr, lambda x: x, "module-object read"),
            "attr": (fn_attr, lambda x: x - 1, "module-object read"),
            "contains": (fn_contains, lambda x: x, "module-__dict__ access"),
            "in_list": (fn_in_list, lambda x: x, "module-__dict__ access"),
            "getattr": (fn_getattr, lambda x: x - 1, "module-object read"),
            "dir": (fn_dir, lambda x: x, "module-__dict__ access"),
            "getattribute": (fn_getattribute, lambda x: x - 1, "module-object read"),
        }[shape]
        with mock.patch.dict(globals(), {"_dg_a": 1}):
            x = torch.ones(2, 2)
            self.assertEqual(fn(x), expected(x))
            self.assertNotIn("_dg_a", globals())
            for fullgraph in (True, False):
                torch._dynamo.reset()
                globals()["_dg_a"] = 1
                with self.subTest(fullgraph=fullgraph):
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported, f"DELETE_GLOBAL followed by a {msg}"
                        ):
                            torch.compile(fn, backend="eager", fullgraph=True)(x)
                    else:
                        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
                        self.assertEqual(opt_fn(x), expected(x))
                        self.assertNotIn("_dg_a", globals())

    @parametrize("shape", ("attr", "setitem"))
    def test_delete_global_module_write_after(self, shape):
        me = sys.modules[__name__]

        def fn_attr(x):
            global _dg_a
            del _dg_a
            me._dg_a = 5
            return x + 1

        def fn_setitem(x):
            global _dg_a
            del _dg_a
            me.__dict__["_dg_a"] = 5
            return x + 1

        fn, expected, g_after, msg = {
            "attr": (fn_attr, lambda x: x + 1, 5, "module-object write"),
            "setitem": (fn_setitem, lambda x: x + 1, 5, "module-__dict__ access"),
        }[shape]
        with mock.patch.dict(globals(), {"_dg_a": 1}):
            x = torch.ones(2, 2)
            self.assertEqual(fn(x), expected(x))
            self.assertEqual(globals().get("_dg_a"), g_after)
            for fullgraph in (True, False):
                torch._dynamo.reset()
                globals()["_dg_a"] = 1
                with self.subTest(fullgraph=fullgraph):
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported, f"DELETE_GLOBAL followed by a {msg}"
                        ):
                            torch.compile(fn, backend="eager", fullgraph=True)(x)
                    else:
                        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
                        self.assertEqual(opt_fn(x), expected(x))
                        self.assertEqual(globals().get("_dg_a"), g_after)

    def test_delete_global_module_dict_alias_after(self):
        me = sys.modules[__name__]

        def fn(x):
            global _dg_a
            d = me.__dict__
            del _dg_a
            return x + (1 if "_dg_a" in list(d) else 0)

        with mock.patch.dict(globals(), {"_dg_a": 1}):
            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x)
            self.assertNotIn("_dg_a", globals())
            for fullgraph in (True, False):
                torch._dynamo.reset()
                globals()["_dg_a"] = 1
                with self.subTest(fullgraph=fullgraph):
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported,
                            "DELETE_GLOBAL followed by a module-__dict__ access",
                        ):
                            torch.compile(fn, backend="eager", fullgraph=True)(x)
                    else:
                        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
                        self.assertEqual(opt_fn(x), x)
                        self.assertNotIn("_dg_a", globals())

    def test_delete_global_module_dict_realized_before(self):
        me = sys.modules[__name__]

        def fn(x):
            global _dg_a
            d = me.__dict__
            d.get("_dg_pad")
            del _dg_a
            return x + (1 if "_dg_a" in list(d) else 0)

        with mock.patch.dict(globals(), {"_dg_a": 1, "_dg_pad": 0}):
            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x)
            self.assertNotIn("_dg_a", globals())
            for fullgraph in (True, False):
                torch._dynamo.reset()
                globals()["_dg_a"] = 1
                globals()["_dg_pad"] = 0
                with self.subTest(fullgraph=fullgraph):
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported,
                            "DELETE_GLOBAL with a materialized module dict",
                        ):
                            torch.compile(fn, backend="eager", fullgraph=True)(x)
                    else:
                        opt_fn = torch.compile(fn, backend="eager", fullgraph=False)
                        self.assertEqual(opt_fn(x), x)
                        self.assertNotIn("_dg_a", globals())

    def test_delete_global_in_unregistered_importlib_module(self):
        module_name = "test_dynamo_unregistered_delete_global_181243"
        self.assertNotIn(module_name, sys.modules)

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w") as f:
                f.write(
                    """
import functools


flag = 1


def my_fn(x):
    global flag
    del flag
    return x + 1


fn = functools.partial(my_fn)
"""
                )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertNotIn(module_name, sys.modules)

            x = torch.randn(4, 4)
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL in non-module globals"
            ):
                torch.compile(mod.fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(mod.flag, 1)
            with self.assertRaisesRegex(
                Unsupported, "DELETE_GLOBAL in unregistered module globals"
            ):
                torch.compile(mod.my_fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(mod.flag, 1)

            compiled = torch.compile(mod.fn, backend="eager")
            self.assertTrue(same(compiled(x), x + 1))
            self.assertNotIn("flag", mod.__dict__)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
