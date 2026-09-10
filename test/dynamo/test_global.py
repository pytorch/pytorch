# Owner(s): ["module: dynamo"]

import contextlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import same


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


_ABSENT = object()


@contextlib.contextmanager
def temp_globals(namespace, **updates):
    """Install `updates` into `namespace` (a module `__dict__`) temporarily.

    A value of `_ABSENT` removes the name for the duration of the block, which
    is how the DELETE_GLOBAL tests set up "this global does not exist".
    """
    saved = {name: namespace.get(name, _ABSENT) for name in updates}
    for name, value in updates.items():
        if value is _ABSENT:
            namespace.pop(name, None)
        else:
            namespace[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _ABSENT:
                namespace.pop(name, None)
            else:
                namespace[name] = value


def import_crossfile_mock():
    try:
        from . import mock_store_global_crossfile_inline
    except ImportError:
        import mock_store_global_crossfile_inline

    return mock_store_global_crossfile_inline


class TestDeleteGlobal(torch._dynamo.test_case.TestCase):
    """`DELETE_GLOBAL` semantics, with eager CPython as the specification.

    Unguarded deletes are exercised under both compile modes because the two
    differ. `fullgraph=True` traces to completion, so an exception observed while
    tracing is reported through the observed-exception convention --
    `Unsupported: Observed exception` -- which is what `LOAD_GLOBAL` of a missing
    name already does. Without `fullgraph` a graph break lets the frame fall back
    to eager, and the user sees the real exception type.
    """

    def _assert_delete_missing_global(self, fn, name, args=()):
        with self.assertRaisesRegex(Unsupported, "Observed exception"):
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
        torch._dynamo.reset()
        with self.assertRaisesRegex(NameError, f"name '{name}' is not defined"):
            torch.compile(fn, backend="eager")(*args)

    # ------------------------------------------------------------------
    # Existence and ordering within a single frame
    # ------------------------------------------------------------------

    def test_delete_global(self):
        with temp_globals(globals(), _dg_a=10):

            def fn(x):
                global _dg_a
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_then_read(self):
        # Eager raises NameError. Dynamo refuses to model the read instead, so
        # this is locked in as-is rather than compared against eager. This
        # matches the existing read-after-delete convention for locals and cells
        # (DELETE_FAST/LOAD_FAST and DELETE_DEREF/LOAD_DEREF graph-break the same
        # way); raise_name_error is deliberately not used on the read path.
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                del _dg_a
                # Reading a deleted global raises NameError.
                return _dg_a  # noqa: F821

            x = torch.ones(2, 2)
            # Tracing fails before the delete is replayed, so the global
            # survives.
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                Unsupported, "Attempted to read a deleted variable"
            ):
                opt_fn(x)
            self.assertEqual(_dg_a, 1)

            torch._dynamo.reset()
            with self.assertRaisesRegex(NameError, "name '_dg_a' is not defined"):
                torch.compile(fn, backend="eager")(x)
            self.assertNotIn("_dg_a", globals())

    def test_delete_missing_global(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                del _dg_a
                return x + 1

            self._assert_delete_missing_global(fn, "_dg_a", (torch.ones(2, 2),))

    def test_delete_missing_global_caught_by_caller(self):
        # The handler lives in the caller, not in the frame doing the delete.
        with temp_globals(globals(), _dg_a=_ABSENT):

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
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)

    def test_delete_missing_global_then_store_same(self):
        # Eager raises at the `del`, so the later store never runs.
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                del _dg_a
                _dg_a = 5
                return x + 1

            self._assert_delete_missing_global(fn, "_dg_a", (torch.ones(2, 2),))
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_created_then_deleted(self):
        with temp_globals(globals(), _dg_a=_ABSENT):
            # The pending store is cancelled by the delete, so nothing must be
            # reported as a replayed side effect.
            with torch._dynamo.config.patch(side_effect_replay_policy="error"):

                def fn(x):
                    global _dg_a
                    _dg_a = 1
                    del _dg_a
                    return x + 1

                x = torch.ones(2, 2)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_created_then_deleted_twice(self):
        # `_dg_a = 1; del _dg_a; del _dg_a`: the sentinel in `symbolic_globals`
        # outlives the store the first delete cancelled, so the second delete
        # must not read that sentinel as "the name is still bound".
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                _dg_a = 1
                del _dg_a
                # Deleting an already-deleted global raises NameError.
                del _dg_a  # noqa: F821
                return x + 1

            self._assert_delete_missing_global(fn, "_dg_a", (torch.ones(2, 2),))

    def test_delete_global_twice(self):
        # The second delete must see the name as gone.
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                del _dg_a
                # Deleting an already-deleted global raises NameError.
                del _dg_a  # noqa: F821
                return x + 1

            self._assert_delete_missing_global(fn, "_dg_a", (torch.ones(2, 2),))
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_then_store(self):
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                del _dg_a
                _dg_a = 7
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(_dg_a, 7)

    def test_delete_global_store_delete_store(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                _dg_a = 5
                del _dg_a
                _dg_a = 7
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(_dg_a, 7)

    def test_delete_global_read_then_delete(self):
        with temp_globals(globals(), _dg_a=3):

            def fn(x):
                global _dg_a
                y = _dg_a
                del _dg_a
                return x + y

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 3)
            self.assertNotIn("_dg_a", globals())

    def test_delete_missing_builtin(self):
        # `len` is neither a global nor deleted from builtins, so eager raises.
        with temp_globals(globals(), len=_ABSENT):

            def fn(x):
                global len
                del len
                return x + 1

            self._assert_delete_missing_global(fn, "len", (torch.ones(2, 2),))

    def test_delete_global_then_missing_builtin(self):
        # The raise on `len` happens after `_dg_a` was deleted, so the delete
        # must already have taken effect.
        with temp_globals(globals(), _dg_a=1, len=_ABSENT):

            def fn(x):
                global _dg_a, len
                del _dg_a
                del len
                return x + 1

            self._assert_delete_missing_global(fn, "len", (torch.ones(2, 2),))
            self.assertNotIn("_dg_a", globals())

    def test_delete_two_globals(self):
        with temp_globals(globals(), _dg_a=1, _dg_b=2):

            def fn(x):
                global _dg_a, _dg_b
                del _dg_a
                del _dg_b
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())
            self.assertNotIn("_dg_b", globals())

    def test_delete_tensor_global(self):
        with temp_globals(globals(), _dg_a=None):

            def fn(x):
                global _dg_a
                _dg_a = torch.ones(4)
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_inlined_same_module(self):
        # An inlined callee in this module delegates to the root-frame handler.
        with temp_globals(globals(), _dg_a=1):

            def inner():
                global _dg_a
                del _dg_a

            def fn(x):
                inner()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

    # ------------------------------------------------------------------
    # Exception-handling context. All of these are expected to behave the
    # same as a bare `del`, independent of the surrounding statement.
    # ------------------------------------------------------------------

    def test_delete_missing_global_in_with(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                with contextlib.nullcontext():
                    del _dg_a
                return x + 1

            self._assert_delete_missing_global(fn, "_dg_a", (torch.ones(2, 2),))

    def test_delete_missing_global_exception_attrs(self):
        # `except NameError as e` must be able to inspect the exception.
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError as e:
                    return e.name == "_dg_a"
                return False

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertTrue(opt_fn(x))

    def test_delete_missing_global_caught_no_graph_break(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError:
                    return x + 2
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 2)

    # ------------------------------------------------------------------
    # Cross-module deletes: the inlined callee's globals dict is a real module
    # `__dict__`, so the delete is replayed against the module.
    # ------------------------------------------------------------------

    def test_delete_global_crossfile_inline(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_global_value=True):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_global_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_global_value", mod.__dict__)

    def test_delete_missing_global_crossfile_inline(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_missing_global_value=_ABSENT):

            def fn(x):
                mod.delete_missing_global_value_fn()
                return x + 1

            self._assert_delete_missing_global(
                fn, "delete_missing_global_value", (torch.ones(2, 2),)
            )

    def test_delete_missing_global_crossfile_caught_by_caller(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_missing_global_value=_ABSENT):

            def fn(x):
                try:
                    mod.delete_missing_global_value_fn()
                except NameError:
                    pass
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)

    def test_delete_global_crossfile_created_then_deleted(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_then_delete_missing_value=_ABSENT):
            # The pending store is cancelled by the delete, so nothing may be
            # reported as a replayed side effect.
            with torch._dynamo.config.patch(side_effect_replay_policy="error"):

                @torch.compile(backend="eager", fullgraph=True)
                def fn(x):
                    mod.store_then_delete_missing_fn()
                    return x + 1

                x = torch.ones(2, 2)
                self.assertEqual(fn(x), x + 1)
            self.assertNotIn("store_then_delete_missing_value", mod.__dict__)

    def test_delete_global_crossfile_created_then_deleted_twice(self):
        # Cross-module twin of test_delete_global_created_then_deleted_twice.
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_delete_delete_value=_ABSENT):

            def fn(x):
                mod.store_delete_delete_fn()
                return x + 1

            self._assert_delete_missing_global(
                fn, "store_delete_delete_value", (torch.ones(2, 2),)
            )

    def test_delete_tensor_global_crossfile(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_then_delete_tensor_value=_ABSENT):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.store_then_delete_tensor_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("store_then_delete_tensor_value", mod.__dict__)

    def test_delete_global_crossfile_replay_twice(self):
        # The module dict only loses the name on the first call's replay, so the
        # second call fails the membership guard and recompiles; the now-absent
        # name is observed while tracing rather than replayed.
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_global_value=True):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_global_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_global_value", mod.__dict__)
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                fn(x)

    def test_delete_global_from_module_missing_name_raises_name_error(self):
        # delete_global_from_module is only reached when the name is present
        # under the membership guard, but it must still convert a racing
        # removal into DELETE_GLOBAL's NameError rather than delattr's
        # AttributeError.
        from torch._dynamo.utils import delete_global_from_module

        module = types.ModuleType("_delete_global_missing_name_test")
        with self.assertRaisesRegex(NameError, "name 'gone' is not defined") as cm:
            delete_global_from_module(module, "gone")
        self.assertEqual(cm.exception.name, "gone")

    def test_delete_global_crossfile_caught_presence_flip(self):
        # Cross-module twin of test_delete_global_caught_presence_flip. The
        # module-dict membership guard forces the second call to recompile, so
        # the caller's handler catches the missing name instead of the replayed
        # delete raising outside the traced try.
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_global_value=True):

            def fn(x):
                try:
                    mod.delete_global_value_fn()
                except NameError:
                    return x
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("delete_global_value", mod.__dict__)
            self.assertEqual(opt_fn(x), x)

    def test_delete_missing_global_crossfile_then_store(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_missing_then_store_value=_ABSENT):

            def fn(x):
                mod.delete_missing_then_store_fn()
                return x + 1

            self._assert_delete_missing_global(
                fn, "delete_missing_then_store_value", (torch.ones(2, 2),)
            )
            self.assertNotIn("delete_missing_then_store_value", mod.__dict__)

    def test_delete_global_crossfile_twice(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_twice_value=True):

            def fn(x):
                mod.delete_twice_fn()
                return x + 1

            self._assert_delete_missing_global(
                fn, "delete_twice_value", (torch.ones(2, 2),)
            )
            self.assertNotIn("delete_twice_value", mod.__dict__)

    def test_delete_global_crossfile_then_read(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_global_then_read_value=True):

            def fn(x):
                mod.delete_global_then_read_fn()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                Unsupported, "Attempted to read a deleted variable"
            ):
                opt_fn(x)
            self.assertIn("delete_global_then_read_value", mod.__dict__)

            torch._dynamo.reset()
            with self.assertRaisesRegex(
                NameError, "name 'delete_global_then_read_value' is not defined"
            ):
                torch.compile(fn, backend="eager")(x)
            self.assertNotIn("delete_global_then_read_value", mod.__dict__)

    def test_delete_global_crossfile_store_then_delete_pre_existing(self):
        # The name exists in the callee's module, so this is the branch that
        # records a GLOBAL_DELETE and emits delete_global_from_module.
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_then_delete_value=True):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.store_then_delete_preexisting_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("store_then_delete_value", mod.__dict__)

    @unittest.expectedFailure
    def test_delete_global_crossfile_then_store_moves_key_to_end(self):
        # Cross-module twin of test_delete_global_then_store_moves_key_to_end,
        # sharing its root cause: STORE_GLOBAL overwrites the recorded
        # DeletedVariable/GLOBAL_DELETE, so the delete is never replayed and the
        # name keeps its original slot.
        mod = import_crossfile_mock()
        with temp_globals(
            mod.__dict__, delete_then_store_value=1, crossfile_order_anchor=2
        ):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_then_store_preexisting_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertEqual(mod.delete_then_store_value, 5)
            keys = list(vars(mod))
            self.assertGreater(
                keys.index("delete_then_store_value"),
                keys.index("crossfile_order_anchor"),
            )

    def test_delete_global_crossfile_store_delete_store(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_delete_store_value=True):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.store_delete_store_preexisting_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertEqual(mod.store_delete_store_value, 7)

    def test_delete_two_globals_crossfile(self):
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, delete_two_a_value=1, delete_two_b_value=2):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_two_globals_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_two_a_value", mod.__dict__)
            self.assertNotIn("delete_two_b_value", mod.__dict__)

    def test_store_then_delete_global_crossfile_presence_flip(self):
        # Cross-module twin of test_store_then_delete_global_presence_flip.
        mod = import_crossfile_mock()
        with temp_globals(mod.__dict__, store_then_delete_missing_value=_ABSENT):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.store_then_delete_missing_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("store_then_delete_missing_value", mod.__dict__)

            mod.store_then_delete_missing_value = 1
            self.assertEqual(fn(x), x + 1)
            # Eager deletes it on the second call too.
            self.assertNotIn("store_then_delete_missing_value", mod.__dict__)

    # ------------------------------------------------------------------
    # Recompilation when the global's presence changes between calls
    # ------------------------------------------------------------------

    def test_delete_global_presence_flip(self):
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

            # The name is gone now, so the membership guard forces a recompile;
            # the delete of a then-absent global is an observed exception.
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                opt_fn(x)

            torch._dynamo.reset()
            with self.assertRaisesRegex(NameError, "name '_dg_a' is not defined"):
                torch.compile(fn, backend="eager")(x)

    def test_delete_global_presence_flip_recompiles(self):
        global _dg_a
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError:
                    return x * 2
                return x + 1

            x = torch.ones(2, 2)
            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 1)
            self.assertNotIn("_dg_a", globals())

            # The name is gone, so the DICT_CONTAINS guard fails and the frame
            # recompiles; the traced handler then takes the NameError branch.
            self.assertEqual(opt_fn(x), x * 2)
            self.assertEqual(cnt.frame_count, 2)

    def test_delete_global_absence_flip_recompiles(self):
        global _dg_a
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                _dg_a = 5
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 1)
            self.assertNotIn("_dg_a", globals())

            # The name appears: the DICT_NOT_CONTAINS guard must fail and force a
            # recompile so the delete is replayed rather than dropped.
            _dg_a = 1
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnt.frame_count, 2)
            self.assertNotIn("_dg_a", globals())

    def test_delete_global_caught_presence_flip(self):
        # Here the name is present while tracing, so the delete is recorded and
        # replayed as a suffix DELETE_GLOBAL. The membership guard forces the
        # second call to recompile once the name is gone, so the traced
        # `except NameError` handles it instead of a suffix raise escaping.
        with temp_globals(globals(), _dg_a=1):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError:
                    return x
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

            # Eager catches the NameError and returns x.
            self.assertEqual(opt_fn(x), x)

    def test_store_then_delete_global_presence_flip(self):
        # The store-then-delete pair is nullified while tracing with `_dg_a`
        # absent, but the membership guard forces a recompile once `_dg_a`
        # appears, so the delete is replayed rather than dropped.
        global _dg_a
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                _dg_a = 5
                del _dg_a
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertNotIn("_dg_a", globals())

            _dg_a = 1
            self.assertEqual(opt_fn(x), x + 1)
            # Eager deletes `_dg_a` on both calls.
            self.assertNotIn("_dg_a", globals())

    @unittest.expectedFailure
    def test_delete_global_then_store_moves_key_to_end(self):
        # CPython appends a re-added key at the end of the dict, so `_dg_a` must
        # end up after `_dg_b`. Only one value is recorded per global name, so
        # the store overwrites the DeletedVariable and no DELETE_GLOBAL is
        # replayed, leaving the name in its original slot. Mirroring
        # `store_instance_dict_attr` does not address this: `del obj.x;
        # obj.x = 7` diverges for `obj.__dict__` in exactly the same way.
        with temp_globals(globals(), _dg_a=1, _dg_b=2):

            def fn(x):
                global _dg_a
                del _dg_a
                _dg_a = 7
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(_dg_a, 7)
            keys = list(globals())
            self.assertGreater(keys.index("_dg_a"), keys.index("_dg_b"))

    def test_delete_missing_global_caught_presence_flip(self):
        # Tracing takes the `except NameError` branch because the delete raised
        # while tracing. The DICT_NOT_CONTAINS membership guard forces a
        # recompile once `_dg_a` is defined, so the name is deleted instead of
        # the handler branch being reused.
        #
        # See test_missing_global_load_caught_presence_flip for the LOAD_GLOBAL
        # control, which still has the identical hole.
        global _dg_a
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                global _dg_a
                try:
                    del _dg_a
                except NameError:
                    pass
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)

            _dg_a = 1
            self.assertEqual(opt_fn(x), x + 1)
            # Eager deletes `_dg_a` on the second call.
            self.assertNotIn("_dg_a", globals())

    # ------------------------------------------------------------------
    # Controls pinning the observed-exception convention DELETE_GLOBAL follows
    # ------------------------------------------------------------------

    def test_missing_global_load_unsupported(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                return x + _dg_a

            x = torch.ones(2, 2)
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                torch.compile(fn, backend="eager", fullgraph=True)(x)

    def test_missing_global_load_caught_by_caller(self):
        with temp_globals(globals(), _dg_a=_ABSENT):

            def inner(x):
                return x + _dg_a

            def fn(x):
                try:
                    return inner(x)
                except NameError:
                    return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)

    @unittest.expectedFailure
    def test_missing_global_load_caught_presence_flip(self):
        # Control for `test_delete_missing_global_caught_presence_flip`: the
        # unguarded `except NameError` branch is not specific to DELETE_GLOBAL,
        # it is how dynamo handles any exception observed while tracing.
        global _dg_a
        with temp_globals(globals(), _dg_a=_ABSENT):

            def fn(x):
                try:
                    return x + _dg_a
                except NameError:
                    return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)

            _dg_a = 5
            # Eager takes the `x + _dg_a` branch on the second call.
            self.assertEqual(opt_fn(x), x + 5)

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
            # The tracing failure above happened before `del flag` ran, so the
            # module dict is untouched.
            self.assertEqual(mod.flag, 1)

            compiled = torch.compile(mod.fn, backend="eager")
            self.assertTrue(same(compiled(x), x + 1))
            self.assertNotIn("flag", mod.__dict__)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
