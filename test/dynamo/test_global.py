# Owner(s): ["module: dynamo"]

import contextlib
import importlib.util
import os
import sys
import tempfile

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


@contextlib.contextmanager
def crossfile_globals(**updates):
    """Set the cross-file mock's globals for the test body, then restore them."""
    try:
        from . import mock_store_global_crossfile_inline
    except ImportError:
        import mock_store_global_crossfile_inline

    with temp_globals(mock_store_global_crossfile_inline.__dict__, **updates):
        yield mock_store_global_crossfile_inline


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

    # ------------------------------------------------------------------
    # Exception-handling context. All of these are expected to behave the
    # same as a bare `del`, independent of the surrounding statement.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cross-module deletes: the inlined callee's globals dict is a real module
    # `__dict__`, so the delete is replayed against the module.
    # ------------------------------------------------------------------

    def test_delete_global_crossfile_inline(self):
        with crossfile_globals(delete_value=True) as mod:

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                mod.delete_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            self.assertEqual(fn(x), x + 1)
            self.assertNotIn("delete_value", mod.__dict__)

    def test_delete_global_crossfile_created_then_deleted(self):
        with crossfile_globals(store_then_delete_missing_value=_ABSENT) as mod:
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

    # ------------------------------------------------------------------
    # Controls pinning the observed-exception convention DELETE_GLOBAL follows
    # ------------------------------------------------------------------

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

    def test_delete_global_crossfile_then_store(self):
        # `del g; g = 7` in an inlined callee: the store overwrites the recorded
        # delete, so the store alone is replayed and the value still lands.
        with crossfile_globals(delete_then_store_value=1) as mod:

            def fn(x):
                mod.delete_then_store_value_fn()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(mod.delete_then_store_value, 7)

    def test_delete_global_crossfile_then_store_multi(self):
        # The re-store follows an insert of a second name, so the two recorded
        # mutations replay in insertion order and both values have to land.
        with crossfile_globals(delete_then_store_multi=1) as mod:
            mod.__dict__.pop("delete_then_store_multi_new", None)

            def fn(x):
                mod.delete_then_store_multi_fn()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(mod.delete_then_store_multi, 3)
            self.assertEqual(mod.delete_then_store_multi_new, 2)

    def test_delete_global_crossfile_created_then_deleted_in_stdlib(self):
        # `test_delete_global_crossfile_created_then_deleted` covers the
        # cancelled store on a normal module. This covers the stdlib case, where
        # the source `store_attr` records is wrapped in SkipGuardSource: the
        # frame has to drop that same wrapped source when the delete cancels the
        # store, or it stays in mutated_sources and reports the module as
        # mutated for the lifetime of the compiled artifact.
        import collections.abc

        from torch._dynamo.symbolic_convert import InstructionTranslator

        name = "stdlib_store_then_delete"
        exec(
            compile(
                f"""
def fn():
    global {name}
    {name} = 1
    del {name}
    return 0
""",
                "<stdlib-globals>",
                "exec",
            ),
            collections.abc.__dict__,
        )
        stdlib_fn = collections.abc.__dict__["fn"]
        self.assertNotIn(name, collections.abc.__dict__)

        mutated_sources = {}
        orig_run = InstructionTranslator.run

        def run(self, *args, **kwargs):
            try:
                return orig_run(self, *args, **kwargs)
            finally:
                mutated_sources["value"] = set(self.output.side_effects.mutated_sources)

        InstructionTranslator.run = run
        try:

            def fn(x):
                stdlib_fn()
                return x + 1

            x = torch.ones(2, 2)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(opt_fn(x), x + 1)
        finally:
            InstructionTranslator.run = orig_run
            del collections.abc.__dict__["fn"]

        self.assertFalse(
            [s for s in mutated_sources["value"] if name in repr(s)],
            "the cancelled store left its source in mutated_sources",
        )
        self.assertNotIn(name, collections.abc.__dict__)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
