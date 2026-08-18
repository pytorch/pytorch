# Owner(s): ["module: dynamo"]

import queue
import threading

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


def my_custom_function(x):
    return x + 1


class RunDiffGuardTests(torch._dynamo.test_case.TestCase):
    def test_concurrent_diff_guard_replacement_rejects_stale_snapshot(self):
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        from torch._dynamo.guards import RootGuardManager

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        self.assertEqual(compiled(x), fn(x))
        guard_manager = _debug_get_cache_entry_list(fn)[0].guard_manager

        guard_entered = threading.Event()
        release_guard = threading.Event()

        def blocking_guard(_locals):
            guard_entered.set()
            self.assertTrue(release_guard.wait(10))
            return True

        old_root = RootGuardManager()
        old_root.add_lambda_guard(blocking_guard, [], None)
        guard_manager.cache_entry.update_diff_guard_root_manager(old_root)
        guard_manager.diff_guard_root = old_root

        errors = queue.SimpleQueue()

        def run_compiled():
            try:
                with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                    compiled(x)
            except BaseException as error:
                errors.put(error)

        worker = threading.Thread(target=run_compiled, daemon=True)
        worker.start()
        self.assertTrue(guard_entered.wait(10))
        new_root = RootGuardManager()
        new_root.add_lambda_guard(lambda _locals: False, [], None)
        guard_manager.cache_entry.update_diff_guard_root_manager(new_root)
        guard_manager.diff_guard_root = new_root
        release_guard.set()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        raised = errors.get_nowait()
        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("Recompilation triggered", str(raised))

    def test_concurrent_false_diff_guard_replacement_rejects_stale_snapshot(self):
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        from torch._dynamo.guards import RootGuardManager

        def fn(x, flag):
            return x + 1 if flag else x - 1

        compiled = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        self.assertEqual(compiled(x, False), fn(x, False))
        self.assertEqual(compiled(x, True), fn(x, True))
        entries = _debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 2)
        first = entries[0].guard_manager
        second = entries[1].guard_manager

        guard_entered = threading.Event()
        release_guard = threading.Event()

        def blocking_false_guard(_locals):
            guard_entered.set()
            self.assertTrue(release_guard.wait(10))
            return False

        old_root = RootGuardManager()
        old_root.add_lambda_guard(blocking_false_guard, [], None)
        first.cache_entry.update_diff_guard_root_manager(old_root)
        first.diff_guard_root = old_root
        second_root = RootGuardManager()
        second_root.add_lambda_guard(lambda _locals: True, [], None)
        second.cache_entry.update_diff_guard_root_manager(second_root)
        second.diff_guard_root = second_root

        errors = queue.SimpleQueue()

        def run_compiled():
            try:
                with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                    compiled(x, True)
            except BaseException as error:
                errors.put(error)

        worker = threading.Thread(target=run_compiled, daemon=True)
        worker.start()
        self.assertTrue(guard_entered.wait(10))
        new_root = RootGuardManager()
        new_root.add_lambda_guard(lambda _locals: True, [], None)
        first.cache_entry.update_diff_guard_root_manager(new_root)
        first.diff_guard_root = new_root
        release_guard.set()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        raised = errors.get_nowait()
        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("Recompilation triggered", str(raised))

    def test_bool_recompile(self):
        def fn(x, y, c):
            if c:
                return x * y
            else:
                return x + y

        opt_fn = torch.compile(fn, backend="inductor")
        x = 2 * torch.ones(4)
        y = 3 * torch.ones(4)

        ref1 = opt_fn(x, y, True)
        ref2 = opt_fn(x, y, False)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res2 = opt_fn(x, y, False)
            res1 = opt_fn(x, y, True)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_tensor_recompile(self):
        def fn(x, y):
            return x * y

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4, dtype=torch.float32)
        y = torch.randn(4, dtype=torch.float32)

        ref1 = opt_fn(x, y)

        x64 = torch.randn(4, dtype=torch.float64)
        y64 = torch.randn(4, dtype=torch.float64)
        ref2 = opt_fn(x64, y64)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, y)
            res2 = opt_fn(x64, y64)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_post_recompile(self):
        class Foo:
            def __init__(self):
                self.a = 4
                self.b = 5

        foo = Foo()

        def fn(x):
            return x + foo.a + foo.b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        foo.a = 11
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            # Set it back to original value
            foo.a = 4
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

            foo.a = 11
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

        # Check that we are back to original behavior
        foo.b = 8
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)

    def test_fail_on_tensor_shape_change(self):
        def fn(dt):
            return dt["x"] + 1

        x = torch.randn(4)
        dt = {}
        dt["x"] = x
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(dt)

        with self.assertRaisesRegex(
            RuntimeError, "Recompilation triggered with skip_guard_eval_unsafe stance"
        ):
            with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                x = torch.randn(4, 4)
                dt["x"] = x
                opt_fn(dt)

    def test_cache_line_pickup(self):
        def fn(x, a=None, b=None):
            x = x * 3
            if a:
                x = x * 5
            if b:
                x = x * 7
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(4)

        ref1 = opt_fn(x, a=None, b=None)
        ref2 = opt_fn(x, a=1, b=None)
        ref3 = opt_fn(x, a=1, b=1)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, a=None, b=None)
            res2 = opt_fn(x, a=1, b=None)
            res3 = opt_fn(x, a=1, b=1)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)

    def test_skip_all_guards_single_cache_entry(self):
        def fn(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(
            fn,
            backend=cnts,
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )

        x = torch.randn(4)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnts.frame_count, 1)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        root = entries[0].guard_manager.root
        self.assertEqual(len(root.get_leaf_guards()), 0)
        self.assertEqual(len(root.get_accessors()), 0)
        self.assertEqual(len(root.get_epilogue_lambda_guards()), 0)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            y = torch.randn(4, 4)
            self.assertEqual(opt_fn(y), fn(y))

        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
