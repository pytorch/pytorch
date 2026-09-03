# Owner(s): ["module: dynamo"]
import sys
import threading

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._C._dynamo.eval_frame import _set_lru_cache
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


def my_custom_function(x):
    return x + 1


class RunDiffGuardTests(torch._dynamo.test_case.TestCase):
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

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_guard_free_entry_does_not_preempt_an_equal_backend(self):
        """Two torch.compile() wrappers over one function carry distinct but
        __eq__-equal backends, which lookup() treats as one backend: the first
        entry whose guards pass wins. The guard-free fast path runs under the
        cache lock, where __eq__ cannot run, so it compares backends by
        identity; an earlier entry with a different object must send it to
        lookup() rather than be skipped for a later guard-free entry."""

        def fn(x, y):
            return x + x.shape[0] + y

        w1 = torch.compile(fn, backend="eager")
        w2 = torch.compile(
            fn,
            backend="eager",
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )
        x3, x4 = torch.zeros(3), torch.zeros(4)
        # Creation order [w1's entry, w2's entry]; hits do not reorder.
        _set_lru_cache(False)
        try:
            w1(x3, 1)
            w2(x4, 1)
            e1, e2 = _debug_get_cache_entry_list(fn)
            self.assertEqual(e1.backend, e2.backend)
            self.assertIsNot(e1.backend, e2.backend)
            self.assertEqual(len(e2.guard_manager.root.get_leaf_guards()), 0)
            self.assertEqual(w2(x3, 1), fn(x3, 1))
            with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                self.assertEqual(w2(x3, 1), fn(x3, 1))
                self.assertEqual(w2(x4, 1), fn(x4, 1))
                self.assertEqual(w1(x3, 1), fn(x3, 1))
                self.assertEqual(w1(x4, 1), fn(x4, 1))
        finally:
            _set_lru_cache(True)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_recompile_keeps_a_diff_guard_root_under_evaluation_alive(self):
        """A recompile rebinds guard_manager.diff_guard_root on every existing
        entry (populate_diff_guard_manager). A thread inside lookup() under
        the skip_guard_eval_unsafe stance evaluates the OLD diff root with the
        cache lock released, so the entry's C++ side must hand that thread a
        reference of its own rather than a raw pointer."""

        def fn(x):
            return x + 1

        opt = torch.compile(fn, backend="eager")
        x3, x4, x5 = torch.zeros(3), torch.zeros(4), torch.zeros(5)
        opt(x3)
        opt(x4)
        entries = sorted(
            _debug_get_cache_entry_list(fn),
            key=lambda e: e.compile_id.frame_compile_id,
        )
        first = entries[0].guard_manager
        old = first.diff_guard_root
        self.assertIsNotNone(old)

        entered, release = threading.Event(), threading.Event()

        def gate(_):
            entered.set()
            release.wait(timeout=60)
            return True

        old.add_lambda_guard(gate, ["gate"], None)
        results = []
        worker = threading.Thread(target=lambda: results.append(opt(x3)))
        try:
            with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                worker.start()
                self.assertTrue(entered.wait(timeout=60))
            opt(x5)
            self.assertIsNot(first.diff_guard_root, old)
            held_while_evaluating = sys.getrefcount(old)
        finally:
            release.set()
            worker.join(timeout=60)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [fn(x3)])
        # Only `old` and getrefcount's argument remain: the evaluating thread
        # held the one extra reference.
        self.assertEqual(sys.getrefcount(old), 2)
        self.assertEqual(held_while_evaluating, 3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
