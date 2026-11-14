# Owner(s): ["module: determinism"]
# Tests for torch.deterministic_policy (thread-local deterministic mode).

import threading
import unittest

import torch
import torch._dynamo
import torch._dynamo.testing
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_CUDA,
    TestCase,
)


def _state():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _max_unpool(x):
    # max_unpool2d alerts as nondeterministic on CPU.
    out, idx = F.max_pool2d(x, 2, return_indices=True)
    return F.max_unpool2d(out, idx, 2)


def _run_in_thread(fn):
    result = {}

    def target():
        result["value"] = fn()

    t = threading.Thread(target=target)
    t.start()
    t.join()
    return result["value"]


class _PolicyTestCase(TestCase):
    def setUp(self):
        super().setUp()
        prior = torch._C._get_deterministic_algorithms_default()
        self.addCleanup(
            lambda: torch._C._set_deterministic_algorithms(prior[0], warn_only=prior[1])
        )
        self.addCleanup(torch._C._set_deterministic_algorithms_override, None)
        torch.use_deterministic_algorithms(False)
        torch._dynamo.reset()

    def _check_backward_policy(self, device):
        seen = []

        class Record(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                seen.append(
                    (threading.current_thread() is threading.main_thread(), _state())
                )
                return grad * 2

        x = torch.randn(4, device=device, requires_grad=True)
        # Forward in a scope, backward outside: the forward's scope applies.
        with torch.deterministic_policy("error"):
            y = Record.apply(x).sum()
        y.backward()
        # Forward outside any scope: the backward call site's policy applies.
        y = Record.apply(x).sum()
        with torch.deterministic_policy("warn"):
            y.backward()
        # The forward's scope wins over the backward call site's.
        with torch.deterministic_policy("default"):
            y = Record.apply(x).sum()
        with torch.deterministic_policy("error"):
            y.backward()
        return seen


class TestDeterministicPolicy(_PolicyTestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_nesting_restores_each_level(self):
        self.assertEqual(_state(), (False, False))
        with torch.deterministic_policy("error"):
            self.assertEqual(_state(), (True, False))
            with torch.deterministic_policy("warn"):
                self.assertEqual(_state(), (True, True))
                with torch.deterministic_policy("default"):
                    self.assertEqual(_state(), (False, False))
                self.assertEqual(_state(), (True, True))
            self.assertEqual(_state(), (True, False))
        self.assertEqual(_state(), (False, False))
        self.assertIsNone(torch._C._get_deterministic_algorithms_override())

    def test_exception_restores(self):
        with self.assertRaisesRegex(ValueError, "boom"):
            with torch.deterministic_policy("error"):
                raise ValueError("boom")
        self.assertEqual(_state(), (False, False))

    def test_local_scope_overrides_global(self):
        torch.use_deterministic_algorithms(True)
        with torch.deterministic_policy("default"):
            self.assertEqual(_state(), (False, False))
            _max_unpool(torch.randn(1, 1, 4, 4))  # no error inside the scope
            # The global setter changes the process default, not the scope.
            torch.use_deterministic_algorithms(True, warn_only=True)
            self.assertEqual(_state(), (False, False))
        self.assertEqual(_state(), (True, True))

    def test_eager_ops_follow_scope(self):
        x = torch.randn(1, 1, 4, 4)
        _max_unpool(x)
        with torch.deterministic_policy("error"):
            with self.assertRaisesRegex(RuntimeError, "max_unpooling2d"):
                _max_unpool(x)
            # Fill-uninitialized-memory also follows the resolved policy.
            self.assertTrue(torch.empty(8).isnan().all())
        with torch.deterministic_policy("warn"):
            with self.assertWarnsRegex(UserWarning, "max_unpooling2d"):
                _max_unpool(x)
        _max_unpool(x)

    def test_thread_isolation(self):
        entered = threading.Event()
        release = threading.Event()
        seen = {}

        def other():
            with torch.deterministic_policy("error"):
                seen["other_in_scope"] = _state()
                entered.set()
                release.wait()

        t = threading.Thread(target=other)
        t.start()
        self.addCleanup(t.join)
        self.addCleanup(release.set)
        entered.wait()
        # The other thread's scope does not leak into this thread.
        self.assertEqual(_state(), (False, False))
        with torch.deterministic_policy("warn"):
            self.assertEqual(_run_in_thread(_state), (False, False))
        release.set()
        t.join()
        self.assertEqual(seen["other_in_scope"], (True, False))

    def test_new_thread_sees_process_default(self):
        torch.use_deterministic_algorithms(True)
        with torch.deterministic_policy("default"):
            self.assertEqual(_run_in_thread(_state), (True, False))

    def test_accumulate_grad_uses_backward_call_site(self):
        # A leaf's accumulator outlives the forward that created it, so it must
        # not keep that forward's scope.
        w = torch.randn(4, requires_grad=True)
        seen = []
        w.register_post_accumulate_grad_hook(lambda p: seen.append(_state()))
        with torch.deterministic_policy("error"):
            loss = (w * 2).sum()
        loss.backward()
        with torch.deterministic_policy("default"):
            loss = (w * 3).sum()  # reuses the accumulator from the first forward
        with torch.deterministic_policy("warn"):
            loss.backward()
        self.assertEqual(seen, [(False, False), (True, True)])

    def test_regenerated_view_grad_fn_uses_inplace_scope(self):
        base = torch.randn(4, 4, requires_grad=True).clone()
        view = base[0]
        with torch.deterministic_policy("error"):
            base.mul_(2)
        seen = []
        # Reading grad_fn outside the scope regenerates the view's node.
        view.grad_fn.register_prehook(lambda grad: seen.append(_state()))
        view.sum().backward()
        self.assertEqual(seen, [(True, False)])

    def test_dynamo_guards_on_resolved_policy(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        x = torch.randn(4)
        fn(x)
        with torch.deterministic_policy("error"):
            fn(x)
            fn(x)
        with torch.deterministic_policy("warn"):
            fn(x)
        fn(x)
        self.assertEqual(cnt.frame_count, 3)

    def test_dynamo_guards_on_override_presence(self):
        # Under an override, the traced global setter does not change the
        # resolved setting, so the graph must not run without an override.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            torch.use_deterministic_algorithms(False)
            return x + (1 if torch.are_deterministic_algorithms_enabled() else 2)

        x = torch.zeros(1)
        with torch.deterministic_policy("error"):
            self.assertEqual(fn(x), x + 1)
        torch.use_deterministic_algorithms(True)
        self.assertEqual(fn(x), x + 2)
        self.assertEqual(cnt.frame_count, 2)

    def test_precompile_workers_inherit_scope(self):
        from torch._inductor.autotune_process import PrecompileThreadPool

        pool = PrecompileThreadPool.get_instance()
        with torch.deterministic_policy("warn"):
            self.assertEqual(pool.submit(_state).result(), (True, True))
        self.assertEqual(pool.submit(_state).result(), (False, False))

    def test_deterministic_guard_does_not_leak_scope(self):
        from torch.testing._internal.common_utils import DeterministicGuard

        with torch.deterministic_policy("error"):
            with DeterministicGuard(False):
                self.assertEqual(_state(), (True, False))
        self.assertEqual(_state(), (False, False))
        with torch.deterministic_policy("error"):
            with DeterministicGuard._current_state():
                pass
        self.assertEqual(_state(), (False, False))

    def test_dynamo_compile_does_not_leak_scope(self):
        @torch.compile(backend="eager")
        def fn(x):
            return x + 1

        with torch.deterministic_policy("error"):
            fn(torch.randn(4))
            self.assertEqual(_state(), (True, False))
        self.assertEqual(_state(), (False, False))
        self.assertEqual(
            torch._C._get_deterministic_algorithms_default(), (False, False)
        )
        self.assertIsNone(torch._C._get_deterministic_algorithms_override())

    def test_scope_inside_compiled_function(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            x = x + 1
            with torch.deterministic_policy("error"):
                y = torch.empty(4)
            return x, y

        x, y = fn(torch.randn(4))
        self.assertTrue(y.isnan().all())
        self.assertEqual(_state(), (False, False))

    def test_inductor_mode_derives_from_scope(self):
        self.assertFalse(inductor_config.deterministic_mode_enabled())
        with torch.deterministic_policy("warn"):
            self.assertTrue(inductor_config.deterministic_mode_enabled())
            self.assertFalse(_run_in_thread(inductor_config.deterministic_mode_enabled))
        # An Inductor-only deterministic mode is independent of the scope.
        with inductor_config.patch(deterministic=True):
            with torch.deterministic_policy("default"):
                self.assertTrue(inductor_config.deterministic_mode_enabled())

    def test_inductor_mode_follows_global_setter_in_every_thread(self):
        torch.use_deterministic_algorithms(True)
        self.assertTrue(_run_in_thread(inductor_config.deterministic_mode_enabled))
        torch.use_deterministic_algorithms(False)
        self.assertFalse(_run_in_thread(inductor_config.deterministic_mode_enabled))

    def test_compiled_backward_outside_forward_scope(self):
        # Backward compiled without determinism rejects a deterministic call.
        fn = torch.compile(lambda x: (x.sin() * 2).sum(), backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        y = fn(x)
        with torch.deterministic_policy("error"):
            with self.assertRaisesRegex(RuntimeError, "compiled backward function"):
                y.backward()
        # Compiled under the scope, backward outside it: allowed.
        with torch.deterministic_policy("error"):
            y = fn(x)
        y.backward()

    def test_lazy_backward_compiles_under_forward_scope(self):
        seen = []

        def record(graph):
            seen.append(inductor_config.deterministic_mode_enabled())

        fn = torch.compile(lambda x: (x.sin() * 2).sum())
        x = torch.randn(4, requires_grad=True)
        with inductor_config.patch(post_grad_custom_post_pass=record):
            with torch.deterministic_policy("error"):
                y = fn(x)
            y.backward()  # the backward graph is compiled here, outside the scope
        self.assertEqual(seen, [True, True])


class TestDeterministicPolicyCPU(_PolicyTestCase):
    hw_classification = HardwareClassification.CPU

    def test_backward_follows_forward_scope(self):
        seen = self._check_backward_policy("cpu")
        # The CPU backward runs on the thread that calls backward().
        self.assertEqual(
            seen, [(True, (True, False)), (True, (True, True)), (True, (False, False))]
        )


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
class TestDeterministicPolicyCUDA(_PolicyTestCase):
    hw_classification = HardwareClassification.CUDA

    def test_backward_follows_forward_scope(self):
        seen = self._check_backward_policy("cuda")
        # The CUDA backward runs on an autograd device thread, which inherits
        # the calling thread's policy through ThreadLocalState.
        self.assertEqual(
            seen,
            [(False, (True, False)), (False, (True, True)), (False, (False, False))],
        )


if __name__ == "__main__":
    run_tests()
