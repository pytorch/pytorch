# Owner(s): ["module: inductor"]
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime import flydsl_cache
from torch.testing._internal.common_utils import run_tests, TestCase


class _TensorArgument:
    def __init__(self, tensor):
        self.tensor = tensor

    def __cache_signature__(self):
        return (self.tensor.dtype,)


class _ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiter_entered = threading.Event()

    def __enter__(self):
        if self.lock.locked():
            self.waiter_entered.set()
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock.release()


class TestFlyDSLCache(TestCase):
    def setUp(self):
        super().setUp()
        self.launcher = SimpleNamespace()
        self.dispatch = mock.Mock(side_effect=lambda output: output.add_(1))
        self.compiler = mock.Mock(side_effect=self._compile)

    def _compile(self, launcher, argument):
        self.assertIs(launcher, self.launcher)
        # FlyDSL compile executes the invocation before returning its dispatcher.
        self.dispatch(argument.tensor)
        return self.dispatch

    def _run(self, output):
        argument = _TensorArgument(output)
        return flydsl_cache.run_cached_flydsl(
            self.launcher,
            argument,
            constexpr_param=argument,
            compiler=self.compiler,
            dispatch_args=(output,),
        )

    def test_cold_and_warm_calls_execute_once(self):
        cold_output = torch.zeros((), dtype=torch.int64, device="cpu")
        warm_output = torch.zeros_like(cold_output)

        self.assertIs(self._run(cold_output), self.dispatch)
        self.assertEqual(cold_output, 1)
        self.assertIs(self._run(warm_output), self.dispatch)

        self.assertEqual(cold_output, 1)
        self.assertEqual(warm_output, 1)
        self.assertEqual(self.compiler.call_count, 1)
        self.assertEqual(self.dispatch.call_count, 2)

    def test_compile_failure_can_be_retried(self):
        output = torch.zeros((), dtype=torch.int64, device="cpu")
        self.compiler.side_effect = RuntimeError("FlyDSL compilation failed")

        with self.assertRaisesRegex(RuntimeError, "FlyDSL compilation failed"):
            self._run(output)
        self.assertEqual(output, 0)
        self.dispatch.assert_not_called()

        self.compiler.side_effect = self._compile
        self.assertIs(self._run(output), self.dispatch)
        self.assertEqual(output, 1)
        self.assertIs(self._run(output), self.dispatch)

        self.assertEqual(output, 2)
        self.assertEqual(self.compiler.call_count, 2)
        self.assertEqual(self.dispatch.call_count, 2)

    def test_concurrent_cold_calls_execute_once_each(self):
        first_output = torch.zeros((), dtype=torch.int64, device="cpu")
        second_output = torch.zeros_like(first_output)
        compile_started = threading.Event()
        release_compile = threading.Event()
        lock = _ObservedLock()

        def compile_while_waiting(launcher, argument):
            compile_started.set()
            if not release_compile.wait(timeout=10):
                raise RuntimeError("Timed out waiting to finish compilation")
            return self._compile(launcher, argument)

        self.compiler.side_effect = compile_while_waiting
        with (
            mock.patch.object(flydsl_cache, "_compiled_cache_lock", lock),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first = pool.submit(self._run, first_output)
            try:
                self.assertTrue(compile_started.wait(timeout=10))
                second = pool.submit(self._run, second_output)
                self.assertTrue(lock.waiter_entered.wait(timeout=10))
            finally:
                release_compile.set()

            self.assertIs(first.result(timeout=10), self.dispatch)
            self.assertIs(second.result(timeout=10), self.dispatch)

        self.assertEqual(first_output, 1)
        self.assertEqual(second_output, 1)
        self.assertEqual(self.compiler.call_count, 1)
        self.assertEqual(self.dispatch.call_count, 2)


if __name__ == "__main__":
    run_tests()
