# Owner(s): ["module: dynamo"]

import contextvars
import threading
from collections.abc import Callable

import torch
import torch._dynamo.eval_frame as eval_frame
import torch._dynamo.test_case
import torch._inductor.config
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


@torch._inductor.config.patch(fx_graph_remote_cache=False)
@torch._dynamo.config.patch(automatic_dynamic_remote_pgo=False)
@instantiate_parametrized_tests
class PythonThreadLocalStateTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def run_with_other_thread_stance(
        self,
        stance: eval_frame.StanceStr,
        body: Callable[[], None],
        *,
        skip_guard_eval_unsafe: bool = False,
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        failures = []
        context = contextvars.copy_context()

        def worker() -> None:
            try:
                with torch.compiler.set_stance(
                    stance, skip_guard_eval_unsafe=skip_guard_eval_unsafe
                ):
                    entered.set()
                    if not release.wait(30):
                        raise AssertionError("caller did not release the stance worker")
            except BaseException as exc:
                failures.append(exc)
                entered.set()

        thread = threading.Thread(target=context.run, args=(worker,), daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(30), "stance worker did not enter")
            if failures:
                raise failures[0]
            body()
        finally:
            release.set()
            thread.join(30)
        self.assertFalse(thread.is_alive(), "stance worker did not exit")
        if failures:
            raise failures[0]

    @parametrize("stance", ["force_eager", "fail_on_recompile", "eager_on_recompile"])
    def test_stance_does_not_change_another_threads_compilation(self, stance) -> None:
        @torch.compile(backend="eager", dynamic=False)
        def fn(x, n):
            if torch.compiler.is_compiling():
                return x + n
            return x + n + 100

        x = torch.ones(4)
        self.assertEqual(fn(x, 1), x + 1)

        def body() -> None:
            self.assertEqual(fn(x, 2), x + 2)

        self.run_with_other_thread_stance(stance, body)
        self.assertEqual(fn(x, 3), x + 3)

    def test_unsafe_stance_does_not_skip_another_threads_guards(self) -> None:
        @torch.compile(backend="eager", dynamic=False)
        def fn(x, n):
            return x + n

        x = torch.ones(4)
        self.assertEqual(fn(x, 1), x + 1)

        def body() -> None:
            self.assertEqual(fn(x, 2), x + 2)

        self.run_with_other_thread_stance("default", body, skip_guard_eval_unsafe=True)

    def test_stance_is_local_to_a_context(self) -> None:
        @torch.compile(backend="eager", dynamic=False)
        def fn(x):
            return x + (1 if torch.compiler.is_compiling() else 100)

        x = torch.ones(4)
        self.assertEqual(fn(x), x + 1)
        with torch.compiler.set_stance("force_eager"):
            self.assertEqual(fn(x), x + 100)
            self.assertEqual(contextvars.Context().run(fn, x), x + 1)
            self.assertEqual(fn(x), x + 100)
        self.assertEqual(fn(x), x + 1)

    def test_copied_stance_inherits_without_leaking_changes(self) -> None:
        @torch.compile(backend="eager", dynamic=False)
        def fn(x):
            return x + (1 if torch.compiler.is_compiling() else 100)

        x = torch.ones(4)
        self.assertEqual(fn(x), x + 1)
        with torch.compiler.set_stance("force_eager"):
            context = contextvars.copy_context()
        self.assertEqual(fn(x), x + 1)

        def body() -> None:
            self.assertEqual(fn(x), x + 100)
            with torch.compiler.set_stance("default"):
                self.assertEqual(fn(x), x + 1)
            self.assertEqual(fn(x), x + 100)

        context.run(body)
        self.assertEqual(fn(x), x + 1)

    def test_concurrent_optimized_module_execution_flags(self) -> None:
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        check_second = threading.Event()
        checked_second = threading.Event()
        tids = {}
        failures = []
        observations = {}

        @torch.compiler.disable
        def gate() -> None:
            tid = tids.get(threading.get_ident())
            if tid is None:
                return
            entered[tid].set()
            if tid == 1:
                if not check_second.wait(30):
                    raise AssertionError("first compiled call did not exit")
                observations["inside_second"] = eval_frame._is_in_optimized_module()
                checked_second.set()
            if not release[tid].wait(30):
                raise AssertionError("compiled call was not released")

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                gate()
                return y

        compiled = torch.compile(Model(), backend="eager", dynamic=False)
        x = torch.ones(4)
        self.assertEqual(compiled(x), x + 1)
        contexts = [contextvars.copy_context() for _ in range(2)]

        def worker(tid: int) -> None:
            try:
                tids[threading.get_ident()] = tid
                self.assertEqual(compiled(x.clone()), x + 1)
            except BaseException as exc:
                failures.append(exc)
                entered[tid].set()
                checked_second.set()

        threads = [
            threading.Thread(target=contexts[tid].run, args=(worker, tid), daemon=True)
            for tid in range(2)
        ]
        try:
            threads[0].start()
            self.assertTrue(entered[0].wait(30), "first compiled call did not enter")
            observations["outside_first"] = eval_frame._is_in_optimized_module()
            threads[1].start()
            self.assertTrue(entered[1].wait(30), "second compiled call did not enter")
            release[0].set()
            threads[0].join(30)
            self.assertFalse(threads[0].is_alive(), "first compiled call did not exit")
            check_second.set()
            self.assertTrue(
                checked_second.wait(30), "second call did not check its flag"
            )
        finally:
            check_second.set()
            for event in release:
                event.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(30)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        if failures:
            raise failures[0]
        self.assertFalse(observations["outside_first"])
        self.assertTrue(observations["inside_second"])
        self.assertFalse(eval_frame._is_in_optimized_module())

    def test_optimized_module_flag_is_nested_and_exception_safe(self) -> None:
        self.assertFalse(eval_frame._is_in_optimized_module())
        with eval_frame._set_in_optimized_module():
            self.assertTrue(eval_frame._is_in_optimized_module())
            with self.assertRaisesRegex(RuntimeError, "nested failure"):
                with eval_frame._set_in_optimized_module():
                    self.assertTrue(eval_frame._is_in_optimized_module())
                    raise RuntimeError("nested failure")
            self.assertTrue(eval_frame._is_in_optimized_module())
        self.assertFalse(eval_frame._is_in_optimized_module())

    def test_new_thread_does_not_inherit_optimized_module_execution(self) -> None:
        flags = []
        with eval_frame._set_in_optimized_module():
            context = contextvars.copy_context()
            thread = threading.Thread(
                target=context.run,
                args=(lambda: flags.append(eval_frame._is_in_optimized_module()),),
                daemon=True,
            )
            thread.start()
            thread.join(30)
            self.assertFalse(thread.is_alive())
            self.assertTrue(eval_frame._is_in_optimized_module())
        self.assertEqual(flags, [False])
        self.assertFalse(eval_frame._is_in_optimized_module())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
