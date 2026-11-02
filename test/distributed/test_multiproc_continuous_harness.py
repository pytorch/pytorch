# Owner(s): ["oncall: distributed"]

"""Checks how MultiProcContinuousTest combines rank results and shuts down."""

import multiprocessing
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import torch.testing._internal.common_distributed as cd
from torch.testing._internal.common_utils import run_tests, TestCase


# Built in a function so unittest doesn't collect and spawn them.
def _classes():
    class Fake(cd.MultiProcContinuousTest):
        def test_x(self):
            pass

    class Hung(cd.MultiProcContinuousTest):
        pass

    return Fake, Hung


def _lifecycle_class(events, errors):
    class Worker(cd.MultiProcContinuousTest):
        rank = 0
        world_size = 2

        @classmethod
        def _ensure_processes_spawned(cls):
            raise AssertionError("worker must not spawn processes")

        def _step(self, name):
            events.append(name)
            if name in errors:
                raise errors[name]

        def setUp(self):
            super().setUp()
            self.addCleanup(self._step, "cleanup_first")
            self.addCleanup(self._step, "cleanup_last")
            self._step("setup")

        def test_x(self):
            self._step("test")

        def tearDown(self):
            super().tearDown()
            self._step("teardown")

    return Worker


def _subtest_class(events, error, expected=False, fixture=None):
    class Worker(cd.MultiProcContinuousTest):
        rank = 0
        world_size = 2

        def nested_exit(self):
            with self.subTest(fixture=fixture):
                raise error

        def setUp(self):
            super().setUp()
            self.addCleanup(events.append, "cleanup")
            if fixture == "cleanup":
                self.addCleanup(self.nested_exit)
            events.append("setup")
            if fixture == "setup":
                self.nested_exit()

        def test_x(self):
            if fixture is not None:
                events.append("test")
                return
            for i in range(3):
                with self.subTest(i=i):
                    events.append(f"body{i}")
                    if i == 0:
                        raise error
                cd.c10d.barrier()

        def tearDown(self):
            events.append("teardown")
            if fixture == "teardown":
                self.nested_exit()
            super().tearDown()

    if expected:
        Worker.test_x = unittest.expectedFailure(Worker.test_x)
    return Worker


def _cpu_worker_class(events):
    class Worker(cd.MultiProcContinuousTest):
        world_size = 2

        @classmethod
        def backend_str(cls):
            return "gloo"

        @classmethod
        def device_type(cls):
            return "cpu"

        def _step(self, phase):
            events.put((self.rank, self._testMethodName, phase))
            if self._testMethodName == f"test_{phase}_exit":
                raise SystemExit(cd.TEST_SKIPS["no_accelerator"].exit_code)
            if self._testMethodName == f"test_{phase}_subtest_exit":
                with self.subTest(phase=phase):
                    raise SystemExit(cd.TEST_SKIPS["no_accelerator"].exit_code)
            if self._testMethodName in (
                f"test_{phase}_failure",
                f"test_xfail_{phase}_failure",
            ):
                raise ValueError(phase)
            if phase == "cleanup" and self._testMethodName in (
                "test_skip_cleanup_failure",
                "test_xfail_cleanup_failure",
            ):
                raise ValueError("cleanup")

        def setUp(self):
            super().setUp()
            if self.rank == self.MAIN_PROCESS_RANK:
                return
            self.addCleanup(self._step, "cleanup")
            self._step("setup")

        def tearDown(self):
            super().tearDown()
            if self.rank != self.MAIN_PROCESS_RANK:
                self._step("teardown")

        def test_pass(self):
            self._step("test")
            events.put((self.rank, "rng", cd.torch.rand(1, device="cpu").item()))
            cd.torch.manual_seed(999 + self.rank)

        def test_setup_failure(self):
            self.fail("setup should have failed")

        def test_test_failure(self):
            self._step("test")

        def test_teardown_failure(self):
            self._step("test")

        def test_cleanup_failure(self):
            self._step("test")

        def test_setup_exit(self):
            self.fail("setup should have failed")

        def test_teardown_exit(self):
            self._step("test")

        def test_cleanup_exit(self):
            self._step("test")

        def test_setup_subtest_exit(self):
            self.fail("setup should have failed")

        def test_teardown_subtest_exit(self):
            self._step("test")

        def test_cleanup_subtest_exit(self):
            self._step("test")

        def test_subtest_assertion_failure(self):
            self._subtest_failure(AssertionError("subtest"))

        def test_subtest_error(self):
            self._subtest_failure(ValueError("subtest"))

        def _subtest_failure(self, error):
            self._step("test")
            for i in range(3):
                with self.subTest(i=i):
                    self._step(f"body{i}")
                    if i == 0:
                        raise error
                self._step("post_subtest_barrier")
                cd.c10d.barrier()

        def test_subtest_exit(self):
            self._step("test")
            with self.subTest(exit=True):
                raise SystemExit(cd.TEST_SKIPS["no_accelerator"].exit_code)
            self._step("post_subtest")

        @unittest.expectedFailure
        def test_xfail_subtest_exit(self):
            self.test_subtest_exit()

        def test_skip(self):
            self._step("test")
            self.skipTest("skip")

        def test_skip_cleanup_failure(self):
            self.test_skip()

        @unittest.expectedFailure
        def test_xfail(self):
            self._step("test")
            self.fail("expected")

        @unittest.expectedFailure
        def test_xpass(self):
            self._step("test")

        @unittest.expectedFailure
        def test_xfail_setup_failure(self):
            self.fail("setup should have failed")

        @unittest.expectedFailure
        def test_xfail_teardown_failure(self):
            self.test_xfail()

        @unittest.expectedFailure
        def test_xfail_cleanup_failure(self):
            self.test_xfail()

        @unittest.expectedFailure
        def test_xfail_backend_skip(self):
            self._step("test")
            raise SystemExit(cd.TEST_SKIPS["no_accelerator"].exit_code)

    return Worker


def _cpu_worker(rank, rdvz_file, tasks, completions, events):
    _cpu_worker_class(events)._worker_loop(rank, 2, rdvz_file, tasks, completions)


def _sleep():
    time.sleep(1000)


class TestMultiProcContinuousHarness(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.Fake, cls.Hung = _classes()

    def _wait(self, results):
        self.Fake.poison_pill = False
        t = self.Fake("test_x")
        t.rank = t.MAIN_PROCESS_RANK
        t.processes = [None] * len(results)
        t.completion_queues = [None] * len(results)
        it = iter(t.id() if r == "ok" else r for r in results)
        with patch.object(
            cd, "retrieve_result_from_completion_queue", lambda *a, **k: next(it)
        ):
            t.test_x()
        # Every queue must be drained so the next test sees its own results.
        self.assertIsNone(next(it, None))

    def test_failure_beats_skip(self):
        for results in (
            [unittest.SkipTest("s"), ValueError("v")],
            [ValueError("v"), unittest.SkipTest("s")],
        ):
            with self.assertRaises(ValueError):
                self._wait(results)
            self.assertTrue(self.Fake.poison_pill)

    def test_wrong_result_is_failure(self):
        with self.assertRaisesRegex(AssertionError, "Expected rv"):
            self._wait(["other", unittest.SkipTest("s")])
        self.assertTrue(self.Fake.poison_pill)

    def test_skip(self):
        with self.assertRaises(unittest.SkipTest):
            self._wait([unittest.SkipTest("s"), "ok"])
        self.assertFalse(self.Fake.poison_pill)

    def test_pass(self):
        self._wait(["ok", "ok"])
        self.assertFalse(self.Fake.poison_pill)

    def test_bootstrap_skip_does_not_destroy_uninitialized_pg(self):
        tasks, completions = queue.Queue(), queue.Queue()
        for name in ("test_x", "test_y", None):
            tasks.put(name)
        skip = cd.TEST_SKIPS["no_accelerator"]
        worker, _ = _classes()
        with (
            patch.object(worker, "_init_pg", side_effect=SystemExit(skip.exit_code)),
            patch.object(worker, "_run_test_given_id") as run,
            patch.object(worker, "setUp") as setup,
            patch.object(worker, "tearDown") as teardown,
            patch.object(worker, "doCleanups") as cleanups,
            patch.object(cd.c10d, "destroy_process_group") as destroy,
        ):
            self.assertIsNone(worker._worker_loop(0, 2, "unused", tasks, completions))
        for _ in range(2):
            result = completions.get_nowait()
            self.assertIsInstance(result, unittest.SkipTest)
            self.assertEqual(str(result), skip.message)
        self.assertTrue(completions.empty())
        run.assert_not_called()
        setup.assert_not_called()
        teardown.assert_not_called()
        cleanups.assert_not_called()
        destroy.assert_not_called()

    def _run_lifecycle(self, errors):
        events = []
        worker = _lifecycle_class(events, errors)
        with patch.object(TestCase, "setUp"), patch.object(TestCase, "tearDown"):
            worker._run_test_given_id("test_x")
        return events

    def test_worker_lifecycle(self):
        self.assertEqual(
            self._run_lifecycle({}),
            ["setup", "test", "teardown", "cleanup_last", "cleanup_first"],
        )

    def test_worker_lifecycle_failures(self):
        for phase in ("setup", "test", "teardown", "cleanup_last"):
            with self.subTest(phase=phase):
                events = []
                error = ValueError(phase)
                worker = _lifecycle_class(events, {phase: error})
                with (
                    patch.object(TestCase, "setUp"),
                    patch.object(TestCase, "tearDown"),
                ):
                    with self.assertRaises(ValueError) as raised:
                        worker._run_test_given_id("test_x")
                self.assertIs(raised.exception, error)
                expected = [
                    "setup",
                    "test",
                    "teardown",
                    "cleanup_last",
                    "cleanup_first",
                ]
                if phase == "setup":
                    expected = ["setup", "cleanup_last", "cleanup_first"]
                self.assertEqual(events, expected)

    def test_worker_cleanup_failure_beats_skip(self):
        for skip in (
            unittest.SkipTest("skip"),
            SystemExit(cd.TEST_SKIPS["no_accelerator"].exit_code),
        ):
            with self.subTest(skip=type(skip).__name__):
                with self.assertRaisesRegex(ValueError, "cleanup"):
                    errors = {"test": skip, "cleanup_last": ValueError("cleanup")}
                    self._run_lifecycle(errors)

    def test_worker_fixture_exit_is_not_skip(self):
        code = cd.TEST_SKIPS["no_accelerator"].exit_code
        for phase in ("setup", "teardown", "cleanup_last"):
            with self.subTest(phase=phase):
                with self.assertRaises(SystemExit) as raised:
                    self._run_lifecycle({phase: SystemExit(code)})
                self.assertEqual(raised.exception.code, code)

    def test_worker_skip_runs_teardown_and_cleanups(self):
        with self.assertRaisesRegex(unittest.SkipTest, "skip"):
            self._run_lifecycle({"test": unittest.SkipTest("skip")})

    def test_worker_preserves_first_failure(self):
        errors = {"test": ValueError("test"), "cleanup_last": RuntimeError("cleanup")}
        with self.assertRaisesRegex(ValueError, "test"):
            self._run_lifecycle(errors)

    def test_worker_rng_reset(self):
        with patch.object(cd.common_utils, "set_rng_seed") as seed:
            self._run_lifecycle({})
            self._run_lifecycle({})
        self.assertEqual(seed.call_count, 2)

    def test_worker_subtest_failfast(self):
        for error in (AssertionError("subtest"), ValueError("subtest")):
            with self.subTest(error=type(error).__name__):
                events = []
                worker = _subtest_class(events, error)
                with (
                    patch.object(TestCase, "setUp"),
                    patch.object(TestCase, "tearDown"),
                    patch.object(cd.c10d, "barrier") as barrier,
                ):
                    with self.assertRaises(type(error)):
                        worker._run_test_given_id("test_x")
                barrier.assert_not_called()
                self.assertEqual(events, ["setup", "body0", "teardown", "cleanup"])

    def test_worker_nested_body_exit_is_skip(self):
        code = cd.TEST_SKIPS["no_accelerator"].exit_code
        for expected in (False, True):
            with self.subTest(expected=expected):
                events = []
                worker = _subtest_class(events, SystemExit(code), expected=expected)
                with (
                    patch.object(TestCase, "setUp"),
                    patch.object(TestCase, "tearDown"),
                    patch.object(cd.c10d, "barrier") as barrier,
                ):
                    with self.assertRaises(unittest.SkipTest):
                        worker._run_test_given_id("test_x")
                barrier.assert_not_called()
                self.assertEqual(events, ["setup", "body0", "teardown", "cleanup"])

    def test_worker_nested_fixture_exit_is_error(self):
        code = cd.TEST_SKIPS["no_accelerator"].exit_code
        for fixture in ("setup", "teardown", "cleanup"):
            with self.subTest(fixture=fixture):
                events = []
                worker = _subtest_class(events, SystemExit(code), fixture=fixture)
                with (
                    patch.object(TestCase, "setUp"),
                    patch.object(TestCase, "tearDown"),
                ):
                    with self.assertRaises(SystemExit):
                        worker._run_test_given_id("test_x")
                phases = ["setup", "test", "teardown", "cleanup"]
                if fixture == "setup":
                    phases = ["setup", "cleanup"]
                self.assertEqual(events, phases)

    def _run_cpu_sequence(self, names):
        if not cd.c10d.is_gloo_available():
            self.skipTest("Gloo is not available")
        ctx = multiprocessing.get_context("spawn")
        events = ctx.Queue()
        worker = _cpu_worker_class(events)
        worker.rank = worker.MAIN_PROCESS_RANK
        worker.poison_pill = False
        worker._processes_spawned = True
        worker.task_queues = [ctx.Queue() for _ in range(2)]
        worker.completion_queues = [ctx.Queue() for _ in range(2)]
        with tempfile.TemporaryDirectory() as rdvz:
            worker.rdvz_file = f"{rdvz}/store"
            worker.processes = [
                ctx.Process(
                    target=_cpu_worker,
                    args=(rank, worker.rdvz_file, tasks, completions, events),
                    daemon=True,
                )
                for rank, (tasks, completions) in enumerate(
                    zip(worker.task_queues, worker.completion_queues)
                )
            ]
            try:
                for process in worker.processes:
                    process.start()
                result = unittest.TestResult()
                unittest.TestSuite(worker(name) for name in names).run(result)
            finally:
                for process, tasks in zip(worker.processes, worker.task_queues):
                    if process.pid is not None and process.is_alive():
                        tasks.put(None)
                deadline = time.monotonic() + 30
                hung = []
                for rank, process in enumerate(worker.processes):
                    if process.pid is None:
                        process.close()
                        continue
                    process.join(max(0, deadline - time.monotonic()))
                    if process.is_alive():
                        hung.append(rank)
                        process.kill()
                        process.join()
                    process.close()
                for q in (*worker.task_queues, *worker.completion_queues):
                    q.close()
                    q.join_thread()
                self.assertEqual(hung, [], f"CPU workers did not stop: {hung}")
        recorded = []
        while True:
            try:
                recorded.append(events.get_nowait())
            except queue.Empty:
                break
        events.close()
        events.join_thread()
        return result, recorded, worker.poison_pill

    def test_cpu_workers_reuse_after_skip_and_xfail(self):
        result, events, poisoned = self._run_cpu_sequence(
            [
                "test_pass",
                "test_skip",
                "test_xfail",
                "test_xfail_backend_skip",
                "test_pass",
            ]
        )
        self.assertTrue(result.wasSuccessful(), (result.errors, result.failures))
        self.assertEqual(result.testsRun, 5)
        self.assertEqual(len(result.expectedFailures), 1)
        self.assertEqual(len(result.skipped), 2)
        self.assertFalse(poisoned)
        self.assertEqual(len(events), 44)
        for rank in range(2):
            self.assertEqual(sum(r == rank for r, _, _ in events), 22)
            rng = [value for r, phase, value in events if r == rank and phase == "rng"]
            self.assertEqual(len(rng), 2)
            self.assertEqual(rng[0], rng[1])
            for name in ("test_skip", "test_xfail"):
                phases = [p for r, test, p in events if r == rank and test == name]
                self.assertEqual(phases, ["setup", "test", "teardown", "cleanup"])

    def test_cpu_workers_xpass_is_failure(self):
        result, events, poisoned = self._run_cpu_sequence(["test_xpass", "test_pass"])
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.unexpectedSuccesses), 1)
        self.assertEqual(result.testsRun, 2)
        self.assertFalse(poisoned)
        self.assertEqual(len(events), 18)

    def test_cpu_worker_failures_poison_following_tests(self):
        for phase in (
            "setup",
            "test",
            "teardown",
            "cleanup",
            "skip_cleanup",
            "xfail_setup",
            "xfail_teardown",
            "xfail_cleanup",
        ):
            with self.subTest(phase=phase):
                name = f"test_{phase}_failure"
                result, events, poisoned = self._run_cpu_sequence([name, "test_pass"])
                self.assertEqual(len(result.errors), 1, result.errors)
                self.assertEqual(len(result.expectedFailures), 0)
                self.assertEqual(len(result.skipped), 1)
                self.assertTrue(poisoned)
                for rank in range(2):
                    phases = [p for r, test, p in events if r == rank and test == name]
                    expected = ["setup", "test", "teardown", "cleanup"]
                    if phase in ("setup", "xfail_setup"):
                        expected = ["setup", "cleanup"]
                    self.assertEqual(phases, expected)
                count = 4 if phase in ("setup", "xfail_setup") else 8
                self.assertEqual(len(events), count)

    def test_cpu_worker_fixture_exit_poison_following_tests(self):
        for phase in ("setup", "teardown", "cleanup"):
            with self.subTest(phase=phase):
                name = f"test_{phase}_exit"
                result, events, poisoned = self._run_cpu_sequence([name, "test_pass"])
                self.assertEqual(len(result.errors), 1, result.errors)
                self.assertEqual(len(result.skipped), 1)
                self.assertTrue(poisoned)
                self.assertEqual(len(events), 4 if phase == "setup" else 8)

    def test_cpu_workers_subtest_failfast(self):
        for name in ("test_subtest_assertion_failure", "test_subtest_error"):
            with self.subTest(name=name):
                result, events, poisoned = self._run_cpu_sequence([name, "test_pass"])
                self.assertEqual(len(result.errors), 1, result.errors)
                self.assertEqual(len(result.skipped), 1)
                self.assertTrue(poisoned)
                self.assertEqual(len(events), 10)
                for rank in range(2):
                    phases = [p for r, test, p in events if r == rank and test == name]
                    self.assertEqual(
                        phases, ["setup", "test", "body0", "teardown", "cleanup"]
                    )

    def test_cpu_workers_nested_body_exit_is_skip(self):
        result, events, poisoned = self._run_cpu_sequence(
            ["test_subtest_exit", "test_xfail_subtest_exit", "test_pass"]
        )
        self.assertTrue(result.wasSuccessful(), (result.errors, result.failures))
        self.assertEqual(result.testsRun, 3)
        self.assertEqual(len(result.skipped), 2)
        self.assertEqual(len(result.expectedFailures), 0)
        self.assertFalse(poisoned)
        self.assertEqual(len(events), 26)
        for rank in range(2):
            for name in ("test_subtest_exit", "test_xfail_subtest_exit"):
                phases = [p for r, test, p in events if r == rank and test == name]
                self.assertEqual(phases, ["setup", "test", "teardown", "cleanup"])

    def test_cpu_worker_nested_fixture_exit_is_error(self):
        for phase in ("setup", "teardown", "cleanup"):
            with self.subTest(phase=phase):
                name = f"test_{phase}_subtest_exit"
                result, events, poisoned = self._run_cpu_sequence([name, "test_pass"])
                self.assertEqual(len(result.errors), 1, result.errors)
                self.assertEqual(len(result.skipped), 1)
                self.assertTrue(poisoned)
                self.assertEqual(len(events), 4 if phase == "setup" else 8)

    def test_teardown_kills_hung_ranks(self):
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_sleep, daemon=True) for _ in range(3)]
        for p in procs:
            p.start()
        self.Hung._processes_spawned = True
        self.Hung.processes = procs
        self.Hung.task_queues = []
        self.Hung.rdvz_file = "/nonexistent"
        start = time.monotonic()
        with patch.object(cd, "TIMEOUT_DEFAULT", 2):
            with self.assertRaisesRegex(
                RuntimeError, r"ranks \[0, 1, 2\] did not exit"
            ):
                self.Hung.tearDownClass()
        # One shared deadline, not one per rank.
        self.assertLess(time.monotonic() - start, 5)
        for p in procs:
            self.assertFalse(p.is_alive())


if __name__ == "__main__":
    run_tests()
