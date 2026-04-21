# Owner(s): ["module: multithreading"]

import weakref

import torch
from torch.testing._internal.common_utils import run_concurrently, run_tests, TestCase


class TestTSan(TestCase):
    def test_storage_thread_safety(self):
        # Concurrent calls to tensor.untyped_storage()
        def access_untyped_storage(tensor):
            return weakref.ref(tensor.untyped_storage())

        for _ in range(10):
            tensor = torch.tensor([1.0, 2.0, 3.0])
            weakrefs = run_concurrently(
                access_untyped_storage, args=(tensor,), num_threads=4
            )
            for wr in weakrefs:
                self.assertEqual(wr(), tensor.untyped_storage())

    def test_concurrent_profiling(self):
        """Repeatedly start/stop profiling while background threads are active.

        On free-threaded Python (3.14t+), this exercises concurrent access to
        the profiler's per-thread state without GIL protection. Without the
        thread-safety fixes (setprofileAllThreads, per-thread ValueCache,
        StopTheWorldGuard), this crashes from heap corruption due to data
        races on the shared hash maps.
        """

        def work():
            for _ in range(100):
                d = {str(i): list(range(i % 10)) for i in range(20)}
                _ = sorted(d.items(), key=lambda x: len(x[1]))
                torch.ones(10) + torch.zeros(10)

        def profile_work():
            for _ in range(30):
                with torch.profiler.profile(with_stack=True, with_modules=True):
                    torch.ones(10)

        run_concurrently([profile_work] + [work] * 8)


if __name__ == "__main__":
    run_tests()
