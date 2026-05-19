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

    def test_concurrent_grad_fn(self):
        # Concurrent access to tensor.grad_fn exercises the atomic
        # PyObjectSlot on Node that lazily creates the Python wrapper.
        def access_grad_fn(tensor):
            return tensor.grad_fn

        for _ in range(10):
            x = torch.randn(4, requires_grad=True)
            y = x * 2 + 1
            grad_fns = run_concurrently(access_grad_fn, args=(y,), num_threads=4)
            for gf in grad_fns:
                self.assertIs(gf, y.grad_fn)

    def test_concurrent_dict_recursive_tag_watcher(self):
        """Race the dict recursive-tag watcher callback against guard checks.

        On free-threaded Python (3.12+), a tag-safe root's CPython dict
        watcher can fire on one thread (e.g. due to nn.Module attribute
        mutation) while another thread is inside RootGuardManager.check and
        iterating the same per-GuardManager _dict_pointers map. Without the
        deferred-cleanup fix, that's UB on std::unordered_map plus a torn
        read on the disable flag.
        """

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 0

            def forward(self, x):
                return x + self.a

        mod = Mod()
        x = torch.randn(4, 4)

        def fn(inp):
            return mod(inp)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # Warm up: first call compiles and installs the tag-safe root plus
        # the dict watcher on mod.__dict__.
        opt_fn(x)

        def mutate():
            for i in range(200):
                # Fires dict_recursive_tag_watch_callback on mod.__dict__.
                mod.a = i

        def call_opt_fn():
            for _ in range(200):
                opt_fn(x)

        run_concurrently([call_opt_fn, mutate])


if __name__ == "__main__":
    run_tests()
