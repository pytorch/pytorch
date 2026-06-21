# Owner(s): ["module: multithreading"]

import weakref
from functools import partial

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

    def test_concurrent_config_flag_toggles(self):
        """Concurrently set/read process-global bool config flags.

        warn_always, the vmap fallback enable/warning flags, and (on CUDA
        builds) the cudnn conv benchmark empty-cache flag are process-global
        toggles accessed through dedicated setters/getters. They used to be
        plain ``static bool``, so concurrent access from multiple threads was a
        data race -- undefined behavior under the C++ memory model and reported
        by TSan even though no real platform tears a bool load. They are now
        ``static constinit std::atomic<bool>`` with relaxed ordering; this
        hammers each accessor from several threads to guard against regressing.
        """
        functorch = torch._C._functorch

        # (setter, getter): getter is None when no read accessor is exposed to
        # Python, in which case the writer-vs-writer race still exercises it.
        flags = [
            (torch._C._set_warnAlways, torch._C._get_warnAlways),
            (
                functorch._set_vmap_fallback_enabled,
                functorch._is_vmap_fallback_enabled,
            ),
            (functorch._set_vmap_fallback_warning_enabled, None),
        ]
        # CUDA-only binding; absent on the CPU-only TSan build.
        if hasattr(torch._C, "_cudnn_set_conv_benchmark_empty_cache"):
            flags.append(
                (
                    torch._C._cudnn_set_conv_benchmark_empty_cache,
                    torch._C._cuda_get_conv_benchmark_empty_cache,
                )
            )

        def toggle(setter, getter, idx):
            for i in range(200):
                setter(bool((idx + i) & 1))
                if getter is not None:
                    getter()

        for setter, getter in flags:
            original = getter() if getter is not None else None
            try:
                run_concurrently([partial(toggle, setter, getter, t) for t in range(4)])
            finally:
                if original is not None:
                    setter(original)


if __name__ == "__main__":
    run_tests()
