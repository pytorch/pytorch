# Owner(s): ["module: inductor"]
"""Test thread pool for Triton compilation."""

from concurrent.futures import ThreadPoolExecutor

import torch._inductor.config as config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.test_case import run_tests, TestCase


class TestThreadPool(TestCase):
    def setUp(self):
        super().setUp()
        self._orig_mode = config.compile_worker_mode
        self._orig_threads = config.compile_threads
        # Ensure we have multiple compile threads
        if config.compile_threads is None or config.compile_threads <= 1:
            config.compile_threads = 4

    def tearDown(self):
        config.compile_worker_mode = self._orig_mode
        config.compile_threads = self._orig_threads
        AsyncCompile.thread_pool.cache_clear()
        AsyncCompile.process_pool.cache_clear()
        super().tearDown()

    def test_thread_pool_creation(self):
        """Thread pool is created as a ThreadPoolExecutor with correct worker count."""
        config.compile_worker_mode = "thread"
        AsyncCompile.thread_pool.cache_clear()

        pool = AsyncCompile.thread_pool()
        self.assertIsInstance(pool, ThreadPoolExecutor)
        self.assertEqual(pool._max_workers, config.compile_threads)

    def test_should_use_thread_workers_thread_mode(self):
        """should_use_thread_workers returns True in thread mode."""
        config.compile_worker_mode = "thread"
        self.assertTrue(AsyncCompile.should_use_thread_workers())

    def test_should_use_thread_workers_process_mode(self):
        """should_use_thread_workers returns False in process mode."""
        config.compile_worker_mode = "process"
        self.assertFalse(AsyncCompile.should_use_thread_workers())


if __name__ == "__main__":
    run_tests()
