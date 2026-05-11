# Owner(s): ["module: inductor"]
from unittest.mock import MagicMock, patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    generate_lookup_hash_from_source_code,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
    requires_triton,
)


@instantiate_parametrized_tests
class TestAsyncCompile(TestCase):
    @requires_gpu()
    @requires_triton()
    @parametrize("method", ("subprocess", "fork", "spawn"))
    def test_pool(self, method):
        def fn(x, y):
            return x + y

        x = torch.rand(10).to(GPU_TYPE)
        y = torch.rand(10).to(GPU_TYPE)

        with config.patch("worker_start_method", method):
            shutdown_compile_workers()
            AsyncCompile.wait_pool_ready()

            with fresh_cache():
                compiled_fn = torch.compile(fn)
                self.assertEqual(fn(x, y), compiled_fn(x, y))

    @requires_gpu()
    @requires_triton()
    def test_bad_kernel(self):
        shutdown_compile_workers()

        with config.patch(worker_start_method="subprocess", compile_threads=8):
            async_compile = AsyncCompile()
            AsyncCompile.wait_pool_ready()
            with self.assertRaises(SubprocException):
                async_compile.triton(
                    "fake_kernel_name", source_code="This definitely doesn't exist"
                ).result()

    @requires_gpu()
    @requires_triton()
    def test_wait_pool_ready(self):
        shutdown_compile_workers()

        with config.patch(worker_start_method="subprocess", compile_threads=8):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile._ready_future.done())
            self.assertTrue(AsyncCompile.use_process_pool())

    @requires_gpu()
    @requires_triton()
    @patch("torch._inductor.runtime.coordinate_descent_tuner.CoordescTuner.autotune")
    @parametrize("method", ("subprocess", "fork", "spawn"))
    def test_autotune_lookup_table(self, mock_autotune, method):
        def f(a, b):
            return (a @ b).to(torch.float32).sum(dim=1)

        # Fake name to make sure the lookup table is name agnostic
        # When codegen/triton.py is changed, func_def must be updated
        loop_header = (
            "for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):"
            if torch.version.hip
            else "for r0_offset in tl.range(0, r0_numel, R0_BLOCK):"
        )

        func_def = f"""
def triton_fused_fake_name(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 11776
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    {loop_header}
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 11776*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)

"""

        fn_hash = generate_lookup_hash_from_source_code(
            str({"x": 1024, "r0_": 16384}), func_def
        )
        block_configs = {
            "XBLOCK": 1,
            "R0_BLOCK": 128,
        }
        num_warps = 16
        num_stages = 1
        autotune_lookup_table = {
            fn_hash: {**block_configs, "num_warps": num_warps, "num_stages": num_stages}
        }
        autotune_config = Config(
            block_configs, num_warps=num_warps, num_stages=num_stages
        )
        mock_autotune.return_value = autotune_config

        a = torch.randn(1152, 1024, device=GPU_TYPE, dtype=torch.float16).T
        b = torch.randn(1152, 11776, device=GPU_TYPE, dtype=torch.float16)
        compiled_f = torch.compile(f)

        with config.patch(
            {
                "autotune_lookup_table": autotune_lookup_table,
                "coordinate_descent_tuning": True,
                "worker_start_method": method,
            }
        ):
            shutdown_compile_workers()
            AsyncCompile.wait_pool_ready()
            with fresh_cache():
                compiled_f(a, b)

        # Check that the input to coordinate descent (the resulting chosen config)
        # is the same as the one in the lookup table
        mock_autotune.assert_called_once()
        args, _ = mock_autotune.call_args
        self.assertTrue(isinstance(args[1], Config))

        self.assertEqual(args[1].kwargs, autotune_config.kwargs)
        self.assertEqual(args[1].num_warps, autotune_config.num_warps)
        self.assertEqual(args[1].num_stages, autotune_config.num_stages)

    @requires_gpu()
    @requires_triton()
    def test_non_combo_parallel_precompile(self):
        # Worker subprocess compiles configs via _precompile_configs_parallel
        # (threads inside subprocess). The parent's _precompile_worker should
        # see compile_results already populated and take the early-return.
        def fn(x):
            return x.softmax(dim=-1).sum(dim=-1)

        x = torch.rand(64, 4096, device=GPU_TYPE)
        out_eager = fn(x)

        parent_compiled: list[bool] = []
        orig_worker = CachingAutotuner._precompile_worker

        def wrap_worker(self):
            had_results = bool(self.compile_results)
            orig_worker(self)
            parent_compiled.append(not had_results and bool(self.compile_results))

        with (
            fresh_cache(),
            config.patch(max_autotune=True, combo_kernels=False, compile_threads=4),
            patch(
                "torch._inductor.async_compile.get_compile_threads",
                return_value=4,
            ),
            patch.object(CachingAutotuner, "_precompile_worker", wrap_worker),
        ):
            # Restart pool so workers pick up the latest source code.
            shutdown_compile_workers()
            AsyncCompile.wait_pool_ready()
            out_compiled = torch.compile(fn)(x)

        self.assertEqual(out_eager, out_compiled)
        # Worker should have compiled; parent's _precompile_worker
        # early-returns. Any True entry means parent compiled (regression).
        parent_did_compile = [x for x in parent_compiled if x]
        self.assertEqual(
            parent_did_compile,
            [],
            f"Parent-side _precompile_worker compiled configs instead of "
            f"early-returning (worker should have compiled). "
            f"parent_compiled={parent_compiled}",
        )

    @requires_gpu()
    @requires_triton()
    def test_compile_threads_one_serial_fallback(self):
        # With compile_threads=1, _precompile_configs_parallel must not
        # touch AsyncCompile.pool() if invoked. The non-combo compile path
        # is fully synchronous in this case, so this primarily guards the
        # helper's serial-fallback branch.
        def fn(x):
            return x.softmax(dim=-1).sum(dim=-1)

        x = torch.rand(64, 4096, device=GPU_TYPE)
        out_eager = fn(x)

        pool_mock = MagicMock(name="AsyncCompile.pool")

        with (
            fresh_cache(),
            config.patch(max_autotune=True, combo_kernels=False, compile_threads=1),
            patch(
                "torch._inductor.async_compile.get_compile_threads",
                return_value=1,
            ),
            patch("torch._inductor.async_compile.AsyncCompile.pool", pool_mock),
        ):
            out_compiled = torch.compile(fn)(x)

        self.assertEqual(out_eager, out_compiled)
        # Pool must never be invoked on the serial fallback branch.
        self.assertEqual(
            pool_mock.call_count,
            0,
            "AsyncCompile.pool() was invoked under compile_threads=1; the "
            "serial fallback branch should not touch the pool.",
        )

    @requires_gpu()
    @requires_triton()
    def test_no_duplicate_bundler_puts(self):
        # Per-config compile runs in the worker subprocess. The parent's
        # _precompile_worker mirrors one TritonBundler.put per result on
        # its early-return path (so the parent's bundler observes the
        # entries). The parent must not see a key more than once.
        def fn(x):
            return x.softmax(dim=-1).sum(dim=-1)

        x = torch.rand(64, 4096, device=GPU_TYPE)
        out_eager = fn(x)

        from torch._inductor.runtime import triton_heuristics as th

        put_calls: list[tuple] = []
        orig_put = th.TritonBundler.put

        def counting_put(key, device):
            put_calls.append((key, device))
            return orig_put(key, device)

        with (
            fresh_cache(),
            config.patch(max_autotune=True, combo_kernels=False, compile_threads=4),
            patch(
                "torch._inductor.async_compile.get_compile_threads",
                return_value=4,
            ),
            patch.object(th.TritonBundler, "put", staticmethod(counting_put)),
        ):
            out_compiled = torch.compile(fn)(x)

        self.assertEqual(out_eager, out_compiled)
        seen_keys: set = set()
        duplicates: list = []
        for key, device in put_calls:
            ident = (key, device)
            if ident in seen_keys:
                duplicates.append(ident)
            seen_keys.add(ident)
        self.assertEqual(
            duplicates,
            [],
            f"TritonBundler.put was called with duplicate (key, device) "
            f"entries on the parent. Duplicates: {duplicates}",
        )

    def test_wait_futures_timeout(self):
        """A compile future that doesn't finish within
        compile_worker_wait_timeout causes _wait_futures to raise
        RuntimeError naming the kernel.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        # Use an Event so the worker thread exits promptly once the assertion
        # passes. A plain time.sleep here would keep the interpreter alive
        # until it completes.
        release = threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            hanging_future = pool.submit(release.wait)
            scope = {"kernel_that_hangs": hanging_future}
            with config.patch(compile_worker_wait_timeout=1):
                async_compile = AsyncCompile()
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"compile-worker future for 'kernel_that_hangs' did not "
                    r"complete within",
                ):
                    async_compile._wait_futures(scope)
        finally:
            release.set()
            pool.shutdown(wait=True)


if __name__ == "__main__":
    run_tests()
