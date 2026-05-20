# Owner(s): ["module: inductor"]
import re
import types
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import (
    AsyncCompile,
    CompiledTritonKernels,
    shutdown_compile_workers,
)
from torch._inductor.autotune_process import TritonBenchmarkRequest
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.compile_tasks import _worker_compile_triton
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
    generate_lookup_hash_from_source_code,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.profiler import ProfilerActivity
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
    @config.patch({"triton.unique_kernel_names": False})
    def test_descriptive_kernel_name_not_in_triton_source(self):
        def add_only(x):
            return x + 1

        def sin_then_add(y, x):
            return y.sin(), x + 1

        x = torch.randn(16, device=GPU_TYPE)
        y = torch.randn(16, device=GPU_TYPE)

        def get_triton_blocks(fn, *args):
            _, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
            code = "\n".join(codes)
            blocks = re.findall(
                r"async_compile\.triton\('triton_', '''\n(.*?)\n'''", code, re.DOTALL
            )
            return code, blocks

        with fresh_cache():
            add_code, add_blocks = get_triton_blocks(add_only, x)
            both_code, both_blocks = get_triton_blocks(sin_then_add, y, x)

        self.assertEqual(len(add_blocks), 1)
        self.assertEqual(len(both_blocks), 2)
        self.assertEqual(add_blocks[0], both_blocks[1])
        self.assertEqual(
            CompiledTritonKernels.key(add_blocks[0]),
            CompiledTritonKernels.key(both_blocks[1]),
        )
        self.assertNotIn("'kernel_name'", add_blocks[0])
        self.assertIn("descriptive_name='triton_poi_fused_add_0'", add_code)
        self.assertIn("descriptive_name='triton_poi_fused_add_1'", both_code)

        with fresh_cache():
            compiled = torch.compile(sin_then_add, fullgraph=True)
            for _ in range(2):
                compiled(y, x)
            getattr(torch, GPU_TYPE).synchronize()
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU], record_shapes=True
            ) as prof:
                compiled(y, x)
            getattr(torch, GPU_TYPE).synchronize()

        self.assertTrue(
            any(event.name == "triton_poi_fused_add_1" for event in prof.events())
        )

    def test_worker_compile_triton_uses_descriptive_kernel_name(self):
        class FakeKernel:
            def __init__(self, inductor_meta=None):
                self.inductor_meta = inductor_meta or {}
                self.kernel_name_at_precompile = None

            def with_kernel_name(self, kernel_name):
                return FakeKernel({**self.inductor_meta, "kernel_name": kernel_name})

            def precompile(self, *, warm_cache_only):
                self.kernel_name_at_precompile = self.inductor_meta.get("kernel_name")

            def prepare_for_pickle(self):
                pass

        original_kernel = FakeKernel()
        kernel, _ = _worker_compile_triton(
            lambda: original_kernel, {}, {}, "triton_template_descriptive_name"
        )

        self.assertIsNot(kernel, original_kernel)
        self.assertNotIn("kernel_name", original_kernel.inductor_meta)
        self.assertIsNone(original_kernel.kernel_name_at_precompile)
        self.assertEqual(
            kernel.kernel_name_at_precompile, "triton_template_descriptive_name"
        )

    def test_triton_benchmark_request_uses_descriptive_kernel_name(self):
        renamed_kernels = []

        class FakeKernel:
            def __init__(self, inductor_meta=None):
                self.inductor_meta = inductor_meta or {}
                self.kernel_name_at_precompile = None
                self.launchers = [types.SimpleNamespace(n_regs=0)]

            def with_kernel_name(self, kernel_name):
                kernel = FakeKernel({**self.inductor_meta, "kernel_name": kernel_name})
                renamed_kernels.append(kernel)
                return kernel

            def precompile(self):
                self.kernel_name_at_precompile = self.inductor_meta.get("kernel_name")

            def run(self, *args, **kwargs):
                pass

        request = TritonBenchmarkRequest(
            kernel_name="triton_template_descriptive_name",
            input_tensor_meta=[],
            output_tensor_meta=[],
            extra_args=(),
            module_path="fake_path",
            module_cache_key="fake_key",
            num_stages=1,
            num_warps=1,
        )
        kernel = FakeKernel()
        module = types.SimpleNamespace(triton_template_descriptive_name=kernel)

        with patch(
            "torch._inductor.autotune_process.PyCodeCache.load_by_key_path",
            return_value=module,
        ):
            run_fn = request.make_run_fn(out=torch.empty(()))

        self.assertEqual(len(renamed_kernels), 1)
        self.assertIs(run_fn.func.__self__, renamed_kernels[0])
        self.assertEqual(
            run_fn.func.__self__.inductor_meta["kernel_name"],
            "triton_template_descriptive_name",
        )
        self.assertNotIn("kernel_name", kernel.inductor_meta)

        with patch(
            "torch._inductor.autotune_process.PyCodeCache.load_by_key_path",
            return_value=module,
        ):
            request.precompile()

        self.assertEqual(len(renamed_kernels), 2)
        self.assertEqual(
            renamed_kernels[1].kernel_name_at_precompile,
            "triton_template_descriptive_name",
        )
        self.assertIsNone(kernel.kernel_name_at_precompile)

    def test_triton_benchmark_module_uses_returned_kernel_name(self):
        class FakeKernel:
            def __init__(self, inductor_meta=None):
                self.inductor_meta = inductor_meta or {}

            def with_kernel_name(self, kernel_name):
                return FakeKernel({**self.inductor_meta, "kernel_name": kernel_name})

        original_kernel = FakeKernel()
        mod = types.SimpleNamespace(triton_=original_kernel)

        TritonScheduling._set_benchmark_kernel_name(mod)

        self.assertIsNot(mod.triton_, original_kernel)
        self.assertNotIn("kernel_name", original_kernel.inductor_meta)
        self.assertEqual(mod.triton_.inductor_meta["kernel_name"], "triton_")

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
