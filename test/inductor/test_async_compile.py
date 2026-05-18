# Owner(s): ["module: inductor"]
import types
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import (
    AsyncCompile,
    CompiledTritonKernels,
    shutdown_compile_workers,
)
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.hints import DeviceProperties, triton_meta_device_context
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
    @staticmethod
    def _device_props(
        cc: int, multi_processor_count: int = 1, index: int = 0
    ) -> DeviceProperties:
        major = cc // 10
        return DeviceProperties(
            type="cuda",
            index=index,
            multi_processor_count=multi_processor_count,
            cc=cc,
            major=major,
            regs_per_multiprocessor=65536,
            max_threads_per_multi_processor=1536,
            max_threads_per_block=1024,
            warp_size=32,
        )

    def test_device_properties_repr_uses_runtime_lookup(self):
        stale_props = self._device_props(cc=80)
        runtime_props = self._device_props(cc=90, multi_processor_count=2, index=1)

        serialized = repr(stale_props)
        self.assertIn("DeviceProperties.create_from_device_str('cuda', 0)", serialized)
        self.assertNotIn("cc=80", serialized)
        self.assertNotIn("multi_processor_count=1", serialized)

        with patch.object(
            DeviceProperties, "create_from_device_str", return_value=runtime_props
        ) as create_from_device_str:
            self.assertIs(
                eval(serialized, {"DeviceProperties": DeviceProperties}),
                runtime_props,
            )

        create_from_device_str.assert_called_once_with("cuda", 0)

    def test_device_properties_context_supplies_runtime_props(self):
        runtime_props = self._device_props(cc=90, multi_processor_count=2)

        with triton_meta_device_context("cuda:0", runtime_props):
            self.assertEqual(
                DeviceProperties.create_from_device_str("cuda", 0),
                runtime_props,
            )
            self.assertEqual(
                eval(
                    "DeviceProperties.create_from_device_str('cuda', 0)",
                    {"DeviceProperties": DeviceProperties},
                ),
                runtime_props,
            )

    @patch("torch._inductor.async_compile._set_triton_ptxas_path")
    @patch("torch._inductor.async_compile._set_triton_libdevice_path")
    @patch("torch._inductor.runtime.triton_heuristics.CachingAutotuner.precompile")
    def test_triton_device_str_replaces_stale_device_properties(
        self, mock_precompile, mock_set_libdevice_path, mock_set_ptxas_path
    ):
        class FakeFn:
            __name__ = "fake_kernel"
            arg_names = []
            src = ""

        stale_props = self._device_props(cc=80)
        runtime_props = self._device_props(cc=90, multi_processor_count=2)

        def fake_load(source_code, extra=""):
            kernel = CachingAutotuner(
                FakeFn(),
                {
                    "device": stale_props,
                    "signature": {},
                    "constants": {},
                    "configs": [object()],
                },
                [Config({}, num_warps=1, num_stages=1)],
                save_cache_hook=None,
                mutated_arg_names=[],
                optimize_mem=False,
                heuristic_type="pointwise",
            )
            return types.SimpleNamespace(fake_kernel=kernel)

        CompiledTritonKernels.cache_clear()
        with (
            config.patch(compile_threads=1),
            patch("torch._inductor.codecache.PyCodeCache.load", new=fake_load),
            patch.object(
                DeviceProperties,
                "create_from_device_str",
                return_value=runtime_props,
            ) as create_from_device_str,
        ):
            kernel = AsyncCompile().triton("fake_kernel", "fake_source")

        self.assertEqual(kernel.device_props, runtime_props)
        self.assertEqual(kernel.triton_meta["device"], runtime_props.index)
        self.assertEqual(kernel.triton_meta["device_type"], runtime_props.type)
        create_from_device_str.assert_any_call("cuda")
        create_from_device_str.assert_any_call("cuda", None)
        mock_precompile.assert_called_once()

    @patch("torch._inductor.async_compile._set_triton_ptxas_path")
    @patch("torch._inductor.async_compile._set_triton_libdevice_path")
    @patch("torch._inductor.runtime.triton_heuristics.CachingAutotuner.precompile")
    def test_triton_runtime_device_properties_discriminate_caches(
        self, mock_precompile, mock_set_libdevice_path, mock_set_ptxas_path
    ):
        class FakeFn:
            __name__ = "fake_kernel"
            arg_names = []
            src = ""

        stale_props = self._device_props(cc=80)
        props_a = self._device_props(cc=80, multi_processor_count=1)
        props_b = self._device_props(cc=90, multi_processor_count=2)
        current_props = props_a
        load_extras = []

        def fake_create_from_device_str(device_str, index=None):
            return current_props

        def fake_load(source_code, extra=""):
            load_extras.append(extra)
            kernel = CachingAutotuner(
                FakeFn(),
                {
                    "device": stale_props,
                    "signature": {},
                    "constants": {},
                    "configs": [object()],
                },
                [Config({}, num_warps=1, num_stages=1)],
                save_cache_hook=None,
                mutated_arg_names=[],
                optimize_mem=False,
                heuristic_type="pointwise",
            )
            return types.SimpleNamespace(fake_kernel=kernel)

        CompiledTritonKernels.cache_clear()
        with (
            config.patch(compile_threads=1),
            patch("torch._inductor.codecache.PyCodeCache.load", new=fake_load),
            patch.object(
                DeviceProperties,
                "create_from_device_str",
                side_effect=fake_create_from_device_str,
            ),
        ):
            kernel_a = AsyncCompile().triton(
                "fake_kernel", "fake_source", device_str="cuda:0"
            )
            current_props = props_b
            kernel_b = AsyncCompile().triton(
                "fake_kernel", "fake_source", device_str="cuda:0"
            )

        self.assertEqual(kernel_a.device_props, props_a)
        self.assertEqual(kernel_b.device_props, props_b)
        self.assertEqual(len(load_extras), 2)
        self.assertNotEqual(load_extras[0], load_extras[1])

        future_a = object()
        future_b = object()
        CompiledTritonKernels.save("fake_source", future_a, extra=load_extras[0])
        CompiledTritonKernels.save("fake_source", future_b, extra=load_extras[1])
        self.assertIs(
            CompiledTritonKernels.get("fake_source", extra=load_extras[0]),
            future_a,
        )
        self.assertIs(
            CompiledTritonKernels.get("fake_source", extra=load_extras[1]),
            future_b,
        )
        mock_precompile.assert_called()

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
