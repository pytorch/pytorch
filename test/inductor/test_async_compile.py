# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
from torch._inductor.compile_worker.subproc_pool import SubprocException
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.runtime.triton_heuristics import (
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
        func_def = """
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
    for r0_offset in range(0, r0_numel, R0_BLOCK):
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


if __name__ == "__main__":
    run_tests()
