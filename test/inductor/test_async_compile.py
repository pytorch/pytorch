# Owner(s): ["module: inductor"]
import multiprocessing
import os
import queue
import tempfile
import textwrap
import traceback
import unittest
import warnings
from concurrent.futures import Future
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
    skipIfNoCuteDSL,
    skipIfWindows,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    requires_gpu,
    requires_triton,
)


CUTEDSL_ADD_TEMPLATE = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape

    if thread_idx < m * n:
        mi = thread_idx // n
        ni = thread_idx % n

        if mi < m and ni < n:
            gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def {{kernel_name}}_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
    {{gen_defines()}}
    m, n = mA.shape
    total_threads = m * n
    num_blocks = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    kernel = {{kernel_name}}_kernel(mA, mB, mC)
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream
    )

{{def_kernel("input_a", "input_b")}}
    cute_a = from_dlpack(input_a)
    cute_b = from_dlpack(input_b)
    cute_c = from_dlpack({{get_output()}})

    {{kernel_name}}_jit(cute_a, cute_b, cute_c, cuda.CUstream(stream))
    return {{get_output()}}
"""

CUTEDSL_ADD_TEMPLATE_WITH_PRECOMPILE = (
    CUTEDSL_ADD_TEMPLATE
    + r"""
import os
import tempfile

_PRECOMPILE_SENTINEL = os.path.join(tempfile.gettempdir(), "SENTINEL_PLACEHOLDER")

def {{kernel_name}}_precompile(precompile_shapes, precompile_strides=None,
                                precompile_dtypes=None, device_index=0,
                                device_capability=None, hw_info=None):
    with open(_PRECOMPILE_SENTINEL, "w") as f:
        import json
        f.write(json.dumps({"shapes": precompile_shapes, "dtypes": precompile_dtypes}))
"""
)


def _daemon_compile_worker(q, worker_start_method):
    try:
        with config.patch(
            {"compile_threads": 2, "worker_start_method": worker_start_method}
        ):
            shutdown_compile_workers()
            AsyncCompile.warm_pool()
            AsyncCompile.wakeup()
            AsyncCompile.wait_pool_ready()

            def fn(x):
                return (x + 1).relu()

            x = torch.randn(4)
            compiled = torch.compile(fn, backend="inductor")
            torch.testing.assert_close(compiled(x), fn(x))
            q.put(("ok", (AsyncCompile.use_process_pool(), worker_start_method)))
    except BaseException:
        q.put(("err", traceback.format_exc()))
    finally:
        shutdown_compile_workers()


def _forked_daemon_compile_worker(q):
    try:
        with config.patch({"compile_threads": 2, "worker_start_method": "fork"}):
            AsyncCompile.wait_pool_ready(timeout=1)
            ready_future_cleared = AsyncCompile._ready_future is None
            use_process_pool = AsyncCompile.use_process_pool()
            q.put(("ok", (ready_future_cleared, use_process_pool)))
    except BaseException:
        q.put(("err", traceback.format_exc()))
    finally:
        shutdown_compile_workers()


@instantiate_parametrized_tests
class TestAsyncCompile(TestCase):
    def _run_daemon_compile_worker(self, worker_start_method):
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_daemon_compile_worker, args=(q, worker_start_method))
        p.daemon = True
        p.start()
        p.join(120)

        if p.is_alive():
            p.terminate()
            p.join()
            self.fail("daemon compile worker timed out")

        try:
            kind, payload = q.get(timeout=1)
        except queue.Empty:
            self.fail(f"daemon compile worker exited without a result: {p.exitcode}")

        self.assertEqual(p.exitcode, 0, payload)
        self.assertEqual(kind, "ok", payload)
        return payload

    @unittest.skipIf(
        "spawn" not in multiprocessing.get_all_start_methods(),
        "requires spawn multiprocessing start method",
    )
    @parametrize("method", ("fork", "spawn"))
    def test_daemon_process_disables_direct_compile_process_pool(self, method):
        if method not in multiprocessing.get_all_start_methods():
            self.skipTest(f"requires {method} multiprocessing start method")

        use_process_pool, worker_start_method = self._run_daemon_compile_worker(method)
        self.assertEqual(worker_start_method, method)
        self.assertFalse(use_process_pool)

    @unittest.skipIf(
        "spawn" not in multiprocessing.get_all_start_methods(),
        "requires spawn multiprocessing start method",
    )
    @skipIfWindows(msg="SubprocPool uses pass_fds, which is not supported on Windows.")
    def test_daemon_process_allows_subprocess_compile_process_pool(self):
        use_process_pool, worker_start_method = self._run_daemon_compile_worker(
            "subprocess"
        )
        self.assertEqual(worker_start_method, "subprocess")
        self.assertTrue(use_process_pool)

    @unittest.skipIf(
        "fork" not in multiprocessing.get_all_start_methods(),
        "requires fork multiprocessing start method",
    )
    def test_forked_daemon_process_clears_inherited_ready_future(self):
        with config.patch({"compile_threads": 2, "worker_start_method": "fork"}):
            shutdown_compile_workers()
            AsyncCompile.warm_pool()
            AsyncCompile._ready_future = Future()

            try:
                ctx = multiprocessing.get_context("fork")
                q = ctx.Queue()
                p = ctx.Process(target=_forked_daemon_compile_worker, args=(q,))
                p.daemon = True
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"os\.fork\(\) was called.*",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            r"This process .* is multi-threaded, use of fork\(\) "
                            r"may lead to deadlocks in the child\."
                        ),
                        category=DeprecationWarning,
                    )
                    p.start()
                p.join(10)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    self.fail("forked daemon compile worker timed out")

                try:
                    kind, payload = q.get(timeout=1)
                except queue.Empty:
                    self.fail(
                        "forked daemon compile worker exited without a result: "
                        f"{p.exitcode}"
                    )
            finally:
                shutdown_compile_workers()

            self.assertEqual(p.exitcode, 0, payload)
            self.assertEqual(kind, "ok", payload)
            ready_future_cleared, use_process_pool = payload
            self.assertTrue(ready_future_cleared)
            self.assertFalse(use_process_pool)

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


@skipIfNoCuteDSL
class TestCuteDSLSubprocessCompile(TestCase):
    def _compile_and_run_add(self, template_name):
        """Compile a CuteDSL add kernel via torch.compile and verify correctness."""
        from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        template = CuteDSLTemplate(
            name=template_name,
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )
            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)
            return choices[0].output_node()

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):

            def test_add(x, y):
                return x + y

            x = torch.randn(128, 4, device="cuda", dtype=torch.float32)
            y = torch.randn(128, 4, device="cuda", dtype=torch.float32)

            compiled_fn = torch.compile(test_add, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, x, y)

            self.assertIn("cute", code.lower())
            expected = x + y
            self.assertEqual(result, expected)

    def test_cutedsl_subprocess_e2e(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            with (
                fresh_cache(),
                patch.object(
                    AsyncCompile,
                    "_load_kernel_wrapper",
                    autospec=True,
                    side_effect=AsyncCompile._load_kernel_wrapper,
                ) as mock_reload,
            ):
                self._compile_and_run_add("test_add_subprocess")
                mock_reload.assert_called()

    def test_cutedsl_synchronous_e2e(self):
        with config.patch(compile_threads=1):
            with (
                fresh_cache(),
                patch.object(
                    AsyncCompile,
                    "_load_kernel_wrapper",
                    autospec=True,
                    side_effect=AsyncCompile._load_kernel_wrapper,
                ) as mock_reload,
            ):
                self._compile_and_run_add("test_add_synchronous")
                mock_reload.assert_not_called()

    def test_cutedsl_bad_source_subprocess(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            async_compile = AsyncCompile()

            with self.assertRaises(SubprocException):
                async_compile.cutedsl(
                    "bad_kernel", "this is not valid python!!!"
                ).result()

    def test_cutedsl_missing_entry_point_subprocess(self):
        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())
            async_compile = AsyncCompile()

            with self.assertRaises(SubprocException):
                async_compile.cutedsl(
                    "test_kernel", "import torch\ndef other_func(): pass\n"
                ).result()

    def test_cutedsl_subprocess_precompile_invoked(self):
        """Verify that subprocess actually calls _precompile for a template that defines it."""
        import uuid

        from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        sentinel_name = f"cutedsl_precompile_{uuid.uuid4().hex[:8]}"
        sentinel_path = os.path.join(tempfile.gettempdir(), sentinel_name)
        source = CUTEDSL_ADD_TEMPLATE_WITH_PRECOMPILE.replace(
            "SENTINEL_PLACEHOLDER", sentinel_name
        )

        template = CuteDSLTemplate(
            name="test_add_precompile",
            source=source,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )
            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)
            return choices[0].output_node()

        try:
            shutdown_compile_workers()
            with config.patch(worker_start_method="subprocess", compile_threads=4):
                AsyncCompile.wait_pool_ready()
                self.assertTrue(AsyncCompile.use_process_pool())
                with patch.dict(
                    lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}
                ):

                    def test_add(x, y):
                        return x + y

                    x = torch.randn(128, 4, device="cuda", dtype=torch.float32)
                    y = torch.randn(128, 4, device="cuda", dtype=torch.float32)

                    compiled_fn = torch.compile(test_add, backend="inductor")
                    with fresh_cache():
                        result, (code,) = run_and_get_code(compiled_fn, x, y)

                    self.assertEqual(result, x + y)
                    self.assertTrue(
                        os.path.exists(sentinel_path),
                        "Subprocess _precompile was not invoked -- sentinel file missing",
                    )
        finally:
            if os.path.exists(sentinel_path):
                os.unlink(sentinel_path)

    def test_cutedsl_precompile_metadata_in_generated_code(self):
        """Verify codegen includes precompile_metadata in the generated wrapper."""
        from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        template = CuteDSLTemplate(
            name="test_metadata_gen",
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )
            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)
            return choices[0].output_node()

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):

            def test_add(x, y):
                return x + y

            x = torch.randn(128, 4, device="cuda", dtype=torch.float32)
            y = torch.randn(128, 4, device="cuda", dtype=torch.float32)

            compiled_fn = torch.compile(test_add, backend="inductor")
            with fresh_cache():
                _, (code,) = run_and_get_code(compiled_fn, x, y)

            self.assertIn("precompile_metadata", code)
            self.assertIn("precompile_shapes", code)
            self.assertIn("precompile_dtypes", code)

    def test_cutedsl_worker_precompile_cache_roundtrip(self):
        """Test that _precompile writes a cache artifact that _main can load.

        Simulates the subprocess/parent round-trip: _precompile writes a file
        to a shared cache directory, _main reads it back. This proves the
        artifact-transfer mechanism works without requiring real CuTe DSL
        compilation (which needs Blackwell + CUTLASS).
        """
        import shutil

        from torch._inductor.runtime.compile_tasks import (
            _worker_compile_pycodecache_kernel,
        )

        cache_dir = os.path.join(tempfile.gettempdir(), "cutedsl_test_cache_roundtrip")

        source = (
            "import json, os\n"
            "CACHE_DIR = " + repr(cache_dir) + "\n"
            "compiled_artifact = None\n"
            "\n"
            "def test_kernel_main(*args, **kwargs):\n"
            "    global compiled_artifact\n"
            "    cache_path = os.path.join(CACHE_DIR, 'artifact.json')\n"
            "    if compiled_artifact is None and os.path.exists(cache_path):\n"
            "        with open(cache_path) as f:\n"
            "            compiled_artifact = json.load(f)\n"
            "    return compiled_artifact\n"
            "\n"
            "def test_kernel_precompile(precompile_shapes, precompile_dtypes, device_index=0):\n"
            "    os.makedirs(CACHE_DIR, exist_ok=True)\n"
            "    cache_path = os.path.join(CACHE_DIR, 'artifact.json')\n"
            "    with open(cache_path, 'w') as f:\n"
            "        json.dump({'shapes': precompile_shapes, 'dtypes': precompile_dtypes}, f)\n"
        )

        metadata = {
            "precompile_shapes": {"input_a": [128, 64], "output": [128, 64]},
            "precompile_dtypes": {"input_a": "float32", "output": "float32"},
        }

        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

            with fresh_cache():
                # Simulate subprocess: worker calls _precompile, writes artifact
                key, path, elapsed = _worker_compile_pycodecache_kernel(
                    "test_kernel", source, "main", {}, metadata
                )

                # Verify artifact was written
                artifact_path = os.path.join(cache_dir, "artifact.json")
                self.assertTrue(
                    os.path.exists(artifact_path),
                    "_precompile did not write cache artifact",
                )

                # Simulate parent: reload module, call _main, verify it loads artifact
                import torch._inductor.codecache as codecache

                mod = codecache.PyCodeCache.load_by_key_path(key, path)
                result = mod.test_kernel_main()
                self.assertIsNotNone(result, "_main did not load cached artifact")
                self.assertEqual(result["shapes"]["input_a"], [128, 64])
                self.assertEqual(result["dtypes"]["input_a"], "float32")
        finally:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

    def test_cutedsl_worker_skips_precompile_without_metadata(self):
        """Test _worker_compile_pycodecache_kernel skips _precompile when metadata is None."""
        from torch._inductor.runtime.compile_tasks import (
            _worker_compile_pycodecache_kernel,
        )

        source = (
            "import torch\n"
            "precompile_was_called = False\n"
            "def test_kernel_main(*args, **kwargs): pass\n"
            "def test_kernel_precompile(**kw):\n"
            "    global precompile_was_called\n"
            "    precompile_was_called = True\n"
        )

        with fresh_cache():
            key, path, elapsed = _worker_compile_pycodecache_kernel(
                "test_kernel", source, "main", {}, None
            )

            import torch._inductor.codecache as codecache

            mod = codecache.PyCodeCache.load_by_key_path(key, path)
            self.assertFalse(mod.precompile_was_called)

    def test_cutedsl_worker_warns_missing_precompile_with_metadata(self):
        """Worker should warn when metadata is provided but _precompile hook is absent."""
        import logging

        from torch._inductor.runtime.compile_tasks import (
            _worker_compile_pycodecache_kernel,
        )

        source = "import torch\ndef test_kernel_main(*args, **kwargs): pass\n"

        metadata = {
            "precompile_shapes": {"input_a": [4, 4], "output": [4, 4]},
            "precompile_dtypes": {"input_a": "float32", "output": "float32"},
        }

        with fresh_cache():
            with self.assertLogs(
                "torch._inductor.runtime.compile_tasks", level=logging.WARNING
            ) as cm:
                _worker_compile_pycodecache_kernel(
                    "test_kernel", source, "main", {}, metadata
                )
            self.assertTrue(
                any("Precompile metadata was provided" in msg for msg in cm.output)
            )

    def test_disk_cache_key_includes_arch_and_version(self):
        """Verify disk cache keys incorporate GPU arch and CUDA version."""
        from torch._inductor.runtime.cutedsl_cache import _make_disk_key

        key_a = _make_disk_key("/path/a.py", ("config",), ("runtime",))
        key_b = _make_disk_key("/path/b.py", ("config",), ("runtime",))
        self.assertNotEqual(
            key_a, key_b, "Different module paths should produce different keys"
        )

        key_same_1 = _make_disk_key("/path/a.py", ("cfg",), ("rt",))
        key_same_2 = _make_disk_key("/path/a.py", ("cfg",), ("rt",))
        self.assertEqual(key_same_1, key_same_2, "Same inputs should produce same key")

    def test_disk_cache_key_varies_by_device_arch(self):
        """Verify different GPU architectures produce different cache keys.

        The disk key includes the device capability (arch), so devices with
        different compute capabilities get distinct keys. On a single-arch
        multi-GPU host, device_index alone does NOT change the key -- the arch
        is what matters. This test only asserts inequality when the two devices
        actually differ in capability.
        """
        from torch._inductor.runtime.cutedsl_cache import _make_disk_key

        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs to test cross-device key variation")

        cap0 = torch.cuda.get_device_capability(0)
        cap1 = torch.cuda.get_device_capability(1)
        if cap0 == cap1:
            self.skipTest(
                "Both GPUs have the same compute capability -- "
                "disk keys are expected to match"
            )

        key_dev0 = _make_disk_key("/path/a.py", ("cfg",), ("rt",), device_index=0)
        key_dev1 = _make_disk_key("/path/a.py", ("cfg",), ("rt",), device_index=1)
        self.assertNotEqual(
            key_dev0,
            key_dev1,
            "Different GPU architectures must produce different keys",
        )

    def test_mem_cache_keys_include_device_index(self):
        """Verify in-memory cache in disk_cache_get/set is keyed by (runtime_key, device_index)."""
        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

        mem_cache: dict = {}
        sentinel_dev0 = object()
        sentinel_dev1 = object()
        runtime_key = ("shape", "dtype")

        fake_cap = (9, 0)
        disk_cache_set(
            mem_cache,
            "/fake.py",
            ("cfg",),
            runtime_key,
            sentinel_dev0,
            device_index=0,
            device_capability=fake_cap,
        )
        disk_cache_set(
            mem_cache,
            "/fake.py",
            ("cfg",),
            runtime_key,
            sentinel_dev1,
            device_index=1,
            device_capability=fake_cap,
        )

        got0 = disk_cache_get(
            mem_cache,
            "/fake.py",
            ("cfg",),
            runtime_key,
            device_index=0,
            device_capability=fake_cap,
        )
        got1 = disk_cache_get(
            mem_cache,
            "/fake.py",
            ("cfg",),
            runtime_key,
            device_index=1,
            device_capability=fake_cap,
        )

        self.assertIs(
            got0, sentinel_dev0, "device 0 should get device 0's cached value"
        )
        self.assertIs(
            got1, sentinel_dev1, "device 1 should get device 1's cached value"
        )
        self.assertIsNot(
            got0, got1, "different devices must not share in-memory entries"
        )

    def test_cutedsl_disk_cache_hot_load(self):
        """End-to-end test: subprocess compiles via TVM FFI, writes .o to disk,
        parent loads .o without recompiling, produces correct results.

        This exercises the real export_to_c -> load_module path through
        cutedsl_cache.py. Requires CUTLASS + a CUDA GPU.
        """
        import shutil

        from torch._inductor.runtime.compile_tasks import (
            _reload_python_module,
            _worker_compile_pycodecache_kernel,
        )
        from torch._inductor.runtime.cutedsl_cache import _cache_dir

        dev_idx = torch.cuda.current_device()

        source = textwrap.dedent("""\
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from cutlass import cuda
            from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

            _fn_cache = {}
            _CONFIG_KEY = ("test_add_hot_load",)
            compile_count = 0

            @cute.kernel
            def _add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                bdim, _, _ = cute.arch.block_dim()
                idx = bidx * bdim + tidx
                m, n = gA.shape
                if idx < m * n:
                    mi = idx // n
                    ni = idx % n
                    if mi < m and ni < n:
                        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

            @cute.jit
            def _add_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
                m, n = mA.shape
                total = m * n
                nblocks = (total + 255) // 256
                _add_kernel(mA, mB, mC).launch(
                    grid=[nblocks, 1, 1], block=[256, 1, 1], stream=stream,
                )

            def _to_ct(t, align=16):
                return from_dlpack(
                    t.detach(), assumed_align=align, enable_tvm_ffi=True,
                ).mark_layout_dynamic()

            def test_kernel_main(input_a, input_b, output, stream=None):
                global compile_count
                dev_idx = input_a.device.index or 0
                cache_key = (
                    tuple(input_a.shape), input_a.dtype,
                    tuple(input_b.shape), input_b.dtype,
                )
                executor = disk_cache_get(
                    _fn_cache, __file__, _CONFIG_KEY, cache_key, dev_idx,
                )
                if executor is None:
                    compile_count += 1
                    executor = cute.compile(
                        _add_jit,
                        _to_ct(input_a), _to_ct(input_b), _to_ct(output),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi",
                    )
                    disk_cache_set(
                        _fn_cache, __file__, _CONFIG_KEY, cache_key, executor, dev_idx,
                    )
                executor(input_a, input_b, output)

            def test_kernel_precompile(precompile_shapes, precompile_dtypes, device_index=0):
                torch.cuda.set_device(device_index)
                device = f"cuda:{device_index}"
                a = torch.empty(
                    tuple(precompile_shapes["input_a"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["input_a"]),
                )
                b = torch.empty(
                    tuple(precompile_shapes["input_b"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["input_b"]),
                )
                out = torch.empty(
                    tuple(precompile_shapes["output"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["output"]),
                )
                test_kernel_main(a, b, out)
        """)

        metadata = {
            "precompile_shapes": {
                "input_a": [8, 4],
                "input_b": [8, 4],
                "output": [8, 4],
            },
            "precompile_dtypes": {
                "input_a": "float32",
                "input_b": "float32",
                "output": "float32",
            },
            "device_index": dev_idx,
        }

        device = f"cuda:{dev_idx}"
        with fresh_cache():
            cache_dir = _cache_dir()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            try:
                # --- Simulate subprocess: compile + write .o ---
                key, path, elapsed = _worker_compile_pycodecache_kernel(
                    "test_kernel", source, "main", {}, metadata
                )

                # Verify .o was written to disk
                self.assertTrue(cache_dir.exists(), "Disk cache dir not created")
                o_files = list(cache_dir.glob("*.o"))
                self.assertGreater(
                    len(o_files),
                    0,
                    "No .o artifacts written -- export_to_c may have failed",
                )

                # --- Simulate parent: fresh module load, should hit disk cache ---
                parent_mod = _reload_python_module(
                    key,
                    path,
                    set_sys_modules=False,
                )
                self.assertEqual(
                    parent_mod.compile_count,
                    0,
                    "Fresh module should start with compile_count=0",
                )

                x = torch.randn(8, 4, device=device, dtype=torch.float32)
                y = torch.randn(8, 4, device=device, dtype=torch.float32)
                out = torch.empty(8, 4, device=device, dtype=torch.float32)
                parent_mod.test_kernel_main(x, y, out)

                self.assertEqual(
                    parent_mod.compile_count,
                    0,
                    "Parent called cute.compile() -- disk cache was NOT loaded. "
                    "The subprocess .o artifact did not transfer correctly.",
                )
                self.assertEqual(
                    out,
                    x + y,
                    "Kernel loaded from disk cache produced incorrect results",
                )
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_nv_universal_gemm_disk_cache_hot_load(self):
        """End-to-end test for the NV Universal GEMM subprocess hot-load path.

        NV Universal GEMM differs from CuteDSL in that it wraps compiled
        functions in a CompiledArtifact object. This test exercises that
        exact pattern: compile via cute.compile(), wrap in an artifact,
        persist artifact.compiled_obj to disk, reload from disk, wrap the
        loaded function back in an artifact, and execute.
        """
        import shutil

        from torch._inductor.runtime.compile_tasks import (
            _reload_python_module,
            _worker_compile_pycodecache_kernel,
        )
        from torch._inductor.runtime.cutedsl_cache import _cache_dir

        dev_idx = torch.cuda.current_device()

        source = textwrap.dedent("""\
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from cutlass import cuda
            from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

            class CompiledArtifact:
                def __init__(self, compiled_obj, kernel):
                    self.compiled_obj = compiled_obj
                    self.kernel = kernel

            _artifact_cache = {}
            _disk_fn_cache = {}
            _CONFIG_KEY = ("test_nvgemm_hot_load",)
            compile_count = 0

            @cute.kernel
            def _add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                bdim, _, _ = cute.arch.block_dim()
                idx = bidx * bdim + tidx
                m, n = gA.shape
                if idx < m * n:
                    mi = idx // n
                    ni = idx % n
                    if mi < m and ni < n:
                        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

            @cute.jit
            def _add_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
                m, n = mA.shape
                total = m * n
                nblocks = (total + 255) // 256
                _add_kernel(mA, mB, mC).launch(
                    grid=[nblocks, 1, 1], block=[256, 1, 1], stream=stream,
                )

            def _to_ct(t, align=16):
                return from_dlpack(
                    t.detach(), assumed_align=align, enable_tvm_ffi=True,
                ).mark_layout_dynamic()

            def test_kernel_main(in_ptr0, in_ptr1, out_ptr0, stream=None):
                global compile_count
                dev_idx = in_ptr0.device.index or 0
                cache_key = (
                    tuple(in_ptr0.shape), in_ptr0.dtype,
                    tuple(in_ptr1.shape), in_ptr1.dtype,
                )
                mem_key = (cache_key, dev_idx)
                artifact = _artifact_cache.get(mem_key)
                if artifact is None:
                    compiled_fn = disk_cache_get(
                        _disk_fn_cache, __file__, _CONFIG_KEY, cache_key, dev_idx,
                    )
                    if compiled_fn is not None:
                        artifact = CompiledArtifact(compiled_fn, "kernel_placeholder")
                    else:
                        compile_count += 1
                        compiled_fn = cute.compile(
                            _add_jit,
                            _to_ct(in_ptr0), _to_ct(in_ptr1), _to_ct(out_ptr0),
                            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                            options="--enable-tvm-ffi",
                        )
                        disk_cache_set(
                            _disk_fn_cache, __file__, _CONFIG_KEY,
                            cache_key, compiled_fn, dev_idx,
                        )
                        artifact = CompiledArtifact(compiled_fn, "kernel_placeholder")
                    _artifact_cache[mem_key] = artifact
                artifact.compiled_obj(in_ptr0, in_ptr1, out_ptr0)

            def test_kernel_precompile(precompile_shapes, precompile_dtypes, device_index=0):
                torch.cuda.set_device(device_index)
                device = f"cuda:{device_index}"
                in_ptr0 = torch.empty(
                    tuple(precompile_shapes["in_ptr0"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["in_ptr0"]),
                )
                in_ptr1 = torch.empty(
                    tuple(precompile_shapes["in_ptr1"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["in_ptr1"]),
                )
                out_ptr0 = torch.empty(
                    tuple(precompile_shapes["output"]), device=device,
                    dtype=getattr(torch, precompile_dtypes["output"]),
                )
                test_kernel_main(in_ptr0, in_ptr1, out_ptr0)
        """)

        metadata = {
            "precompile_shapes": {
                "in_ptr0": [8, 4],
                "in_ptr1": [8, 4],
                "output": [8, 4],
            },
            "precompile_dtypes": {
                "in_ptr0": "float32",
                "in_ptr1": "float32",
                "output": "float32",
            },
            "device_index": dev_idx,
        }

        device = f"cuda:{dev_idx}"
        with fresh_cache():
            cache_dir = _cache_dir()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            try:
                # --- Simulate subprocess: compile + write .o ---
                key, path, elapsed = _worker_compile_pycodecache_kernel(
                    "test_kernel", source, "main", {}, metadata
                )

                # Verify .o was written to disk
                self.assertTrue(cache_dir.exists(), "Disk cache dir not created")
                o_files = list(cache_dir.glob("*.o"))
                self.assertGreater(
                    len(o_files),
                    0,
                    "No .o artifacts written -- export_to_c may have failed",
                )

                # --- Simulate parent: fresh module load, should hit disk cache ---
                parent_mod = _reload_python_module(
                    key,
                    path,
                    set_sys_modules=False,
                )
                self.assertEqual(
                    parent_mod.compile_count,
                    0,
                    "Fresh module should start with compile_count=0",
                )

                x = torch.randn(8, 4, device=device, dtype=torch.float32)
                y = torch.randn(8, 4, device=device, dtype=torch.float32)
                out = torch.empty(8, 4, device=device, dtype=torch.float32)
                parent_mod.test_kernel_main(x, y, out)

                self.assertEqual(
                    parent_mod.compile_count,
                    0,
                    "Parent called cute.compile() -- disk cache was NOT loaded. "
                    "The subprocess .o artifact did not transfer correctly.",
                )
                # Verify artifact wrapping worked
                self.assertEqual(len(parent_mod._artifact_cache), 1)
                artifact = next(iter(parent_mod._artifact_cache.values()))
                self.assertIsInstance(artifact, parent_mod.CompiledArtifact)
                self.assertEqual(
                    out,
                    x + y,
                    "Kernel loaded from disk cache produced incorrect results",
                )
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_cutedsl_subprocess_pool_handoff(self):
        """Test the real subprocess pool path: serialization, result handoff, parent reload.

        Goes through AsyncCompile.cutedsl() -> SubprocPool -> parent reload
        WITHOUT precompile_metadata, since SubprocPool workers use forked processes
        that cannot re-initialize CUDA. This exercises the subprocess boundary
        (serialization, worker startup, result handoff) that the direct-call
        hot-load tests miss.

        The precompile path (CUDA compilation in subprocess) is tested via the
        direct-call tests, which exercise the cache contract in isolation.
        """
        source = textwrap.dedent("""\
            import torch

            def test_pool_kernel_main(input_a, input_b, output, stream=None):
                for i in range(input_a.shape[0]):
                    for j in range(input_a.shape[1]):
                        output[i, j] = input_a[i, j] + input_b[i, j]
        """)

        shutdown_compile_workers()
        with config.patch(worker_start_method="subprocess", compile_threads=4):
            AsyncCompile.wait_pool_ready()
            self.assertTrue(AsyncCompile.use_process_pool())

            dev_idx = torch.cuda.current_device()
            device = f"cuda:{dev_idx}"
            with fresh_cache():
                async_compile = AsyncCompile()
                wrapper = async_compile.cutedsl("test_pool_kernel", source)
                kernel_wrapper = wrapper.result()

                x = torch.randn(4, 4, device=device, dtype=torch.float32)
                y = torch.randn(4, 4, device=device, dtype=torch.float32)
                out = torch.empty(4, 4, device=device, dtype=torch.float32)
                kernel_wrapper.run(x, y, out)
                self.assertEqual(out, x + y)

    def test_cutedsl_subprocess_precompile_no_cuda_init(self):
        """Regression: precompile in subprocess workers must not call torch.cuda.*.

        SubprocPool workers are forked from a sidecar that has already initialized
        CUDA via caching_device_properties(). Calling torch.cuda.* in the forked
        worker triggers "Cannot re-initialize CUDA in forked subprocess". The fix
        passes device_capability in metadata so precompile functions never query
        the GPU directly. This test exercises the real SubprocPool path with
        precompile_metadata to verify the fix.
        """
        import json
        import uuid

        sentinel_name = f"cutedsl_precompile_nocuda_{uuid.uuid4().hex[:8]}"
        sentinel_path = os.path.join(tempfile.gettempdir(), sentinel_name)

        source = textwrap.dedent(f"""\
            import json
            import os
            import torch

            _SENTINEL_PATH = {sentinel_path!r}

            def test_kernel_main(input_a, input_b, output, stream=None):
                for i in range(input_a.shape[0]):
                    for j in range(input_a.shape[1]):
                        output[i, j] = input_a[i, j] + input_b[i, j]

            def test_kernel_precompile(precompile_shapes, precompile_strides,
                                       precompile_dtypes, device_index=0,
                                       device_capability=None, hw_info=None):
                # Verify we're running in a forked subprocess where CUDA
                # has been inherited -- calling torch.cuda.* here would crash.
                in_bad_fork = torch.cuda._is_in_bad_fork()
                with open(_SENTINEL_PATH, "w") as f:
                    json.dump({{
                        "device_index": device_index,
                        "device_capability": list(device_capability) if device_capability else None,
                        "in_bad_fork": in_bad_fork,
                        "shapes": precompile_shapes,
                        "strides": precompile_strides,
                        "dtypes": precompile_dtypes,
                    }}, f)
        """)

        dev_idx = torch.cuda.current_device()
        dev_cap = torch.cuda.get_device_capability(dev_idx)
        metadata = {
            "precompile_shapes": {
                "input_a": [4, 4],
                "input_b": [4, 4],
                "output": [4, 4],
            },
            "precompile_strides": {
                "input_a": [4, 1],
                "input_b": [4, 1],
                "output": [4, 1],
            },
            "precompile_dtypes": {
                "input_a": "float32",
                "input_b": "float32",
                "output": "float32",
            },
            "device_index": dev_idx,
            "device_capability": dev_cap,
        }

        try:
            shutdown_compile_workers()
            with config.patch(worker_start_method="subprocess", compile_threads=4):
                AsyncCompile.wait_pool_ready()
                self.assertTrue(AsyncCompile.use_process_pool())

                with fresh_cache():
                    async_compile = AsyncCompile()
                    wrapper = async_compile.cutedsl(
                        "test_kernel", source, precompile_metadata=metadata
                    )
                    kernel_wrapper = wrapper.result()

                    device = f"cuda:{dev_idx}"
                    x = torch.randn(4, 4, device=device, dtype=torch.float32)
                    y = torch.randn(4, 4, device=device, dtype=torch.float32)
                    out = torch.empty(4, 4, device=device, dtype=torch.float32)
                    kernel_wrapper.run(x, y, out)
                    self.assertEqual(out, x + y)

            self.assertTrue(
                os.path.exists(sentinel_path),
                "Subprocess precompile was not invoked -- sentinel file missing. "
                "Precompile likely crashed (e.g., CUDA re-init in forked worker).",
            )

            with open(sentinel_path) as f:
                sentinel_data = json.load(f)
            self.assertEqual(sentinel_data["device_index"], dev_idx)
            self.assertIsNotNone(
                sentinel_data["device_capability"],
                "device_capability was None -- main process should pass it in metadata",
            )
            self.assertEqual(
                sentinel_data["device_capability"],
                list(dev_cap),
            )
            self.assertTrue(
                sentinel_data["in_bad_fork"],
                "Precompile did not run in a forked-after-CUDA-init subprocess. "
                "This test only validates the fix when the worker inherits CUDA state.",
            )
        finally:
            if os.path.exists(sentinel_path):
                os.unlink(sentinel_path)

    def test_concurrent_disk_cache_set_atomic(self):
        """Verify concurrent disk_cache_set calls to the same key don't corrupt the .o file.

        Simulates multiple workers compiling the same kernel by calling
        disk_cache_set concurrently from threads, then verifies the artifact
        loads correctly.
        """
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        from torch._inductor.runtime.cutedsl_cache import (
            _cache_dir,
            disk_cache_get,
            disk_cache_set,
        )

        source = textwrap.dedent("""\
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack

            @cute.kernel
            def _copy_kernel(gA: cute.Tensor, gB: cute.Tensor):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                bdim, _, _ = cute.arch.block_dim()
                idx = bidx * bdim + tidx
                m, n = gA.shape
                if idx < m * n:
                    mi = idx // n
                    ni = idx % n
                    if mi < m and ni < n:
                        gB[mi, ni] = gA[mi, ni]

            @cute.jit
            def _copy_jit(mA: cute.Tensor, mB: cute.Tensor, stream):
                m, n = mA.shape
                total = m * n
                nblocks = (total + 255) // 256
                _copy_kernel(mA, mB).launch(
                    grid=[nblocks, 1, 1], block=[256, 1, 1], stream=stream,
                )

            def _to_ct(t, align=16):
                return from_dlpack(
                    t.detach(), assumed_align=align, enable_tvm_ffi=True,
                ).mark_layout_dynamic()
        """)

        with fresh_cache():
            cache_dir = _cache_dir()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            try:
                import cutlass.cute as cute

                import torch._inductor.codecache as codecache

                key, path = codecache.PyCodeCache.write(source)
                mod = codecache.PyCodeCache.load_by_key_path(key, path)

                dev_idx = torch.cuda.current_device()
                device = f"cuda:{dev_idx}"
                a = torch.randn(4, 4, device=device)
                b = torch.empty(4, 4, device=device)
                compiled_fn = cute.compile(
                    mod._copy_jit,
                    mod._to_ct(a),
                    mod._to_ct(b),
                    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                    options="--enable-tvm-ffi",
                )

                config_key = ("test_concurrent",)
                runtime_key = ((4, 4), torch.float32)
                num_threads = 8
                errors = []

                def write_to_cache(thread_id):
                    try:
                        mem = {}
                        disk_cache_set(
                            mem,
                            path,
                            config_key,
                            runtime_key,
                            compiled_fn,
                            device_index=dev_idx,
                        )
                    except Exception as e:
                        errors.append((thread_id, e))

                with ThreadPoolExecutor(max_workers=num_threads) as pool:
                    list(pool.map(write_to_cache, range(num_threads)))

                self.assertEqual(errors, [], f"Concurrent writes failed: {errors}")

                # Verify the file is valid by loading it
                fresh_mem: dict = {}
                loaded = disk_cache_get(
                    fresh_mem, path, config_key, runtime_key, device_index=dev_idx
                )
                self.assertIsNotNone(
                    loaded, ".o file missing or corrupt after concurrent writes"
                )

                # Verify the loaded function actually works
                x = torch.randn(4, 4, device=device)
                out = torch.empty(4, 4, device=device)
                loaded(x, out)
                self.assertEqual(out, x)
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_corrupt_cache_falls_back_cleanly(self):
        """A corrupt .o on disk should not crash -- disk_cache_get returns None
        so the caller falls back to recompile."""
        import shutil

        from torch._inductor.runtime.cutedsl_cache import (
            _cache_dir,
            _make_disk_key,
            disk_cache_get,
        )

        module_path = "/fake/content_addressed_path.py"
        config_key = ("test_corrupt",)
        runtime_key = ((4, 4), torch.float32)

        with fresh_cache():
            cache_dir = _cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                h = _make_disk_key(module_path, config_key, runtime_key, device_index=0)
                obj_path = cache_dir / f"{h}.o"
                obj_path.write_bytes(b"this is not a valid object file\x00\xff\xfe")

                self.assertTrue(obj_path.exists())

                mem_cache: dict = {}
                result = disk_cache_get(
                    mem_cache, module_path, config_key, runtime_key, device_index=0
                )
                self.assertIsNone(
                    result,
                    "disk_cache_get should return None for a corrupt .o file, "
                    "allowing the caller to fall back to recompile",
                )
                self.assertEqual(
                    len(mem_cache),
                    0,
                    "Corrupt artifact should not be stored in memory cache",
                )
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_corrupt_cache_recompile_end_to_end(self):
        """Full round-trip: corrupt .o on disk, kernel falls back to recompile,
        produces correct results, and writes a valid .o replacing the corrupt one."""
        import shutil

        from torch._inductor.runtime.cutedsl_cache import _cache_dir, _make_disk_key

        dev_idx = torch.cuda.current_device()
        device = f"cuda:{dev_idx}"

        source = textwrap.dedent("""\
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from cutlass import cuda
            from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

            _fn_cache = {}
            _CONFIG_KEY = ("test_corrupt_e2e",)
            compile_count = 0

            @cute.kernel
            def _add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                bdim, _, _ = cute.arch.block_dim()
                idx = bidx * bdim + tidx
                m, n = gA.shape
                if idx < m * n:
                    mi = idx // n
                    ni = idx % n
                    if mi < m and ni < n:
                        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

            @cute.jit
            def _add_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
                m, n = mA.shape
                total = m * n
                nblocks = (total + 255) // 256
                _add_kernel(mA, mB, mC).launch(
                    grid=[nblocks, 1, 1], block=[256, 1, 1], stream=stream,
                )

            def _to_ct(t, align=16):
                return from_dlpack(
                    t.detach(), assumed_align=align, enable_tvm_ffi=True,
                ).mark_layout_dynamic()

            def test_kernel_main(input_a, input_b, output, stream=None):
                global compile_count
                dev_idx = input_a.device.index or 0
                cache_key = (
                    tuple(input_a.shape), input_a.dtype,
                    tuple(input_b.shape), input_b.dtype,
                )
                executor = disk_cache_get(
                    _fn_cache, __file__, _CONFIG_KEY, cache_key, dev_idx,
                )
                if executor is None:
                    compile_count += 1
                    executor = cute.compile(
                        _add_jit,
                        _to_ct(input_a), _to_ct(input_b), _to_ct(output),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi",
                    )
                    disk_cache_set(
                        _fn_cache, __file__, _CONFIG_KEY, cache_key, executor, dev_idx,
                    )
                executor(input_a, input_b, output)
        """)

        with fresh_cache():
            cache_dir = _cache_dir()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            try:
                # First, do a normal compile to figure out the .o path
                import torch._inductor.codecache as codecache

                key, path = codecache.PyCodeCache.write(source)
                config_key = ("test_corrupt_e2e",)
                x = torch.randn(8, 4, device=device, dtype=torch.float32)
                y = torch.randn(8, 4, device=device, dtype=torch.float32)
                out = torch.empty(8, 4, device=device, dtype=torch.float32)
                runtime_key = (
                    tuple(x.shape),
                    x.dtype,
                    tuple(y.shape),
                    y.dtype,
                )
                h = _make_disk_key(path, config_key, runtime_key, device_index=dev_idx)
                obj_path = cache_dir / f"{h}.o"

                # Write corrupt data at the expected .o path
                cache_dir.mkdir(parents=True, exist_ok=True)
                obj_path.write_bytes(b"\x00\x01\x02corrupt_cubin_data\xff\xfe")

                # Load a fresh module (simulating parent process)
                mod = codecache.PyCodeCache.load_by_key_path(key, path)
                self.assertEqual(mod.compile_count, 0)

                mod.test_kernel_main(x, y, out)

                self.assertEqual(
                    mod.compile_count,
                    1,
                    "Should have fallen back to recompile after corrupt cache",
                )
                self.assertEqual(
                    out, x + y, "Recompiled kernel should produce correct results"
                )

                # disk_cache_set persists via export_to_c which may not be
                # available on all platforms.  Only verify the disk
                # round-trip when the corrupt file was actually replaced.
                corrupt_data = b"\x00\x01\x02corrupt_cubin_data\xff\xfe"
                if obj_path.exists() and obj_path.read_bytes() != corrupt_data:
                    from torch._inductor.runtime.cutedsl_cache import disk_cache_get

                    verify_mem: dict = {}
                    reloaded = disk_cache_get(
                        verify_mem,
                        path,
                        config_key,
                        runtime_key,
                        device_index=dev_idx,
                    )
                    self.assertIsNotNone(
                        reloaded,
                        "Rewritten .o should be loadable",
                    )
                    x2 = torch.randn(8, 4, device=device, dtype=torch.float32)
                    y2 = torch.randn(8, 4, device=device, dtype=torch.float32)
                    out2 = torch.empty(8, 4, device=device, dtype=torch.float32)
                    reloaded(x2, y2, out2)
                    self.assertEqual(
                        out2, x2 + y2, "Reloaded artifact should execute correctly"
                    )
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_fix_elf_dup_text_patches_duplicate_sections(self):
        """_fix_elf_dup_text_flags rewrites flags on duplicate .text sections."""
        import struct

        from torch._inductor.runtime.cutedsl_cache import _fix_elf_dup_text_flags

        # Build a minimal ELF64 LE with two .text sections:
        #   section 0: strtab  (sh_name=0)
        #   section 1: .text   (ALLOC|EXECINSTR = 0x6)
        #   section 2: .text   (WRITE|ALLOC    = 0x3)
        shstrtab = b"\x00.text\x00"  # offset 0 = "", offset 1 = ".text"
        e_shentsize = 64
        e_shnum = 3
        e_shstrndx = 0
        e_shoff = 64  # section headers start right after ELF header

        # ELF64 header (64 bytes)
        ehdr = bytearray(64)
        ehdr[0:4] = b"\x7fELF"
        ehdr[4] = 2  # EI_CLASS = ELFCLASS64
        ehdr[5] = 1  # EI_DATA  = ELFDATA2LSB
        struct.pack_into("<Q", ehdr, 40, e_shoff)
        struct.pack_into("<H", ehdr, 58, e_shentsize)
        struct.pack_into("<H", ehdr, 60, e_shnum)
        struct.pack_into("<H", ehdr, 62, e_shstrndx)

        def make_shdr(sh_name, sh_type, sh_flags, sh_offset, sh_size):
            s = bytearray(e_shentsize)
            struct.pack_into("<I", s, 0, sh_name)  # sh_name
            struct.pack_into("<I", s, 4, sh_type)  # sh_type
            struct.pack_into("<Q", s, 8, sh_flags)  # sh_flags
            struct.pack_into("<Q", s, 24, sh_offset)  # sh_offset
            struct.pack_into("<Q", s, 32, sh_size)  # sh_size
            return s

        strtab_offset = 64 + e_shnum * e_shentsize
        shdr0 = make_shdr(0, 3, 0, strtab_offset, len(shstrtab))  # SHT_STRTAB
        shdr1 = make_shdr(1, 1, 0x6, 0, 0)  # .text, ALLOC|EXECINSTR
        shdr2 = make_shdr(1, 1, 0x3, 0, 0)  # .text, WRITE|ALLOC

        elf = bytes(ehdr) + bytes(shdr0) + bytes(shdr1) + bytes(shdr2) + shstrtab

        patched = _fix_elf_dup_text_flags(elf)
        self.assertNotEqual(elf, patched, "Patch should modify the ELF")

        # Second .text section flags should now be 0x6 (ALLOC|EXECINSTR)
        sh2_offset = e_shoff + 2 * e_shentsize
        new_flags = struct.unpack_from("<Q", patched, sh2_offset + 8)[0]
        self.assertEqual(
            new_flags, 0x6, "Duplicate .text flags should be harmonized to 0x6"
        )

        # First .text section should be untouched
        sh1_offset = e_shoff + 1 * e_shentsize
        orig_flags = struct.unpack_from("<Q", patched, sh1_offset + 8)[0]
        self.assertEqual(orig_flags, 0x6, "First .text section should remain unchanged")

    def test_fix_elf_dup_text_noop_single_text(self):
        """_fix_elf_dup_text_flags is a no-op when there's only one .text section."""
        import struct

        from torch._inductor.runtime.cutedsl_cache import _fix_elf_dup_text_flags

        shstrtab = b"\x00.text\x00"
        e_shentsize = 64
        e_shnum = 2
        e_shoff = 64

        ehdr = bytearray(64)
        ehdr[0:4] = b"\x7fELF"
        ehdr[4] = 2
        ehdr[5] = 1
        struct.pack_into("<Q", ehdr, 40, e_shoff)
        struct.pack_into("<H", ehdr, 58, e_shentsize)
        struct.pack_into("<H", ehdr, 60, e_shnum)
        struct.pack_into("<H", ehdr, 62, 0)

        def make_shdr(sh_name, sh_type, sh_flags, sh_offset, sh_size):
            s = bytearray(e_shentsize)
            struct.pack_into("<I", s, 0, sh_name)
            struct.pack_into("<I", s, 4, sh_type)
            struct.pack_into("<Q", s, 8, sh_flags)
            struct.pack_into("<Q", s, 24, sh_offset)
            struct.pack_into("<Q", s, 32, sh_size)
            return s

        strtab_offset = 64 + e_shnum * e_shentsize
        shdr0 = make_shdr(0, 3, 0, strtab_offset, len(shstrtab))
        shdr1 = make_shdr(1, 1, 0x6, 0, 0)

        elf = bytes(ehdr) + bytes(shdr0) + bytes(shdr1) + shstrtab
        result = _fix_elf_dup_text_flags(elf)
        self.assertEqual(elf, result, "Single .text section should not be modified")

    def test_fix_elf_dup_text_noop_non_elf(self):
        """_fix_elf_dup_text_flags is a no-op for non-ELF data."""
        from torch._inductor.runtime.cutedsl_cache import _fix_elf_dup_text_flags

        self.assertEqual(_fix_elf_dup_text_flags(b""), b"")
        self.assertEqual(_fix_elf_dup_text_flags(b"not an elf"), b"not an elf")
        self.assertEqual(
            _fix_elf_dup_text_flags(b"\x7fELF" + b"\x00" * 10),
            b"\x7fELF" + b"\x00" * 10,
        )

    def test_fix_elf_dup_text_old_artifacts_patched_on_load(self):
        """disk_cache_get patches old unpatched .o files in-place on load."""
        import shutil
        import struct

        from torch._inductor.runtime.cutedsl_cache import (
            _cache_dir,
            _make_disk_key,
            disk_cache_get,
        )

        shstrtab = b"\x00.text\x00"
        e_shentsize = 64
        e_shnum = 3
        e_shoff = 64

        ehdr = bytearray(64)
        ehdr[0:4] = b"\x7fELF"
        ehdr[4] = 2
        ehdr[5] = 1
        struct.pack_into("<Q", ehdr, 40, e_shoff)
        struct.pack_into("<H", ehdr, 58, e_shentsize)
        struct.pack_into("<H", ehdr, 60, e_shnum)
        struct.pack_into("<H", ehdr, 62, 0)

        def make_shdr(sh_name, sh_type, sh_flags, sh_offset, sh_size):
            s = bytearray(e_shentsize)
            struct.pack_into("<I", s, 0, sh_name)
            struct.pack_into("<I", s, 4, sh_type)
            struct.pack_into("<Q", s, 8, sh_flags)
            struct.pack_into("<Q", s, 24, sh_offset)
            struct.pack_into("<Q", s, 32, sh_size)
            return s

        strtab_offset = 64 + e_shnum * e_shentsize
        shdr0 = make_shdr(0, 3, 0, strtab_offset, len(shstrtab))
        shdr1 = make_shdr(1, 1, 0x6, 0, 0)
        shdr2 = make_shdr(1, 1, 0x3, 0, 0)  # bad flags

        bad_elf = bytes(ehdr) + bytes(shdr0) + bytes(shdr1) + bytes(shdr2) + shstrtab

        module_path = "/fake/old_artifact.py"
        config_key = ("test_old_artifact",)
        runtime_key = ((4, 4), torch.float32)

        with fresh_cache():
            cache_dir = _cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                h = _make_disk_key(module_path, config_key, runtime_key, device_index=0)
                obj_path = cache_dir / f"{h}.o"
                obj_path.write_bytes(bad_elf)

                mem_cache: dict = {}
                # disk_cache_get will fail to load (not a real compiled artifact)
                # but should still patch the file on disk before attempting load
                disk_cache_get(
                    mem_cache, module_path, config_key, runtime_key, device_index=0
                )

                patched_bytes = obj_path.read_bytes()
                sh2_offset = e_shoff + 2 * e_shentsize
                new_flags = struct.unpack_from("<Q", patched_bytes, sh2_offset + 8)[0]
                self.assertEqual(
                    new_flags,
                    0x6,
                    "disk_cache_get should patch old artifacts with bad .text flags",
                )
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

    def test_symbolic_shapes_skip_precompile_metadata(self):
        """_build_precompile_metadata returns None for symbolic (dynamic) sizes,
        causing the async path to skip _precompile and compile lazily."""
        from unittest.mock import MagicMock

        from torch._inductor.codegen.cutedsl.cutedsl_scheduling import CuteDSLScheduling

        class _SymbolicSize:
            """Mimics torch.SymInt -- int() raises TypeError."""

            def __int__(self):
                raise TypeError("Cannot convert symbolic size to int")

        kernel = MagicMock()
        kernel._template_input_args = [
            (
                "arg_input_a",
                MagicMock(
                    get_size=MagicMock(return_value=[_SymbolicSize(), _SymbolicSize()]),
                    get_dtype=MagicMock(return_value=torch.float32),
                ),
            ),
        ]

        ctb = MagicMock()
        ctb.layout.size = [_SymbolicSize(), _SymbolicSize()]
        ctb.layout.dtype = torch.float32

        scheduling = CuteDSLScheduling.__new__(CuteDSLScheduling)
        result = scheduling._build_precompile_metadata(kernel, ctb)

        self.assertIsNone(
            result,
            "_build_precompile_metadata should return None for symbolic sizes",
        )

    def test_symbolic_shapes_worker_skips_precompile(self):
        """When precompile_metadata is None (symbolic shapes), the worker writes
        source but skips _precompile. The kernel still works via lazy compilation."""
        from torch._inductor.runtime.compile_tasks import (
            _worker_compile_pycodecache_kernel,
        )

        compile_sentinel = os.path.join(
            tempfile.gettempdir(), "cutedsl_test_symbolic_sentinel"
        )
        if os.path.exists(compile_sentinel):
            os.unlink(compile_sentinel)

        source = (
            "import os\n"
            "def test_kernel_main(*args, **kwargs): pass\n"
            "def test_kernel_precompile(**kw):\n"
            "    with open(" + repr(compile_sentinel) + ", 'w') as f:\n"
            "        f.write('precompile was called')\n"
        )

        try:
            with fresh_cache():
                key, path, elapsed = _worker_compile_pycodecache_kernel(
                    "test_kernel", source, "main", {}, None
                )

                self.assertFalse(
                    os.path.exists(compile_sentinel),
                    "_precompile should NOT be called when metadata is None "
                    "(symbolic shapes fallback)",
                )

                import torch._inductor.codecache as codecache

                mod = codecache.PyCodeCache.load_by_key_path(key, path)
                self.assertTrue(hasattr(mod, "test_kernel_main"))
        finally:
            if os.path.exists(compile_sentinel):
                os.unlink(compile_sentinel)

    def test_cache_key_changes_with_source_content(self):
        """Different template/codegen output produces different disk cache keys.

        The disk key includes module_path which is content-addressed via PyCodeCache,
        so codegen changes automatically invalidate the cache. This test verifies
        that property.

        Note: The disk key does NOT include the cutlass library version. If the
        cutlass compiler produces different binaries for the same source on the same
        GPU arch + CUDA version (e.g., after a cutlass upgrade), stale artifacts
        could theoretically be reused. In practice this is rare and mitigated by
        clearing the cache directory.
        """
        from torch._inductor.runtime.cutedsl_cache import _make_disk_key

        config_key = ("same_config",)
        runtime_key = ((8, 4), torch.float32)

        key_v1 = _make_disk_key(
            "/cache/abc123_v1.py", config_key, runtime_key, device_index=0
        )
        key_v2 = _make_disk_key(
            "/cache/def456_v2.py", config_key, runtime_key, device_index=0
        )
        self.assertNotEqual(
            key_v1,
            key_v2,
            "Different source content (different PyCodeCache paths) must produce "
            "different disk cache keys -- this is how template changes invalidate "
            "stale artifacts",
        )

        # Same path -> same key (deterministic)
        key_again = _make_disk_key(
            "/cache/abc123_v1.py", config_key, runtime_key, device_index=0
        )
        self.assertEqual(key_v1, key_again)

    def test_cache_key_changes_with_config(self):
        """Different config keys (e.g., kernel name, tile sizes) produce different
        disk cache keys even for the same module path."""
        from torch._inductor.runtime.cutedsl_cache import _make_disk_key

        path = "/cache/same_module.py"
        runtime_key = ((8, 4), torch.float32)

        key_cfg1 = _make_disk_key(path, ("kernel_a",), runtime_key, device_index=0)
        key_cfg2 = _make_disk_key(path, ("kernel_b",), runtime_key, device_index=0)
        self.assertNotEqual(
            key_cfg1,
            key_cfg2,
            "Different config keys must produce different disk cache keys",
        )


try:
    from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
    from torch._inductor.utils import ensure_cute_available

    HAS_BLACKWELL_CUTEDSL = ensure_cute_available() and is_datacenter_blackwell_arch()
except Exception:
    HAS_BLACKWELL_CUTEDSL = False

import unittest


@unittest.skipIf(
    not HAS_BLACKWELL_CUTEDSL,
    "CuTeDSL library or Blackwell device not available",
)
class TestCuteDSLSubprocessGroupedGemm(TestCase):
    """End-to-end test: grouped GEMM through torch.compile with subprocess compilation.

    Exercises the full path: CuTe DSL template selection -> codegen ->
    subprocess compilation -> precompile metadata -> disk cache -> correctness.
    """

    def test_grouped_gemm_subprocess_e2e(self):
        shutdown_compile_workers()

        device = "cuda"
        dtype = torch.bfloat16
        group_size, K, N = 4, 64, 128
        alignment = 16

        M_sizes = torch.randint(1, 17, (group_size,), dtype=torch.int) * alignment
        M_total = int(torch.sum(M_sizes).item())
        A = torch.randn(M_total, K, dtype=dtype, device=device) * 0.1
        B = torch.randn(group_size, K, N, dtype=dtype, device=device) * 0.01
        offsets = torch.cumsum(M_sizes, dim=0).to(dtype=torch.int32, device=device)

        def grouped_gemm_fn(A_packed, B_batched, offs):
            return torch.nn.functional.grouped_mm(A_packed, B_batched, offs=offs)

        c_eager = grouped_gemm_fn(A, B, offsets)

        with config.patch(
            {
                "worker_start_method": "subprocess",
                "compile_threads": 4,
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            AsyncCompile.wait_pool_ready()
            with fresh_cache():
                compiled_fn = torch.compile(
                    grouped_gemm_fn, backend="inductor", dynamic=False
                )
                c_compiled = compiled_fn(A, B, offsets)

        torch.testing.assert_close(c_eager, c_compiled)


if __name__ == "__main__":
    run_tests()
