# Owner(s): ["module: inductor"]

import os
import re
import subprocess
import sys
import tempfile
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import CompiledArtifact, config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _code_for(fn, *args, **config_kwargs):
    torch._dynamo.reset()
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn), *args)
    return result, "\n".join(codes)


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _cond_softmax(x):
    # Kernels in both the root and the branch subgraphs, whose text the root splices in.
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


def _compiled_in_this_process(*args, **kwargs):
    raise AssertionError("a kernel was compiled in the compiling process")


class TestModuleLevelKernels(TestCase):
    @requires_cuda_and_triton
    @parametrize("fn", [_softmax, _cond_softmax])
    def test_kernels_are_defined_at_module_level(self, fn):
        x = torch.randn(64, 128, device="cuda")
        result, code = _code_for(fn, x, **{"triton.module_level_kernels": True})
        self.assertEqual(result, fn(x))
        self.assertNotIn("async_compile.triton", code)
        self.assertTrue(re.search(r"^@triton_heuristics\.\w+\(", code, re.MULTILINE))
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertTrue(kernels, code)
        self.assertEqual(len(kernels), len(set(kernels)), kernels)
        # Unlike readable_wrapper, the provenance comment keeps the kernel's cache file.
        self.assertIn("# kernel path:", code)

    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        self.assertIn("async_compile.triton(", code)

    @requires_cuda_and_triton
    @config.patch({"compile_threads": 2, "triton.unique_kernel_names": False})
    def test_kernels_without_unique_names(self):
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        x = torch.randn(64, 128, device="cuda")
        flags = {"triton.module_level_kernels": True}
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, code = _code_for(_cond_softmax, x, **flags)
        self.assertEqual(result, _cond_softmax(x))
        self.assertNotIn("def triton_(", code)
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertGreater(len(kernels), 1, code)
        self.assertEqual(len(kernels), len(set(kernels)), kernels)

    @requires_cuda_and_triton
    @parametrize(
        "patch",
        [
            {"benchmark_kernel": True},
            {"benchmark_combo_kernel": True, "combo_kernels": True},
        ],
    )
    def test_kernel_benchmark_harness(self, patch):
        def fn(x, y):
            # Independent pointwise kernels, which combo_kernels fuses into one.
            return x.sin() * 2, y.cos() + 1

        x, y = torch.randn(64, 128, device="cuda"), torch.randn(32, device="cuda")
        flags = {"triton.module_level_kernels": True, **patch}
        PyCodeCache.cache_clear()
        result, code = _code_for(fn, x, y, **flags)
        self.assertEqual(result, fn(x, y))
        # Only the wrapper's own harness is at module level; each kernel's stays in the
        # module the kernel is compiled from, where benchmark_all_kernels finds it.
        self.assertEqual(code.count("__main__"), 1, code)
        self.assertNotIn("def get_args", code.split("def call(")[0])
        kernels = [m for m in PyCodeCache.modules if hasattr(m, "get_args")]
        self.assertTrue(kernels)
        # Run as a script, the wrapper compiles its kernels from its own defs, so
        # benchmark_all_kernels has to find their harnesses from those.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "module.py")
            with open(path, "w") as f:
                f.write(code)
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
            cmd = [sys.executable, path, "-kc"]
            out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
        defs = re.findall(r"^def triton_\w+\(", code, re.MULTILINE)
        # -c prints a line per config, which needs each kernel precompiled.
        self.assertGreaterEqual(out.decode().count("GB/s"), len(defs), out.decode())

    @requires_cuda_and_triton
    @config.patch(compile_threads=2)
    def test_kernels_compile_on_the_worker_pool(self):
        # The pool's workers are separate processes, so the patch only catches a compile
        # in this one: every kernel has to arrive already compiled.
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        x = torch.randn(64, 128, device="cuda")
        counters.clear()
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, code = _code_for(
                _cond_softmax, x, **{"triton.module_level_kernels": True}
            )
        self.assertEqual(result, _cond_softmax(x))
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertEqual(counters["inductor"]["async_compile_cache_hit"], len(kernels))

        (loaded,) = [m for m in PyCodeCache.modules if hasattr(m, kernels[0])]
        pooled = {k: getattr(loaded, k).kernel_hash for k in kernels}
        self.assertEqual(len(set(pooled.values())), len(kernels), pooled)

        # Anyone else's load of the module, such as running a copy of it, builds its
        # kernels from the defs in it, so a hand edit there takes effect. They compile
        # at async_compile.wait, and key the autotune cache on the same per-kernel name
        # as the pool's kernels rather than on the module they share.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "module.py")
            with open(path, "w") as f:
                f.write(code)
            ns = {"__file__": path, "__name__": "_module_level_kernels"}
            exec(compile(code, path, "exec"), ns)
        for k in kernels:
            self.assertEqual(ns[k].fn.fn.__code__.co_filename, path)
            self.assertTrue(ns[k].launchers, k)
            self.assertEqual(ns[k].kernel_hash, pooled[k])
        self.assertEqual(ns["call"]([x])[0], result)

    @requires_cuda_and_triton
    @config.patch(compile_threads=2)
    def test_kernels_reloaded_from_their_source_while_the_wrapper_loads(self):
        # dynamic_scale_rblock and coordesc tuning reload a pool-compiled kernel from
        # its own module in this process, which runs while the wrapper module loads.
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        scale_rblock = CachingAutotuner._dynamic_scale_rblock

        def reload_first(self):
            self._ensure_kernel_loaded()
            scale_rblock(self)

        x = torch.randn(64, 128, device="cuda")
        flags = {"triton.module_level_kernels": True}
        with mock.patch.object(CachingAutotuner, "_dynamic_scale_rblock", reload_first):
            result, _ = _code_for(_softmax, x, **flags)
        self.assertEqual(result, _softmax(x))

    @requires_cuda_and_triton
    @config.patch(compile_threads=2, fx_graph_cache=True)
    def test_cached_kernels_compile_on_the_worker_pool(self):
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        # torch.cond bypasses the FX graph cache, so this uses a graph without one.
        x = torch.randn(64, 128, device="cuda")
        flags = {"triton.module_level_kernels": True}
        _code_for(_softmax, x, **flags)
        counters.clear()
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, _ = _code_for(_softmax, x, **flags)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(result, _softmax(x))

    @requires_cuda_and_triton
    def test_hand_edit_to_a_saved_module_takes_effect(self):
        x = torch.ones(4, device="cuda")
        gms = []
        torch.compile(lambda t: t * 2, backend=lambda gm, _: gms.append(gm) or gm)(x)
        flags = {"triton.module_level_kernels": True, "fx_graph_cache": True}
        with tempfile.TemporaryDirectory() as d, config.patch(flags):
            with fresh_cache():
                artifact = torch._inductor.standalone_compile(gms[0], [x])
                self.assertEqual(artifact(x)[0], x * 2)
                artifact.save(path=d, format="unpacked")
            paths = [os.path.join(r, n) for r, _, ns in os.walk(d) for n in ns]
            for path in paths:
                if path.endswith(".py"):
                    with open(path) as f:
                        code = f.read()
                    if "def call(" in code:
                        with open(path, "w") as f:
                            f.write(code.replace("2.0, tl.float32", "8.0, tl.float32"))
            counters.clear()
            with fresh_cache():
                loaded = CompiledArtifact.load(path=d, format="unpacked")
                self.assertEqual(loaded(x)[0], x * 8)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @requires_cuda_and_triton
    @parametrize("wrapper", ["cpp_wrapper", "fx_wrapper"])
    def test_wrappers_that_keep_kernels_as_strings(self, wrapper):
        # Both consume each kernel's async_compile.triton(...) source themselves.
        x = torch.randn(64, 128, device="cuda")
        torch._dynamo.reset()
        with config.patch({wrapper: True, "triton.module_level_kernels": True}):
            result = torch.compile(_softmax)(x)
        self.assertEqual(result, _softmax(x))

    @requires_cuda_and_triton
    @config.patch(compile_threads=1)
    def test_kernels_compile_serially_without_a_pool(self):
        x = torch.randn(64, 128, device="cuda")
        result, _ = _code_for(_cond_softmax, x, **{"triton.module_level_kernels": True})
        self.assertEqual(result, _cond_softmax(x))


instantiate_parametrized_tests(TestModuleLevelKernels)


if __name__ == "__main__":
    run_tests()
