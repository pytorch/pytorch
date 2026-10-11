# Owner(s): ["module: inductor"]

import itertools
import os
import re
import tempfile
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor import CompiledArtifact, config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    collect_defined_kernels,
    fresh_cache,
    is_big_gpu,
    run_and_get_code,
)
from torch.nn.attention.flex_attention import flex_attention
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


def _run_from_file(code, args):
    # A module-level kernel resolves its own source through its file, so it cannot be
    # exec'd from a bare string.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "module.py")
        with open(path, "w") as f:
            f.write(code)
        ns = {"__file__": path, "__name__": "_module_level_kernels"}
        exec(compile(code, path, "exec"), ns)
        return ns["call"](args)


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _double(x):
    return x * 2


def _cond_softmax(x):
    # Kernels in both the root and the branch subgraphs, whose text the root splices in.
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


def _compiled_in_this_process(*args, **kwargs):
    raise AssertionError("a kernel was compiled in the compiling process")


class TestModuleLevelKernels(TestCase):
    def setUp(self):
        super().setUp()
        # Module-level kernels are not the default yet.
        patcher = mock.patch.object(
            PythonWrapperCodegen, "defines_triton_kernels_as_code", lambda self: True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @requires_cuda_and_triton
    @parametrize("fn", [_softmax, _cond_softmax])
    def test_kernels_are_defined_at_module_level(self, fn):
        x = torch.randn(64, 128, device="cuda")
        result, code = _code_for(fn, x)
        self.assertEqual(result, fn(x))
        self.assertNotIn("async_compile.triton", code)
        self.assertTrue(re.search(r"^@triton_heuristics\.\w+\(", code, re.MULTILINE))
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertTrue(kernels, code)
        self.assertEqual(len(kernels), len(set(kernels)), kernels)
        self.assertIn("# kernel path:", code)

    @requires_cuda_and_triton
    @parametrize("autotune_at_compile_time", [None, True])
    def test_module_runs_from_its_file(self, autotune_at_compile_time):
        x = torch.randn(64, 128, device="cuda")
        cfg = {"triton.autotune_at_compile_time": autotune_at_compile_time}
        result, code = _code_for(_softmax, x, **cfg)
        self.assertEqual(result, _softmax(x))
        # Only the compile-time autotune block, which execs its kernels, keeps the
        # string form.
        expected_strings = 1 if autotune_at_compile_time else 0
        self.assertEqual(code.count("= async_compile.triton("), expected_strings)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_cuda_and_triton
    @parametrize(
        "cfg",
        [
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"},
            {"combo_kernels": True},
        ],
        name_fn=lambda cfg: next(iter(cfg)),
    )
    def test_template_and_combo_kernels(self, cfg):
        # Both reach emit_triton_kernel_definition by their own route: a Triton matmul
        # template, and a combo kernel with its module-level device functions.
        if "max_autotune" in cfg and not is_big_gpu():
            self.skipTest("Triton GEMM templates need a big GPU")

        def fn(a, b, x, y):
            return a @ b, x.sin(), y.cos()

        a, b = torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda")
        x, y = torch.randn(128, device="cuda"), torch.randn(96, device="cuda")
        result, code = _code_for(fn, a, b, x, y, **cfg)
        self.assertNotIn("async_compile.triton", code)
        self.assertIn("triton_tem_" if "max_autotune" in cfg else "pid_offset", code)
        expected = fn(a, b, x, y)
        self.assertEqual(result, expected)
        self.assertEqual(_run_from_file(code, [a, b, x, y]), expected)

    @requires_cuda_and_triton
    @config.patch({"triton.multi_kernel": 1})
    def test_multi_kernel(self):
        # multi_kernel_N = async_compile.multi_kernel(..., [<module-level kernels>])
        x = torch.rand(2, 1024, device="cuda")
        result, code = _code_for(torch.softmax, x, -1)
        self.assertIn("= async_compile.multi_kernel(", code)
        self.assertNotIn("async_compile.triton", code)
        self.assertEqual(result, torch.softmax(x, -1))
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_cuda_and_triton
    def test_collect_defined_kernels(self):
        # Its define_kernel wrapper must forward standalone= and autotune_body=.
        kernels = []
        with collect_defined_kernels(kernels):
            _code_for(_softmax, torch.randn(64, 128, device="cuda"))
        self.assertTrue(kernels)
        self.assertTrue(all("tl.store" in k for k in kernels), kernels)
        self.assertFalse(any("async_compile.triton(" in k for k in kernels), kernels)

    @requires_cuda_and_triton
    @parametrize("case", ["scan", "flex_attention"])
    def test_same_named_helpers_do_not_shadow(self, case):
        # Scan combine_fns are named by op sequence, and flex attention's template
        # defines forward_inner & co. Both kernels here define the same names with
        # different bodies.
        if case == "scan":

            def fn(x, y):
                kw = {"dim": 0, "combine_mode": "pointwise"}
                a = associative_scan(lambda p, q: p + q + 1, x, **kw)
                return a, associative_scan(lambda p, q: p + q + 2, y, **kw)

            args = [torch.randn(64, device="cuda"), torch.randn(64, device="cuda")]
        else:

            def fn(x):
                a = flex_attention(x, x, x, score_mod=lambda s, b, h, m, n: s * 2)
                return a, flex_attention(x, x, x, score_mod=lambda s, b, h, m, n: s + m)

            args = [torch.randn(1, 2, 128, 64, device="cuda")]
        result, code = _code_for(fn, *args)
        defs = re.findall(r"^def (\w+)\(", code, re.MULTILINE)
        self.assertEqual(len(defs), len(set(defs)), defs)
        expected = fn(*args)
        tol = {"atol": 2e-2, "rtol": 2e-2}
        self.assertEqual(result, expected, **tol)
        self.assertEqual(_run_from_file(code, args), expected, **tol)

    @requires_cuda_and_triton
    @config.patch(compile_threads=2)
    def test_kernels_compile_on_the_worker_pool(self):
        # The pool's workers are separate processes, so the patch only catches a compile
        # in this one: every kernel has to arrive already compiled.
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        x = torch.randn(64, 128, device="cuda")
        counters.clear()
        PyCodeCache.cache_clear()
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, code = _code_for(_cond_softmax, x)
        self.assertEqual(result, _cond_softmax(x))
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertEqual(counters["inductor"]["async_compile_cache_hit"], len(kernels))

        mods = PyCodeCache.modules
        (loaded,) = [m for m in mods if hasattr(m, kernels[0]) and hasattr(m, "call")]
        pooled = {k: getattr(loaded, k).kernel_hash for k in kernels}
        self.assertEqual(len(set(pooled.values())), len(kernels), pooled)

        # Anyone else's load of the module, such as running a copy of it, builds its
        # kernels from the defs in it, so a hand edit there takes effect. They key the
        # autotune cache on the same per-kernel name as the pool's kernels rather than
        # on the module they share.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "module.py")
            with open(path, "w") as f:
                f.write(code)
            ns = {"__file__": path, "__name__": "_module_level_kernels"}
            exec(compile(code, path, "exec"), ns)
        for k in kernels:
            self.assertEqual(ns[k].fn.fn.__code__.co_filename, path)
            self.assertEqual(ns[k].kernel_hash, pooled[k])
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            with self.assertRaisesRegex(Exception, "compiling process"):
                ns["call"]([x])

    @requires_cuda_and_triton
    @config.patch(compile_threads=2)
    @parametrize("case", ["template", "foreach", "combo"])
    def test_template_and_combo_kernels_compile_on_the_worker_pool(self, case):
        # Their decorators must key inductor_meta["kernel_name"] on the def's name too.
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        if case == "template":

            def fn(x):
                return flex_attention(x, x, x, score_mod=lambda s, b, h, m, n: s * 2)

            args = [torch.randn(1, 2, 128, 64, device="cuda")]
        elif case == "foreach":

            def fn(x, y):
                return torch._foreach_add([x, y], [y, x])

            args = [torch.randn(128, device="cuda"), torch.randn(128, device="cuda")]
        else:

            def fn(x, y):
                return x.sin(), y.cos()

            args = [torch.randn(128, device="cuda"), torch.randn(96, device="cuda")]
        counters.clear()
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, code = _code_for(fn, *args, combo_kernels=case == "combo")
        self.assertEqual(result, fn(*args), atol=2e-2, rtol=2e-2)
        marker = {"template": "def triton_tem_", "foreach": "def triton_for_"}
        self.assertIn(marker.get(case, "pid_offset"), code)
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertEqual(counters["inductor"]["async_compile_cache_hit"], len(kernels))

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
        with mock.patch.object(CachingAutotuner, "_dynamic_scale_rblock", reload_first):
            result, _ = _code_for(_softmax, x)
        self.assertEqual(result, _softmax(x))

    @requires_cuda_and_triton
    @config.patch(compile_threads=1)
    def test_kernels_compile_serially_without_a_pool(self):
        x = torch.randn(64, 128, device="cuda")
        result, _ = _code_for(_cond_softmax, x)
        self.assertEqual(result, _cond_softmax(x))

    @requires_cuda_and_triton
    @config.patch(compile_threads=2, fx_graph_cache=True)
    def test_cached_kernels_compile_on_the_worker_pool(self):
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        # torch.cond bypasses the FX graph cache, so this uses a graph without one.
        x = torch.randn(64, 128, device="cuda")
        _code_for(_softmax, x)
        counters.clear()
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, _ = _code_for(_softmax, x)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(result, _softmax(x))

    @requires_cuda_and_triton
    def test_hand_edit_to_a_saved_module_takes_effect(self):
        x = torch.ones(4, device="cuda")
        gms = []
        torch.compile(lambda t: t * 2, backend=lambda gm, _: gms.append(gm) or gm)(x)
        with tempfile.TemporaryDirectory() as d, config.patch(fx_graph_cache=True):
            with fresh_cache():
                artifact = torch._inductor.standalone_compile(gms[0], [x])
                self.assertEqual(artifact(x)[0], x * 2)
                artifact.save(path=d, format="unpacked")
            paths = [os.path.join(r, n) for r, _, ns in os.walk(d) for n in ns]
            edited = 0
            for path in paths:
                if path.endswith(".py"):
                    with open(path) as f:
                        code = f.read()
                    if "def call(" in code and "2.0, tl.float32" in code:
                        with open(path, "w") as f:
                            f.write(code.replace("2.0, tl.float32", "8.0, tl.float32"))
                        edited += 1
            self.assertEqual(edited, 1)
            counters.clear()
            with fresh_cache():
                loaded = CompiledArtifact.load(path=d, format="unpacked")
                self.assertEqual(loaded(x)[0], x * 8)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @requires_cuda_and_triton
    @config.patch(fx_graph_cache=False)
    def test_hand_edit_to_a_cached_module_survives_a_recompile(self):
        x = torch.ones(4, device="cuda")
        # A fresh AOT counter makes the recompile emit the same module, at the same path.
        counter = mock.patch(
            "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
        )
        with counter:
            _, code = _code_for(_double, x)
        _, path = PyCodeCache.write(code)
        self.assertIn("2.0, tl.float32", code)
        with open(path, "w") as f:
            f.write(code.replace("2.0, tl.float32", "8.0, tl.float32"))
        # The first recompile misses the FX graph cache and caches the edited module;
        # the second loads that entry.
        for hits in (0, 1):
            PyCodeCache.cache_clear()
            counters.clear()
            with counter, config.patch(fx_graph_cache=True):
                result, _ = _code_for(_double, x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], hits)
            self.assertEqual(result, x * 8)

    @requires_cuda_and_triton
    @config.patch({"compile_threads": 2, "triton.unique_kernel_names": False})
    def test_kernels_without_unique_names(self):
        self.assertTrue(AsyncCompile.wait_process_pool_ready())
        x = torch.randn(64, 128, device="cuda")
        with mock.patch.object(
            CachingAutotuner, "_precompile_config", _compiled_in_this_process
        ):
            result, code = _code_for(_cond_softmax, x)
        self.assertEqual(result, _cond_softmax(x))
        # The pool built the pre-rename sources; this compiles the renamed defs.
        self.assertEqual(_run_from_file(code, [x])[0], result)
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
            {
                "benchmark_kernel": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        ],
        name_fn=lambda p: "_".join(k for k, v in p.items() if v is True),
    )
    # In-process compiles, so each kernel's module is loaded into PyCodeCache here.
    @config.patch(compile_threads=1)
    def test_kernel_benchmark_harness(self, patch):
        if "max_autotune" in patch and not is_big_gpu():
            self.skipTest("Triton GEMM templates need a big GPU")

        def fn(a, b, x, y):
            # Independent pointwise kernels, which combo_kernels fuses into one, and a
            # matmul, which max_autotune emits through the template path.
            return a @ b, x.sin() * 2, y.cos() + 1

        a, b = torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda")
        x, y = torch.randn(64, 128, device="cuda"), torch.randn(32, device="cuda")
        PyCodeCache.cache_clear()
        result, code = _code_for(fn, a, b, x, y, **patch)
        self.assertEqual(result, fn(a, b, x, y))
        self.assertEqual(_run_from_file(code, [a, b, x, y]), result)
        if "max_autotune" in patch:
            self.assertIn("triton_tem_", code)
        if "combo_kernels" in patch:
            self.assertIn("pid_offset", code)
        # Only the wrapper's own harness is at module level; each kernel's stays in the
        # module the kernel is compiled from, where benchmark_all_kernels finds it.
        self.assertEqual(code.count("__main__"), 1, code)
        self.assertNotIn("def get_args", code.split("def call(")[0])
        # The wrapper defines get_args too; only a kernel's harness has this.
        mods = [m for m in PyCodeCache.modules if hasattr(m, "benchmark_all_configs")]
        self.assertTrue(mods)


class TestDefaultWrapper(TestCase):
    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        self.assertIn("async_compile.triton(", code)


instantiate_parametrized_tests(TestModuleLevelKernels)


if __name__ == "__main__":
    run_tests()
