# Owner(s): ["module: inductor"]

import ast
import importlib
import itertools
import os
import re
import subprocess
import sys
import tempfile
import threading
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import CompiledArtifact, config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.wrapper import (
    _rename_kernel_module_globals,
    PythonWrapperCodegen,
)
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


# Two user kernels from different modules that bind the same top-level names to
# different values. The kernel's SCALE parameter (its numel) shadows the global SCALE,
# which only the helper reads.
_SCALE_MODULE = """
import triton
import triton.language as tl
from triton.language import {op} as op

SCALE = tl.constexpr({scale})


@triton.jit
def scale(x):
    return op(x) * SCALE


@triton.jit
def scale_kernel(in_ptr, out_ptr, SCALE, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SCALE
    tl.store(out_ptr + offs, scale(tl.load(in_ptr + offs, mask=mask)), mask=mask)
"""

# A kernel with no helpers or globals of its own, only an imported alias. The kernels
# are named apart: Triton keys its cache on the source, which does not cover `op`.
_OP_MODULE = """
import triton
import triton.language as tl
from triton.language import {op} as op


@triton.jit
def {op}_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, op(tl.load(in_ptr + offs, mask=mask)), mask=mask)
"""


# A user kernel whose source holds a backslash escape.
_DOC_MODULE = r'''
import triton
import triton.language as tl


@triton.jit
def doc_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Adds one.\n"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask) + 1, mask=mask)
'''


def _import_kernels(d, template, kernels):
    for name, fields in kernels.items():
        with open(os.path.join(d, f"{name}.py"), "w") as f:
            f.write(template.format(**fields))
    sys.path.insert(0, d)
    try:
        return [importlib.import_module(name) for name in kernels]
    finally:
        sys.path.remove(d)


def _compiled_in_this_process(*args, **kwargs):
    raise AssertionError("a kernel was compiled in the compiling process")


class TestRenameKernelModuleGlobals(TestCase):
    def test_renames_only_what_refers_to_a_kernel_global(self):
        # Parameters and locals that shadow a global keep their names, as do the
        # header imports above the first def; the non-ASCII text checks byte offsets.
        src = """\
import triton
X = 1
from triton.language import exp as exp

@triton.jit
def k(a, SCALE):
    b = ("\u00e9", SCALE + X)
    return helper(exp(b))

@triton.jit
def helper(x):
    X = 2
    return x * SCALE * X + op(x)

SCALE = 4
from triton.language import floor as op
import triton.language.math
"""
        expected = """\
import triton
X_k_0 = 1
from triton.language import exp as exp

@triton.jit
def k_0(a, SCALE):
    b = ("\u00e9", SCALE + X_k_0)
    return helper_k_0(exp(b))

@triton.jit
def helper_k_0(x):
    X = 2
    return x * SCALE_k_0 * X + op_k_0(x)

SCALE_k_0 = 4
from triton.language import floor as op_k_0
import triton.language.math
"""
        self.assertEqual(_rename_kernel_module_globals(src, "k_0", "k"), expected)


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
        # kernels from the defs in it, so a hand edit there takes effect. They compile
        # at async_compile.wait, and key the autotune cache on the same per-kernel name
        # as the pool's kernels rather than on the module they share.
        loaded_on = []
        make_launchers = CachingAutotuner._make_launchers

        def _record_thread(autotuner):
            loaded_on.append(threading.current_thread())
            make_launchers(autotuner)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "module.py")
            with open(path, "w") as f:
                f.write(code)
            ns = {"__file__": path, "__name__": "_module_level_kernels"}
            with mock.patch.object(CachingAutotuner, "_make_launchers", _record_thread):
                exec(compile(code, path, "exec"), ns)
        for k in kernels:
            self.assertEqual(ns[k].fn.fn.__code__.co_filename, path)
            self.assertTrue(ns[k].launchers, k)
            self.assertEqual(ns[k].kernel_hash, pooled[k])
        # Only the Triton compile runs on the pool's threads.
        self.assertEqual(set(loaded_on), {threading.main_thread()})
        self.assertEqual(ns["call"]([x])[0], result)

    @requires_cuda_and_triton
    @parametrize("case", ["interpreter", "one_thread"])
    def test_copied_module_kernels_stay_lazy(self, case):
        # async_compile.wait leaves the defs' kernels uncompiled with compile_threads=1,
        # and under the interpreter, which returns even string-form kernels uncompiled.
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        threads = 1 if case == "one_thread" else 2
        env = {"TRITON_INTERPRET": "1"} if case == "interpreter" else {}
        with config.patch(compile_threads=threads), mock.patch.dict(os.environ, env):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "module.py")
                with open(path, "w") as f:
                    f.write(code)
                ns = {"__file__": path, "__name__": "_module_level_kernels"}
                exec(compile(code, path, "exec"), ns)
        self.assertTrue(kernels)
        for k in kernels:
            self.assertFalse(ns[k].launchers, k)

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

        # Run as a script, the wrapper compiles its kernels from its own defs, so
        # benchmark_all_kernels has to find their harnesses from those.
        def run_script(code, flag):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "module.py")
                with open(path, "w") as f:
                    f.write(code)
                env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
                cmd = [sys.executable, path, flag]
                out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
            return out.decode()

        defs = re.findall(r"^def triton_\w+\(", code, re.MULTILINE)
        keys = ast.literal_eval(re.search(r"kernel_modules=(\[.*?\])", code).group(1))
        self.assertEqual(len(keys), len(defs), keys)
        # -c prints each kernel's key, then a line per config, which needs the kernel
        # precompiled.
        out = run_script(code, "-kc")
        for key in keys:
            self.assertRegex(out, rf"{key[:10]}\n  .*GB/s")
        # A module missing from the cache dir is skipped and the rest still run.
        missing = "z" * len(keys[0])
        code = code.replace("kernel_modules=[", f"kernel_modules=[{missing!r}, ")
        out = run_script(code, "-k")
        self.assertIn(f"Skipping kernel module {missing}", out)
        for key in keys:
            self.assertRegex(out, rf"{key[:10]} .*GB/s")

    @requires_cuda_and_triton
    @parametrize("wrapper", ["cpp_wrapper", "fx_wrapper"])
    @parametrize("fn", [_softmax, _cond_softmax])
    def test_wrappers_that_keep_kernels_as_strings(self, wrapper, fn):
        # Both consume each kernel's async_compile.triton(...) source themselves.
        x = torch.randn(64, 128, device="cuda")
        torch._dynamo.reset()
        with config.patch({wrapper: True}):
            result = torch.compile(fn)(x)
        self.assertEqual(result, fn(x))

    @requires_cuda_and_triton
    def test_user_defined_kernels(self):
        with tempfile.TemporaryDirectory() as d:
            modules = {
                "_kd_scale_a": {"op": "abs", "scale": 2.0},
                "_kd_scale_b": {"op": "floor", "scale": 3.0},
            }
            mod_a, mod_b = _import_kernels(d, _SCALE_MODULE, modules)
            kernel_a, kernel_b = mod_a.scale_kernel, mod_b.scale_kernel

            def fn(x):
                a, b = torch.empty_like(x), torch.empty_like(x)
                kernel_a[(4,)](x, a, x.numel(), BLOCK=64)
                kernel_b[(4,)](x, b, x.numel(), BLOCK=64)
                return a + 1, b + 1

            x = torch.randn(256, device="cuda")
            expected = (x.abs() * 2 + 1, x.floor() * 3 + 1)
            result, code = _code_for(fn, x)
            self.assertEqual(result, expected)
            self.assertNotIn("async_compile.triton", code)
            kernels = re.findall(r"^def scale_kernel_\d\(", code, re.MULTILINE)
            self.assertEqual(len(kernels), 2, code)
            # The pool compiles each kernel from its own module; only a standalone run
            # of the wrapper resolves the kernels' globals in the one shared namespace.
            self.assertEqual(tuple(_run_from_file(code, [x])), expected)

    @requires_cuda_and_triton
    def test_user_defined_kernels_that_import_the_same_alias(self):
        with tempfile.TemporaryDirectory() as d:
            modules = {"_kd_op_a": {"op": "abs"}, "_kd_op_b": {"op": "floor"}}
            mod_a, mod_b = _import_kernels(d, _OP_MODULE, modules)
            kernel_a, kernel_b = mod_a.abs_kernel, mod_b.floor_kernel

            def fn(x):
                a, b = torch.empty_like(x), torch.empty_like(x)
                kernel_a[(4,)](x, a, x.numel(), BLOCK=64)
                kernel_b[(4,)](x, b, x.numel(), BLOCK=64)
                return a + 1, b + 1

            x = torch.randn(256, device="cuda")
            expected = (x.abs() + 1, x.floor() + 1)
            result, code = _code_for(fn, x)
            self.assertEqual(result, expected)
            self.assertNotIn("async_compile.triton", code)
            for op in ("abs", "floor"):
                self.assertRegex(code, rf"(?m) import {op} as op_{op}_kernel_\d+$")
            self.assertEqual(tuple(_run_from_file(code, [x])), expected)

    @requires_cuda_and_triton
    def test_user_defined_kernel_source_with_backslashes(self):
        with tempfile.TemporaryDirectory() as d:
            (mod,) = _import_kernels(d, _DOC_MODULE, {"_kd_doc": {}})

            def fn(x):
                out = torch.empty_like(x)
                mod.doc_kernel[(4,)](x, out, x.numel(), BLOCK=64)
                return out

            x = torch.randn(256, device="cuda")
            result, code = _code_for(fn, x)
            self.assertEqual(result, x + 1)
            self.assertIn(r'"""Adds one.\n"""', code)

    @requires_cuda_and_triton
    @parametrize("autotune_at_compile_time", [False, True])
    def test_kernel_source_with_backslashes(self, autotune_at_compile_time):
        # Inline asm escapes its newlines for the string form's ''' literal.
        asm = "{\n.reg .pred p;\nsetp.ge.s32 p, $1, $2;\nselp.u32 $0, 1, 0, p;\n}"

        def fn(x, y):
            return inline_asm_elementwise(
                x, y, asm_str=asm, constraints="=r,r,r", dtype=torch.int32
            )

        x = torch.randint(-8, 8, (256,), device="cuda", dtype=torch.int32)
        y = torch.randint(-8, 8, (256,), device="cuda", dtype=torch.int32)
        cfg = {"triton.autotune_at_compile_time": autotune_at_compile_time}
        result, code = _code_for(fn, x, y, **cfg)
        self.assertEqual(result, (x >= y).int())
        # The def holds the asm as the string form's literal decodes it.
        self.assertIn(r"{\n.reg .pred p;\n", code)


class TestDefaultWrapper(TestCase):
    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        self.assertIn("async_compile.triton(", code)


instantiate_parametrized_tests(TestModuleLevelKernels)


if __name__ == "__main__":
    run_tests()
