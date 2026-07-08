# Owner(s): ["module: inductor"]
import ast
import textwrap
from unittest import mock, skipIf

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor import compile_to_python, config, load_from_python
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.standalone_compile import NoRunnableInductorModuleError
from torch._inductor.utils import fresh_cache
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import IS_BIG_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _capture(m, x, tracing_mode="fake"):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), mirroring how ``torch.compiler.precompile`` feeds a post-AOTAutograd inner
    graph to ``torch._inductor.compile_to_python``. The graph is decomposed against the
    inductor decomposition table (``compile_to_python`` drives inductor codegen directly
    and requires an already-decomposed graph) and traced under a single ``FakeTensorMode``
    (placeholders carry the fake ``val`` metadata inductor lowers against). Tracing runs
    ``m(x)`` once, so pass a throwaway module. ``tracing_mode="symbolic"`` produces a
    graph with dynamic dims."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(
            flat_fn,
            decomposition_table=select_decomp_table(),
            tracing_mode=tracing_mode,
        )(pb + [x])


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


def _extract_call(src):
    """Return the dedented source of the ``call`` entry point, normalized to the flat
    ``def call(args)`` signature. graph_partition (on by default in OSS, off in fbcode)
    wraps the body in a ``Runner.call(self, args)`` method; the body is byte-identical
    either way, so normalizing the signature makes the golden independent of the
    graph_partition default. The expect goldens lock this runtime entry point only: the
    rest of the emitted module (imports, the inert compile-time auto-tuning docstring, the
    ``# AOT ID`` global-counter comment) carries build- and ordering-dependent noise that
    should not be goldened."""
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            body = "\n".join(src.split("\n")[node.lineno - 1 : node.end_lineno])
            body = textwrap.dedent(body)
            return body.replace("def call(self, args):", "def call(args):", 1)
    raise AssertionError("generated module has no module-level def call")


class _Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2.0 + 1.0)


class _SumDim1(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class TestInductorCompileToPythonCodegen(TestCase):
    # These golden the runtime ``call`` body emitted under DEFAULT inductor config (no
    # option overrides), so the test reflects what callers get out of the box. The
    # extern-kernel / CPU codegen body is deterministic and build-independent (CPU tensors);
    # assert_size_stride lines come from the default size_asserts=True (the same default the
    # rest of the inductor golden suite relies on). ``_extract_call`` normalizes the
    # entry-point signature, so the golden is the same whether graph_partition wraps the body
    # in a ``Runner`` method (OSS default) or emits a flat top-level ``def call`` (fbcode).
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        return src, _extract_call(src)

    def test_addmm_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_3, flat_1), ((3, ), (5, 4), (3, 4)), ((1, ), (4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.addmm(flat_2, flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del flat_1
    del flat_2
    del flat_3
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_matmul_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3, bias=False).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_1), ((5, 4), (3, 4)), ((4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.mm(flat_2, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), out=buf0)
    del flat_1
    del flat_2
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_pointwise_codegen(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_3, flat_1), ((3, ), (5, 4), (3, 4)), ((1, ), (4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.addmm(flat_2, flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del flat_1
    del flat_2
    del flat_3
    buf1 = buf0; del buf0  # reuse
    cpp_fused_0(buf1)
    return (buf1, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_benchmark_harness_suppressed(self):
        # #187858 pins benchmark_harness=False, so the emitted module is runnable rather
        # than an Inductor profiling harness: none of these debug entry points appear.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _call_src = self._inner_call(m, x)
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
        ):
            self.assertNotIn(marker, src)

    def test_no_runnable_module_for_no_compute(self):
        # A graph with no compute (returns its input unchanged) lowers to no module-level
        # ``call``, so there is nothing runnable to inline.
        m = torch.nn.Identity().eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with self.assertRaises(NoRunnableInductorModuleError):
            compile_to_python(gm, _flat_inputs(m, x))


@requires_cuda_and_triton
class TestInductorCompileToPythonCudaCodegen(TestCase):
    # On CUDA, compile_to_python emits actual @triton.jit kernels rather than the CPU
    # extern_kernels / cpp_fused path the sibling class goldens. The expect goldens lock
    # the ``call()`` body only: it carries NO autotuning artifacts -- XBLOCK / num_warps
    # are chosen inside ``.run`` at launch time and the arch-specific DeviceProperties
    # live in the kernel decorator, not in ``call`` -- so it is arch-independent. The
    # hardware- and Triton-version-dependent kernel body is instead checked structurally
    # (assertIn). The device ordinal is hardcoded to 0 (these tests run on device 0, as
    # the rest of the inductor codegen-golden suite does). Default inductor config is used
    # (no option overrides); ``_extract_call`` normalizes the entry-point signature so the
    # golden is independent of the graph_partition default (Runner method in OSS, flat
    # ``def call`` in fbcode).
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        return src, _extract_call(src)

    def test_pointwise_triton_kernel_codegen(self):
        m = _Pointwise().eval().cuda()
        x = torch.randn(128, 64, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, = args
    args.clear()
    assert_size_stride(flat_1, (128, 64), (64, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(flat_1, buf0, 8192, stream=raw_stream0)
        del flat_1
    return (buf0, )""",
        )
        # The pointwise fusion lowers to a single @triton.jit pointwise kernel; the
        # call drives it directly with no extern (BLAS/cuDNN) kernel.
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.pointwise", src)
        # Kernel name is generic (triton_poi_fused_0) because the make_fx test graph
        # carries no source-node provenance; a real AOTAutograd graph would name it after
        # the fused ops (e.g. triton_poi_fused_add_mul_relu_0).
        self.assertIn("def triton_poi_fused_0(", src)
        self.assertIn("tl.load", src)
        self.assertIn("tl.store", src)
        self.assertIn("tl.maximum", src)  # the fused relu
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_triton_kernel_codegen(self):
        m = _SumDim1().eval().cuda()
        x = torch.randn(64, 256, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, = args
    args.clear()
    assert_size_stride(flat_1, (64, 256), (256, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_0.run(flat_1, buf0, 64, 256, stream=raw_stream0)
        del flat_1
    return (buf0, )""",
        )
        # A row reduction lowers to a (persistent) reduction Triton kernel doing the
        # cross-row ``tl.sum``; there is no extern kernel.
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        # Generic kernel name (no source-node provenance on the make_fx test graph); a real
        # AOTAutograd graph would name it triton_per_fused_sum_0.
        self.assertIn("def triton_per_fused_0(", src)
        self.assertIn("tl.sum", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_triton_epilogue_codegen(self):
        # CUDA counterpart of test_addmm_relu_fused_pointwise_codegen: the matmul is an
        # extern BLAS call but the relu epilogue fuses into a @triton.jit kernel
        # (triton_poi_fused_add_0 here; a real AOTAutograd graph, with source-node
        # provenance, would name it triton_poi_fused_addmm_relu_0) rather than the CPU
        # cpp_fused_relu_0.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval().cuda()
        x = torch.randn(5, 4, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_3, flat_1), ((5, 4), (3, 4)), ((4, 1), (4, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_3 = copy_if_misaligned(flat_3)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((5, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), out=buf0)
        del flat_1
        del flat_3
        assert_size_stride(flat_2, (3, ), (1, ), 'input')
        flat_2 = copy_if_misaligned(flat_2)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(buf1, flat_2, 15, stream=raw_stream0)
        del flat_2
    return (buf1, )""",
        )
        self.assertIn("extern_kernels.mm", call_src)
        self.assertIn("@triton.jit", src)
        self.assertIn("def triton_poi_fused_add_0(", src)
        self.assertIn("tl.maximum", src)  # the fused relu epilogue
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_softmax_fused_reduction_triton_kernel(self):
        # softmax is the canonical multi-stage fusion: max, subtract, exp, sum, divide
        # all collapse into ONE persistent-reduction Triton kernel. Its exact name
        # (the decomposition route) varies, so this checks structure + numerics rather
        # than goldening the call body.
        m = _Softmax().eval().cuda()
        x = torch.randn(32, 128, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        self.assertIn("tl.sum", src)  # the denominator reduction
        # the exp numerator: libdevice.exp / tl_math.exp / tl.exp depending on fast-math
        self.assertIn(".exp(", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    @skipIf(
        not IS_BIG_GPU,
        "Skipping triton backend only since not big GPU (not enough SMs for "
        "max_autotune_gemm; the TRITON-only GEMM backend then has no choices)",
    )
    def test_max_autotune_excludes_benchmark_lowerings(self):
        # max_autotune's decompose_k GEMM choice compiles each k-split as a nested
        # SubgraphChoiceCaller benchmark GraphLowering during autotuning. compile_to_python
        # reads the FINAL module's source off the compiled artifact, so those benchmark
        # modules are excluded by construction -- they never become the artifact, no
        # filtering needed. fresh_cache forces a cold autotune cache so the benchmark
        # compiles actually run, and the spy asserts they did, so the test cannot silently
        # degrade into a no-op. k >= 32*m and k >= 32*n makes decompose_k a candidate.
        from torch._inductor.codegen.subgraph import SubgraphChoiceCaller

        m = (
            torch.nn.Sequential(torch.nn.Linear(8192, 64, bias=False), torch.nn.ReLU())
            .eval()
            .cuda()
        )
        x = torch.randn(64, 8192, device="cuda")
        gm = _capture(m, x)
        orig = SubgraphChoiceCaller._compile_for_benchmarking
        bench_calls = []

        def _spy(choice, *args, **kwargs):
            bench_calls.append(1)
            return orig(choice, *args, **kwargs)

        with fresh_cache():
            with mock.patch.object(
                SubgraphChoiceCaller, "_compile_for_benchmarking", _spy
            ):
                src, _cache = compile_to_python(
                    gm,
                    _flat_inputs(m, x),
                    options={
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "TRITON",
                    },
                )
        self.assertGreater(len(bench_calls), 0)  # benchmark lowerings actually ran
        self.assertIn("def call(", src)  # the final runnable module
        self.assertNotIn(
            "benchmark_", src
        )  # excluded by construction, never in the artifact
        with torch.no_grad():
            self.assertEqual(
                _exec(src)(_flat_inputs(m, x))[0], m(x), atol=1e-2, rtol=1e-2
            )

    @config.patch({"compile_threads": 1})
    def test_warm_load_rehydrates_static_launcher(self):
        # The cache must let a COLD (fresh-dir) load reuse the compiled kernels AND the
        # static CUDA launcher, not fall back to the slower dynamic launch. This works only
        # because compile_to_python defaults keep_static_cubin_raw=True, travelling the raw
        # cubin in the bundle; with the default False, reload_cubin_path can't find the cubin
        # file in a fresh dir and the static launcher is silently dropped. Assert the static
        # autotuner is rehydrated on the warm load (counter > 0) and the result matches eager.
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        if not config.use_static_cuda_launcher:
            self.skipTest("requires the static CUDA launcher")
        m = _Pointwise().eval().cuda()
        x = torch.randn(1024, 1024, device="cuda")
        src, cache = compile_to_python(_capture(m, x), _flat_inputs(m, x))
        self.assertIsInstance(cache, bytes)
        with fresh_cache():
            counters.clear()
            with torch.no_grad():
                out = load_from_python(src, cache)(_flat_inputs(m, x))
            rehydrated = counters["inductor"]["triton_bundler_load_static_autotuner"]
        self.assertGreater(rehydrated, 0)
        self.assertEqual(out[0], m(x))


class TestInductorCompileToPythonContract(TestCase):
    # Contract + branch coverage the codegen-golden classes above do not exercise: the
    # option pin-override, the cache return value, the graph_partition runner form, the
    # no-compute error path, and warm-cache reuse. All CPU, so no GPU is required.
    def test_rejects_non_graphmodule(self):
        with self.assertRaises(TypeError):
            compile_to_python("not a graph module", [])

    def test_pins_override_conflicting_user_options(self):
        # The benchmark_harness/cpp_wrapper pins must beat conflicting user options so the
        # captured module stays the runnable python wrapper rather than a C++ wrapper or a
        # profiling harness. ``def call(`` matches both the flat ``def call(args)`` and the
        # graph_partition ``Runner.call(self, args)`` form, so this does not pin partitioning.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={
                "cpp_wrapper": True,
                "benchmark_harness": True,
            },
        )
        self.assertIn("def call(", src)
        self.assertNotIn('extern "C"', src)
        self.assertNotIn("AOTInductorModel", src)
        self.assertNotIn("benchmark_compiled_module", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_load_from_python_standalone_and_warm(self):
        # load_from_python(python_code, cache) is the inverse of compile_to_python. The
        # python_code is self-contained: it loads and runs with cache=None (JIT path). The
        # cache is a PURE ACCELERATOR -- passing it warms the kernel caches so exec loads
        # precompiled binaries instead of recompiling; both paths must match eager. Run each
        # in a fresh cache dir so the standalone path is a genuine cold load. The cache
        # bytes require force_disable_caches off AND fx_graph_cache on; those flags are
        # env-authoritative on PyTorch CI's cache-disabled shards (cannot be patched back
        # on), where compile_to_python returns None -- skip there (test_no_cache_when_caches
        # _disabled covers the None path).
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, cache = compile_to_python(_capture(m, x), _flat_inputs(m, x))
        self.assertIsInstance(cache, bytes)
        # Standalone: no cache, the module JIT-compiles its own kernels.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src)(_flat_inputs(m, x))[0], m(x))
        # Warm: the cache accelerates the same module; result is identical.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src, cache)(_flat_inputs(m, x))[0], m(x))

    def test_no_cache_when_caches_disabled(self):
        # With caches disabled there is no saveable artifact, so cache is None; the source
        # still runs (the kernels JIT-compile from it on first call).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={"force_disable_caches": True},
        )
        self.assertIsNone(cache)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_graph_partition_runner_call_form(self):
        # graph_partition=True emits the Runner form (``call = runner.call``) instead of a
        # top-level ``def call``; the returned source must carry it and the emitted module
        # must still run.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x), _flat_inputs(m, x), options={"graph_partition": True}
        )
        self.assertIn("call = runner.call", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_dynamic_shapes_emits_symbolic_codegen(self):
        # A symbolically-traced graph carries symbolic sizes in its placeholder val
        # metadata, so the emitted call() is keyed on symbolic sizes (sN) rather than baked
        # constants and the single module runs at multiple shapes. Shapes come from the
        # graph (there is no dynamic_shapes knob); the other tests exercise the static path.
        # Symbol names are non-deterministic, so assert structure + multi-shape numerics
        # rather than goldening.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        call_src = _extract_call(src)
        self.assertRegex(call_src, r"\bs\d+\b")  # a symbolic size symbol is present
        self.assertNotIn("(8, 4)", call_src)  # the input shape is not baked in
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    @config.patch({"compile_threads": 1})
    def test_warm_cache_still_yields_source(self):
        # On a warm cache the 2nd compile of the same graph is an FxGraphCache hit (no fresh
        # codegen), yet compile_to_python still returns a runnable module: Inductor
        # populates source_code on the restored artifact too. fresh_cache isolates an empty
        # cache dir so the first compile is a guaranteed miss and the second a guaranteed
        # hit; compile_threads=1 keeps codegen in-process so the hit counter is
        # deterministic. The warm path needs caching, env-disabled on some CI shards
        # (force_disable_caches cannot be patched back on), so skip there.
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with fresh_cache():
            counters.clear()
            src1, _ = compile_to_python(_capture(m, x), _flat_inputs(m, x))
            hits_after_first = counters["inductor"]["fxgraph_cache_hit"]
            src2, _ = compile_to_python(_capture(m, x), _flat_inputs(m, x))
            hits_after_second = counters["inductor"]["fxgraph_cache_hit"]
        self.assertEqual(hits_after_first, 0)
        self.assertEqual(hits_after_second, 1)
        self.assertIn("def call(", src1)
        self.assertIn("def call(", src2)
        with torch.no_grad():
            self.assertEqual(_exec(src2)(_flat_inputs(m, x))[0], m(x))


if __name__ == "__main__":
    run_tests()
