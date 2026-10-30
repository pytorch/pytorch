# Owner(s): ["module: inductor"]

import ast
import contextlib
import functools
import os
import re
import tempfile
import unittest
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import compile_to_python, config, load_from_python
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.runtime.autotune_cache import (
    _comment_stripped_hash,
    AutotuneCacheBundler,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.runtime.hints import HeuristicType, KERNEL_CONFIGS_NAME
from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.runtime.triton_heuristics import (
    AutotuneCache,
    cached_autotune,
    CachingAutotuner,
    CachingAutotunerPlugin,
    config_to_dict,
    DebugAutotuner,
    DEFER,
    StaticTritonCompileResult,
    TritonCompileResult,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton
from torch.utils._debug_mode import DebugMode
from torch.utils._debug_mode._calls import _TritonKernelCall
from torch.utils._triton import has_triton_package


def _code_for(fn, *args, dynamic=None, **config_kwargs):
    torch._dynamo.reset()
    config_kwargs.setdefault("triton.autotune_at_compile_time", True)
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn, dynamic=dynamic), *args)
    return result, "\n".join(codes)


def _kernels(code):
    return set(re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE))


def _load_from_file(code):
    # A module-level kernel reads its source from the module's file, so the file must
    # outlive the load. The test case's fresh cache dir is removed after the test.
    path = os.path.join(tempfile.mkdtemp(dir=cache_dir()), "module.py")
    with open(path, "w") as f:
        f.write(code)
    ns = {"__file__": path, "__name__": "_kernel_configs"}
    exec(compile(code, path, "exec"), ns)
    return ns


@contextlib.contextmanager
def _no_runtime_tuning():
    def tune(*args, **kwargs):
        raise AssertionError("a kernel autotuned at its first launch")

    with (
        mock.patch.object(CachingAutotuner, "autotune_to_one_config", tune),
        mock.patch.object(CachingAutotuner, "_coordinate_descent_tuning", tune),
        mock.patch.object(CachingAutotuner, "_combo_sequential_autotune", tune),
        # a cached best config would replace the pinned one
        mock.patch.object(AutotuneCache, "create", tune),
    ):
        yield


@contextlib.contextmanager
def _spy_launches(record):
    """Calls record(kernel, in_block) after each kernel launch, in_block telling whether
    the launch was in the compile-time autotune block."""
    in_block = [False]
    block = PythonWrapperCodegen.generate_and_run_autotune_block
    run = CachingAutotuner.run

    def block_spy(self, *args, **kwargs):
        in_block[0] = True
        try:
            return block(self, *args, **kwargs)
        finally:
            in_block[0] = False

    def run_spy(self, *args, **kwargs):
        out = run(self, *args, **kwargs)
        record(self, in_block[0])
        return out

    with (
        mock.patch.object(
            PythonWrapperCodegen, "generate_and_run_autotune_block", block_spy
        ),
        mock.patch.object(CachingAutotuner, "run", run_spy),
    ):
        yield


@contextlib.contextmanager
def _configs_tuned_at_compile_time():
    """Each kernel's config right after its launch in the compile-time autotune block,
    from the kernel object that launched. The wrapper reads the same objects once the
    block is done, so this checks it pins each kernel's own finished config; it does
    not check that tuning chose well."""
    tuned = {}

    def record(kernel, in_block):
        name = kernel.inductor_meta.get("kernel_name")
        if in_block and name is not None and len(kernel.launchers) == 1:
            tuned[name] = config_to_dict(kernel.launchers[0].config)

    with _spy_launches(record):
        yield tuned


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _cond_softmax(x):
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


def _bucketize(x):
    # its inductor_meta holds an AutotuneHint
    return torch.bucketize(x, torch.linspace(-1, 1, 8, device=x.device))


def _foreach(x):
    return tuple(torch._foreach_add([x, x.t()], 1))


def _mix_order(x):
    return x.sum(dim=1), x.sum(dim=0)


class TestKernelConfigs(TestCase):
    """Under triton.autotune_at_compile_time, the module launches each kernel with the
    config the compile tuned it to."""

    @requires_gpu_and_triton
    @parametrize(
        "case",
        [
            # (name, fn, input shape, text the kind of kernel leaves in the code, config)
            ("persistent_reduction", _softmax, (64, 128), "persistent_reduction("),
            ("reduction", lambda x: x.sum(-1), (64, 3000), "heuristics.reduction("),
            ("pointwise", lambda x: (x.sin() * 2).relu(), (4096,), "pointwise("),
            ("bucketize", _bucketize, (4096,), "bucketize_binary_search"),
            ("subgraph", _cond_softmax, (64, 128), "def true_graph_0("),
            ("split_scan", lambda x: x.abs().cumsum(0), (2**20,), "split_scan("),
            ("foreach", _foreach, (64, 128), "foreach("),
            (
                "mix_order_reduction",
                _mix_order,
                (4096, 1536),
                "RSPLIT_SIZE",
                {"triton.mix_order_reduction": True},
            ),
            (
                "cooperative_reduction",
                lambda x: x.sum(-1),
                (8, 2**16),
                "cooperative_reduction(",
                {"triton.force_cooperative_reductions": True},
            ),
        ],
        name_fn=lambda case: case[0],
    )
    @config.patch(coordinate_descent_tuning=True, max_autotune_pointwise=True)
    def test_kernels_launch_with_their_compile_time_config(self, case):
        _, fn, shape, marker, *cfg = case
        x = torch.randn(shape, device=GPU_TYPE)
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, x, **(cfg[0] if cfg else {}))
        self.assertIn(marker, code)
        kernels = _kernels(code)
        self.assertTrue(kernels)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            # -x takes the other torch.cond branch, so every kernel launches
            for arg in (x, -x):
                out, expected = ns["call"]([arg]), fn(arg)
                if not isinstance(expected, tuple):
                    out = out[0]
                self.assertEqual(out, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), kernels)
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            self.assertEqual(cfg, tuned[name], name)
            (launcher,) = ns[name].launchers
            self.assertEqual(config_to_dict(launcher.config), cfg, name)

    @requires_gpu_and_triton
    def test_a_fixed_config_kernel_keeps_its_config(self):
        cfg = {"XBLOCK": 128}

        class FixedChoices(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {**kernel_kwargs, "fixed_config": FixedTritonConfig(cfg)}

        fn = lambda x: (x.sin() * 2).relu()  # noqa: E731
        x = torch.randn(4096, device=GPU_TYPE)
        with V.set_choices_handler(FixedChoices()):
            _, code = _code_for(fn, x)
        self.assertIn("@triton_heuristics.fixed_config(", code)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([x])[0], fn(x))
        ((name, pinned),) = ns["KERNEL_CONFIGS"].items()
        self.assertLessEqual(cfg.items(), pinned.items())
        (launcher,) = ns[name].launchers
        self.assertEqual(config_to_dict(launcher.config), pinned)

    @requires_gpu_and_triton
    @parametrize("threads", [None, 1])
    @parametrize("cache_hit", [False, True])
    @config.patch(coordinate_descent_tuning=True, fx_graph_cache=True)
    def test_inductor_keeps_the_pool_kernel_in_process(self, cache_hit, threads):
        # compiled_kernels_for_load binds the pool-built kernel ahead of the
        # KERNEL_CONFIGS lookup, so inductor's own load does not pin, from the FX graph
        # cache too.
        fn = lambda x: x.sum(-1)  # noqa: E731
        x = torch.randn(64, 3000, device=GPU_TYPE)
        if cache_hit:
            _code_for(fn, x)
        counters.clear()
        launched = []

        def record(kernel, in_block):
            # Only the module's own launches, not the autotune block's.
            if not in_block:
                launched.append(kernel.heuristic_type)

        # With one compile thread the def binds a serial-compile thunk, not a pool future.
        patches = {} if threads is None else {"compile_threads": threads}
        with _spy_launches(record):
            _, code = _code_for(fn, x, **patches)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], int(cache_hit))
        self.assertIn("KERNEL_CONFIGS = {", code)
        self.assertIn(HeuristicType.REDUCTION, launched)
        self.assertNotIn(HeuristicType.FIXED, launched)

    @requires_gpu_and_triton
    def test_compile_to_python_artifact_launches_pinned(self):
        # compile_to_python (as standalone_compile) turns triton.autotune_at_compile_time
        # on, so its module carries the table and its kernels load pinned.
        x = torch.randn(64, 3000, device=GPU_TYPE)
        decomps = select_decomp_table()
        gm = make_fx(lambda x: (x.sum(-1),), decomps, tracing_mode="fake")(x)
        src, _ = compile_to_python(gm, [x])
        with _no_runtime_tuning():
            call = load_from_python(src)
            self.assertEqual(call([x])[0], x.sum(-1), atol=1e-3, rtol=1e-3)
        ns = getattr(call, "__func__", call).__globals__
        ((kernel, cfg),) = ns["KERNEL_CONFIGS"].items()
        self.assertEqual(ns[kernel].heuristic_type, HeuristicType.FIXED)
        (launcher,) = ns[kernel].launchers
        self.assertEqual(config_to_dict(launcher.config), cfg)

    @requires_gpu_and_triton
    def test_a_kernel_the_block_did_not_launch_is_not_pinned(self):
        # async_compile precompiles every kernel, so a single-config kernel has one
        # launcher even if the autotune block never ran it.
        block = PythonWrapperCodegen.generate_and_run_autotune_block

        def launch_nothing(self, *args, **kwargs):
            self.kernel_autotune_names_all_graphs.clear()
            return block(self, *args, **kwargs)

        fn = lambda x: (x.sin() * 2).relu()  # noqa: E731
        x = torch.randn(4096, device=GPU_TYPE)
        name = "generate_and_run_autotune_block"
        with mock.patch.object(PythonWrapperCodegen, name, launch_nothing):
            _, code = _code_for(fn, x)
        (kernel,) = _kernels(code)
        ns = _load_from_file(code)
        self.assertEqual(ns["KERNEL_CONFIGS"], {})
        self.assertEqual(ns[kernel].heuristic_type, HeuristicType.POINTWISE)

    @requires_gpu_and_triton
    def test_bundled_autotune_cache_key_ignores_the_winners(self):
        # AutotuneCacheBundler keys on the wrapper code, both when the graph is compiled
        # and on an FX graph cache hit. The key must not change with the config each
        # kernel's tuning picked, which the table records.
        keys, xblock = [], [1]
        begin = AutotuneCacheBundler.begin_compile.__func__
        block = PythonWrapperCodegen.generate_and_run_autotune_block

        def begin_spy(cls, inductor_meta, *, code=None, code_hash=None):
            keys.append(_comment_stripped_hash(code))
            return begin(cls, inductor_meta, code=code, code_hash=code_hash)

        def another_winner(self, *args, **kwargs):
            result = block(self, *args, **kwargs)
            for name, cfg in self.kernel_configs.items():
                self.kernel_configs[name] = {**cfg, "XBLOCK": xblock[0]}
            return result

        fn = lambda x: x.sum(-1)  # noqa: E731
        x = torch.randn(64, 3000, device=GPU_TYPE)
        with (
            mock.patch.object(
                AutotuneCacheBundler, "begin_compile", classmethod(begin_spy)
            ),
            mock.patch.object(
                PythonWrapperCodegen, "generate_and_run_autotune_block", another_winner
            ),
        ):
            for xblock[0] in (1, 2):
                _, code = _code_for(fn, x, fx_graph_cache=False)
                self.assertIn(f"'XBLOCK': {xblock[0]}", code)
            _code_for(fn, x)
            counters.clear()
            _, code = _code_for(fn, x)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertIn("KERNEL_CONFIGS = {\n    'triton_", code)
        self.assertEqual(len(keys), 4)
        self.assertEqual(len(set(keys)), 1)

    @requires_gpu_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_a_subgraph_kernel_pinned_first_writes_the_table(self):
        # The predicate is an input, so the root launches no kernel: the first one
        # pinned is in a branch nested in another branch.
        def fn(p, x):
            def outer(t):
                return torch.cond(p, _softmax, lambda u: u.cos(), (t,))

            return torch.cond(p, outer, lambda t: t.sin() * 2, (x,))

        p = torch.tensor(True, device=GPU_TYPE)
        x = torch.randn(64, 128, device=GPU_TYPE)
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, p, x)
        kernels = _kernels(code)
        self.assertTrue(kernels)
        self.assertNotIn(".run(", code.split("def call(")[-1])
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            for pred in (p, ~p):
                out = ns["call"]([pred, x])[0]
                self.assertEqual(out, fn(pred, x), atol=1e-4, rtol=1e-4)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), kernels)
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            self.assertEqual(cfg, tuned[name], name)

    @requires_gpu_and_triton
    @parametrize("per_subkernel", [False, True])
    @config.patch(
        combo_kernels=True,
        benchmark_combo_kernel=False,
        combo_kernel_max_distance=-1,
        coordinate_descent_tuning=True,
    )
    def test_combo_kernel_launches_with_its_compile_time_config(self, per_subkernel):
        def fn(a, b):
            return a.relu(), b.sigmoid()

        a, b = (torch.randn(n, n, device=GPU_TYPE) for n in (10, 20))
        per_subkernel_cfg = {"combo_kernel_per_subkernel_blocks": per_subkernel}
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, a, b, **per_subkernel_cfg)
        self.assertIn("combo_grid_meta", code)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([a, b]), fn(a, b))
        ((name, cfg),) = ns["KERNEL_CONFIGS"].items()
        self.assertEqual(cfg, tuned[name])
        (launcher,) = ns[name].launchers
        self.assertEqual(config_to_dict(launcher.config), cfg)
        if per_subkernel:
            self.assertIn("XBLOCK_1", cfg)

    @requires_gpu_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_dynamic_shape_kernel_keeps_its_config_at_every_size(self):
        # Tuned once on the size hints, it launches with that config at other sizes.
        def fn(x):
            return x.sum(-1)

        x = torch.randn(64, 3000, device=GPU_TYPE)
        with _configs_tuned_at_compile_time() as tuned:
            _, code = _code_for(fn, x, dynamic=True)
        with _no_runtime_tuning():
            ns = _load_from_file(code)
            for shape in [(64, 3000), (17, 5000), (200, 129)]:
                y = torch.randn(shape, device=GPU_TYPE)
                # The size symbols come before or after the tensor, as get_args shows.
                dims = iter(shape)
                example = ns["get_args"]()
                args = [y if torch.is_tensor(a) else next(dims) for a in example]
                out = ns["call"](args)[0]
                self.assertEqual(out, fn(y), atol=1e-3, rtol=1e-3)
        (cfg,) = ns["KERNEL_CONFIGS"].values()
        (kernel,) = _kernels(code)
        self.assertEqual(cfg, tuned[kernel])
        (launcher,) = ns[kernel].launchers
        self.assertEqual(config_to_dict(launcher.config), cfg)

    @requires_gpu_and_triton
    def test_template_keeps_its_one_config(self):
        if not is_big_gpu():
            self.skipTest("Triton GEMM templates need a big GPU")
        a, b = (torch.randn(64, 64, device=GPU_TYPE) for _ in range(2))
        cfg = {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        result, code = _code_for(lambda a, b: (a @ b).relu(), a, b, **cfg)
        (template,) = {k for k in _kernels(code) if k.startswith("triton_tem_")}
        ns = _load_from_file(code)
        self.assertNotIn(template, ns.get("KERNEL_CONFIGS", {}))
        self.assertEqual(ns[template].heuristic_type, HeuristicType.TEMPLATE)
        self.assertEqual(ns["call"]([a, b])[0], result)
        # Naming it in a table by hand does not pin it either. The module has no table
        # of its own to replace this one.
        self.assertNotIn(KERNEL_CONFIGS_NAME, code)
        row = {"num_warps": 1, "num_stages": 1}
        table = f"KERNEL_CONFIGS = {{{template!r}: {row!r}}}\n"
        ns = _load_from_file(table + code)
        self.assertEqual(ns[template].heuristic_type, HeuristicType.TEMPLATE)
        self.assertEqual(ns["call"]([a, b])[0], result)

    @requires_gpu_and_triton
    def test_user_defined_kernel_keeps_its_autotuning(self):
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
            key=["n"],
        )
        @triton.jit
        def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            add_one[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out, n)
            return out.sin()

        x = torch.randn(4096, device=GPU_TYPE)
        result, code = _code_for(fn, x)
        self.assertEqual(result, fn(x))
        (user_kernel,) = re.findall(r"^def (add_one\w*)\(", code, re.MULTILINE)
        ns = _load_from_file(code)
        self.assertEqual(set(ns["KERNEL_CONFIGS"]), _kernels(code))
        self.assertNotIn(user_kernel, ns["KERNEL_CONFIGS"])
        self.assertEqual(ns[user_kernel].heuristic_type, HeuristicType.USER_AUTOTUNE)
        self.assertEqual(ns["call"]([x])[0], result)
        # Naming it in the table by hand does not pin it either.
        row = f"KERNEL_CONFIGS = {{\n    {user_kernel!r}: {{'BLOCK': 64}},\n"
        ns = _load_from_file(code.replace("KERNEL_CONFIGS = {\n", row, 1))
        self.assertEqual(ns[user_kernel].heuristic_type, HeuristicType.USER_AUTOTUNE)
        self.assertEqual(ns["call"]([x])[0], result)

    @requires_gpu_and_triton
    def test_kernel_without_a_compile_time_config_tunes_on_first_launch(self):
        # As for a kernel the block leaves with more than one launcher.
        block = PythonWrapperCodegen.generate_and_run_autotune_block

        def leave_no_config(self, *args, **kwargs):
            result = block(self, *args, **kwargs)
            self.kernel_configs.update(dict.fromkeys(self.kernel_configs))
            return result

        def fn(x):
            return x.sum(-1)

        x = torch.randn(64, 3000, device=GPU_TYPE)
        spy = mock.patch.object(
            PythonWrapperCodegen, "generate_and_run_autotune_block", leave_no_config
        )
        with spy:
            # max_autotune, so that the kernel has configs to choose between
            cfg = {"autotune_local_cache": False, "max_autotune": True}
            result, code = _code_for(fn, x, **cfg)
        (kernel,) = _kernels(code)
        ns = _load_from_file(code)
        self.assertEqual(ns["KERNEL_CONFIGS"], {})
        self.assertEqual(ns[kernel].heuristic_type, HeuristicType.REDUCTION)
        self.assertGreater(len(ns[kernel].launchers), 1)
        tune = CachingAutotuner.autotune_to_one_config
        with mock.patch.object(
            CachingAutotuner, "autotune_to_one_config", autospec=True, side_effect=tune
        ) as tuned:
            self.assertEqual(ns["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
        tuned.assert_called_once()

    @requires_gpu_and_triton
    @parametrize(
        "failure", ["compile", "load", "load_pipelined", "load_cached", "load_profiled"]
    )
    def test_a_pinned_config_that_does_not_fit_falls_back_to_tuning(self, failure):
        # As on a GPU with less shared memory than the one the module was tuned on: Triton
        # raises OutOfResources when it loads the binary, and with num_stages > 1 the
        # launcher is first retried with pipelining disabled. A config that does not
        # compile fails on the module's own precompile threads instead. With the local
        # autotune cache on, the compile-time winner, the pinned config, is cached. Under
        # profile_bandwidth the kernel is a DebugAutotuner, which precompiles in run().
        num_stages = 2 if failure == "load_pipelined" else 1
        fn = lambda x: x.sum(-1)  # noqa: E731
        x = torch.randn(64, 3000, device=GPU_TYPE)
        cached = failure == "load_cached"
        profiled = failure == "load_profiled"
        cfg = {"autotune_local_cache": cached, "max_autotune": True}
        result, code = _code_for(fn, x, profile_bandwidth=profiled, **cfg)
        self.assertEqual(result, fn(x), atol=1e-3, rtol=1e-3)
        (kernel,) = _kernels(code)
        row = re.search(rf"^    '{kernel}': (.*),$", code, re.MULTILINE)
        tuned_row = ast.literal_eval(row[1])
        # Unchanged when cached, so that the cached winner is the pinned config.
        pinned = tuned_row if cached else {**tuned_row, "num_stages": num_stages}
        num_stages = pinned["num_stages"]
        code = code.replace(row[0], f"    {kernel!r}: {pinned!r},")
        no_stages = lambda c: {k: v for k, v in c.items() if k != "num_stages"}  # noqa: E731
        failed = []

        def does_not_fit(cls, name, config_of, error):
            orig = getattr(cls, name)

            def fail(self, *args):
                c = config_of(self, *args)
                if no_stages(config_to_dict(c)) == no_stages(pinned):
                    failed.append(c.num_stages)
                    raise error
                return orig(self, *args)

            return mock.patch.object(cls, name, fail)

        smem = OutOfResources(1 << 20, 1 << 16, "shared memory")
        if failure == "compile":
            targets = [(CachingAutotuner, "_precompile_config", lambda s, c: c, smem)]
        else:
            # The static launcher reports shared memory overflow as a CUDA OOM.
            oom = torch.cuda.OutOfMemoryError("out of shared memory")
            config_of = lambda s: s.config  # noqa: E731
            targets = [
                (TritonCompileResult, "make_launcher", config_of, smem),
                (StaticTritonCompileResult, "make_launcher", config_of, oom),
            ]
        autotune = CachingAutotuner.autotune_to_one_config
        candidates = []

        def tune(kernel, *args, **kwargs):
            candidates.extend(config_to_dict(c.config) for c in kernel.launchers)
            return autotune(kernel, *args, **kwargs)

        spy = mock.patch.object(
            CachingAutotuner, "autotune_to_one_config", autospec=True, side_effect=tune
        )
        with contextlib.ExitStack() as stack, spy as tuned:
            for target in targets:
                stack.enter_context(does_not_fit(*target))
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
            if cached:
                # The new winner was saved over the pinned one, so this load does not tune.
                again = _load_from_file(code)
                self.assertEqual(again["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
        self.assertEqual(failed[:num_stages], [2, 1][-num_stages:])
        tuned.assert_called_once()
        # It is the unpinned kernel now, which tunes as it did before pinning.
        self.assertIsNone(ns[kernel].fallback)
        self.assertEqual(ns[kernel].heuristic_type, HeuristicType.REDUCTION)
        self.assertEqual(isinstance(ns[kernel], DebugAutotuner), profiled)
        (launcher,) = ns[kernel].launchers
        self.assertIn(config_to_dict(launcher.config), candidates)

    @requires_gpu_and_triton
    def test_a_plugin_compiles_the_kernel_again_after_falling_back(self):
        x = torch.randn(64, 3000, device=GPU_TYPE)
        cfg = {"autotune_local_cache": False, "max_autotune": True}
        result, code = _code_for(lambda x: x.sum(-1), x, **cfg)
        (kernel,) = _kernels(code)
        row = re.search(rf"^    '{kernel}': (.*),$", code, re.MULTILINE)
        pinned = ast.literal_eval(row[1])
        seen = []

        class Plugin(CachingAutotunerPlugin):
            def pre_compile(self, autotuner):
                seen.append((autotuner.heuristic_type, hasattr(autotuner, "mark")))
                autotuner.mark = True
                return DEFER

        def does_not_fit(orig):
            def make_launcher(self):
                if config_to_dict(self.config) == pinned:
                    raise OutOfResources(1 << 20, 1 << 16, "shared memory")
                return orig(self)

            return make_launcher

        plugins = mock.patch(
            "torch._inductor.runtime.triton_heuristics.get_caching_autotuner_plugins",
            lambda autotuner: [Plugin()],
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(plugins)
            for cls in (TritonCompileResult, StaticTritonCompileResult):
                fail = does_not_fit(cls.make_launcher)
                stack.enter_context(mock.patch.object(cls, "make_launcher", fail))
            ns = _load_from_file(code)
            self.assertEqual(ns["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
        # The unpinned kernel does not have what the plugin set on the pinned one.
        fixed, reduction = HeuristicType.FIXED, HeuristicType.REDUCTION
        self.assertEqual(seen, [(fixed, False), (reduction, False)])

    @requires_gpu_and_triton
    def test_falling_back_decides_on_every_attribute_set_after_init(self):
        # Falling back replaces a pinned kernel's __dict__ with a fresh unpinned one's,
        # so state set after __init__ is dropped unless _kept_on_fallback names it.
        x = torch.randn(64, 128, device=GPU_TYPE)
        _, code = _code_for(_softmax, x)
        ns = _load_from_file(code)
        ns["call"]([x])
        cached = {
            attr
            for cls in CachingAutotuner.__mro__
            for attr, value in vars(cls).items()
            if isinstance(value, functools.cached_property)
        }
        for name in _kernels(code):
            kernel = ns[name]
            after_init = set(vars(kernel)) - set(vars(kernel.fallback()))
            # A cached property is recomputed by the unpinned kernel, so it is dropped.
            dropped = after_init - set(CachingAutotuner._kept_on_fallback) - cached
            self.assertEqual(dropped, {"fallback"}, name)

    @requires_gpu_and_triton
    def test_the_fallback_can_be_built_again(self):
        # Building the unpinned kernel pops from its metas, so a retry after a failed
        # build would otherwise tune an in-place kernel without cloning what it mutates.
        def fn(x):
            x.add_(1)
            return x.sum()

        x = torch.randn(64, 128, device=GPU_TYPE)
        _, code = _code_for(fn, x.clone())
        ns = _load_from_file(code)
        (name,) = _kernels(code)
        kernel = ns[name]
        mutated = kernel.mutated_arg_names
        self.assertTrue(mutated)
        for _ in range(2):
            self.assertEqual(kernel.fallback().mutated_arg_names, mutated)

    @requires_gpu_and_triton
    def test_an_unrelated_error_in_a_pinned_kernel_does_not_fall_back(self):
        x = torch.randn(64, 3000, device=GPU_TYPE)
        _, code = _code_for(lambda x: x.sum(-1), x)
        calls = []

        def load_error(self):
            calls.append(self.config)
            raise RuntimeError("unrelated")

        with contextlib.ExitStack() as stack:
            for cls in (TritonCompileResult, StaticTritonCompileResult):
                stack.enter_context(mock.patch.object(cls, "make_launcher", load_error))
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                _load_from_file(code)
        self.assertEqual(len(calls), 1)

    @requires_gpu_and_triton
    def test_falling_back_keeps_the_launch_being_recorded(self):
        # run() records a DebugMode launch before a lazy precompile, which is where a
        # pinned kernel may fall back; the launch must still be finalized afterwards.
        x = torch.randn(64, 3000, device=GPU_TYPE)
        cfg = {"autotune_local_cache": False, "max_autotune": True}
        result, code = _code_for(lambda x: x.sum(-1), x, **cfg)
        ns = _load_from_file(code)
        (kernel,) = _kernels(code)
        autotuner = ns[kernel]
        (launcher,) = autotuner.launchers
        pinned = config_to_dict(launcher.config)
        autotuner.launchers = []
        smem = OutOfResources(1 << 20, 1 << 16, "shared memory")

        def does_not_fit(orig):
            def make_launcher(self):
                if config_to_dict(self.config) == pinned:
                    raise smem
                return orig(self)

            return make_launcher

        with contextlib.ExitStack() as stack:
            for cls in (TritonCompileResult, StaticTritonCompileResult):
                fail = does_not_fit(cls.make_launcher)
                stack.enter_context(mock.patch.object(cls, "make_launcher", fail))
            with DebugMode() as debug_mode:
                self.assertEqual(ns["call"]([x])[0], result, atol=1e-3, rtol=1e-3)
        self.assertIsNone(autotuner.fallback)
        (call,) = [c for c in debug_mode.operators if isinstance(c, _TritonKernelCall)]
        # finalize() drops the launch's tensors.
        self.assertFalse(hasattr(call, "kwargs"))

    # Nothing launches, so no GPU is needed.
    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_pinned_kernel_keeps_the_backends_autotuner_class(self):
        # An out-of-tree backend's heuristics pass their own CachingAutotuner subclass.
        import triton.language as tl

        # The table in the python wrapper module that defines the kernel.
        ns = {"tl": tl, "KERNEL_CONFIGS": {"triton_poi_0": {"XBLOCK": 128}}}
        exec("def triton_fn(XBLOCK: tl.constexpr):\n    pass\n", ns)

        class FakeJitFunction:
            fn = staticmethod(ns["triton_fn"])

        class BackendAutotuner:
            def __init__(self, *args, configs, heuristic_type, inductor_meta, **kwargs):
                self.configs = configs
                self.heuristic_type = heuristic_type
                self.inductor_meta = inductor_meta

        tuners = "coordinate_descent_tuning combo_tuning_groups incremental_autotune"
        retuning = dict.fromkeys(tuners.split(), True)
        meta = {"kernel_name": "triton_poi_0", **retuning}
        kernel = cached_autotune(
            [1024],
            [],
            triton_meta={},
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta=meta,
            caching_autotuner_cls=BackendAutotuner,
        )(FakeJitFunction())
        self.assertIsInstance(kernel, BackendAutotuner)
        self.assertEqual(kernel.heuristic_type, HeuristicType.FIXED)
        (cfg,) = kernel.configs
        self.assertEqual(cfg.kwargs, {"XBLOCK": 128})
        # Tuning that would replace the pinned config is off.
        self.assertEqual(kernel.inductor_meta.keys() & retuning.keys(), set())

    def test_the_bundled_autotune_cache_key_is_the_same_with_no_rows(self):
        empty = f"{KERNEL_CONFIGS_NAME} = {{\n}}\nx = 1\n"
        row = "    'triton_poi_0': {'XBLOCK': 128},\n"
        full = empty.replace("\n", "\n" + row, 1)
        self.assertEqual(_comment_stripped_hash(empty), _comment_stripped_hash(full))

    @requires_gpu_and_triton
    def test_pinned_kernel_keeps_the_backends_extra_options(self):
        # An out-of-tree backend can tune options into Config.extra_options.
        options = {"backend_option": 2}

        def record(kernel, in_block):
            if in_block:
                kernel.launchers[0].config.extra_options = options

        x = torch.randn(64, 128, device=GPU_TYPE)
        with _spy_launches(record):
            result, code = _code_for(_softmax, x)
        ns = _load_from_file(code)
        self.assertEqual(ns["call"]([x])[0], result)
        self.assertTrue(ns["KERNEL_CONFIGS"])
        for name, cfg in ns["KERNEL_CONFIGS"].items():
            self.assertEqual(cfg["extra_options"], options, name)
            (launcher,) = ns[name].launchers
            self.assertEqual(launcher.config.extra_options, options, name)
            self.assertNotIn("extra_options", launcher.config.kwargs, name)

    @requires_gpu_and_triton
    @parametrize("field", ["extra_options", "num_ctas", "pre_hook"])
    def test_a_row_the_module_cannot_read_back_is_left_out(self, field):
        # An enum does not read back, and a row has no num_ctas or pre_hook.
        value = {
            "extra_options": {"opt": HeuristicType.FIXED},
            "num_ctas": 2,
            "pre_hook": lambda args: None,
        }[field]

        def record(kernel, in_block):
            if in_block:
                setattr(kernel.launchers[0].config, field, value)

        x = torch.randn(64, 128, device=GPU_TYPE)
        with _spy_launches(record):
            result, code = _code_for(_softmax, x)
        ns = _load_from_file(code)
        self.assertEqual(ns["KERNEL_CONFIGS"], {})
        self.assertEqual(ns["call"]([x])[0], result)

    @requires_gpu_and_triton
    def test_without_compile_time_autotuning_nothing_is_pinned(self):
        x = torch.randn(64, 128, device=GPU_TYPE)
        cfg = {"triton.autotune_at_compile_time": None}
        result, code = _code_for(_softmax, x, **cfg)
        self.assertNotIn("KERNEL_CONFIGS", code)
        self.assertEqual(_load_from_file(code)["call"]([x])[0], result)


instantiate_parametrized_tests(TestKernelConfigs)


if __name__ == "__main__":
    run_tests()
