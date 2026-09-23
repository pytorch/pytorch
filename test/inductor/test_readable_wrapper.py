# Owner(s): ["module: inductor"]

import contextlib
import os
import re
import tempfile
from unittest import mock

import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor import config
from torch._inductor.codegen.common import (
    get_wrapper_codegen_for_device,
    init_backend_registration,
)
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    has_cpp_wrapper_for_device,
    patch_inductor_backend,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _code_for(fn, *args, **config_kwargs):
    torch._dynamo.reset()
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn), *args)
    return result, "\n".join(codes)


@contextlib.contextmanager
def _no_runtime_tuning():
    from torch._inductor.runtime.triton_heuristics import (
        AutotuneCache,
        CachingAutotuner,
    )

    def runtime_tuning(*args, **kwargs):
        raise AssertionError("a readable module autotuned at runtime")

    with (
        mock.patch.object(CachingAutotuner, "autotune_to_one_config", runtime_tuning),
        mock.patch.object(
            CachingAutotuner, "_coordinate_descent_tuning", runtime_tuning
        ),
        # a cached best config would replace the pinned one
        mock.patch.object(AutotuneCache, "create", runtime_tuning),
    ):
        yield


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


def _cond_softmax(x):
    # Kernels in both the root and the branch subgraphs, whose text the root splices in.
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


class TestReadableWrapperCodegen(TestCase):
    """The readable wrapper emits kernels as code rather than as source strings."""

    @requires_cuda_and_triton
    def test_triton_kernel_is_defined_at_module_level(self):
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("async_compile.triton", code)
        # A module-level def, not a string: no leading indent, and the heuristics
        # decorator that builds the autotuner is right above it.
        self.assertTrue(
            re.search(r"^@triton_heuristics\.\w+\(", code, re.MULTILINE),
            code[:2000],
        )
        self.assertTrue(
            re.search(r"^def triton_\w+\(", code, re.MULTILINE), code[:2000]
        )
        # The launch site is unchanged, which is what makes the hoist a pure
        # reformatting: the name still binds a CachingAutotuner.
        self.assertIn(".run(", code)

    @requires_cuda_and_triton
    def test_every_kernel_is_defined_exactly_once(self):
        # Hoisting puts kernel names in one module namespace, so a duplicate definition
        # would silently shadow rather than fail.
        def fn(x):
            return torch.softmax(x, dim=-1).sum(0), (x * 3).relu().mean(1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        names = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertGreater(len(names), 1)
        self.assertEqual(len(names), len(set(names)), names)

    @requires_cuda_and_triton
    def test_subgraph_kernels_are_hoisted_too(self):
        # graph_partition is on by default in OSS, so the subgraph wrapper is on the
        # normal path; delegating it to the stock subgraph class would leave these
        # kernels stringified.
        def fn(p, x):
            return torch.cond(p, lambda t: t.sin(), lambda t: t.cos(), (x,))

        p = torch.tensor(True, device="cuda")
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, p, x, readable_wrapper=True)
        self.assertNotIn("async_compile.triton", code)
        # each branch's kernel is emitted by its subgraph wrapper, and both are code
        for op in ("sin", "cos"):
            pattern = rf"^def triton_\w+_{op}_\d+\("
            self.assertTrue(re.search(pattern, code, re.MULTILINE), op)
        for pred in (True, False):
            p = torch.tensor(pred, device="cuda")
            self.assertEqual(self._run_standalone(code, [p, x])[0], fn(p, x))

    @requires_cuda_and_triton
    def test_emitted_module_runs_standalone_and_matches_eager(self):
        # The point of the mode: the file on its own is the program. Compile it from a
        # real path -- @triton.jit resolves its own source by filename, so a hoisted
        # kernel cannot be exec'd from a bare string.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        expected, code = _code_for(fn, x, readable_wrapper=True)
        self.assertEqual(expected, fn(x))
        # The compile-time autotune script is dropped from the module, and the wrapper
        # itself defines the kernel as code.
        self.assertNotIn("async_compile.triton", code)
        self.assertEqual(self._run_standalone(code, [x])[0], expected)

    def _run_standalone(self, code, args):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "artifact.py")
            with open(path, "w") as f:
                f.write(code)
            ns: dict[str, object] = {"__file__": path, "__name__": "_readable_artifact"}
            with open(path) as f:
                exec(compile(f.read(), path, "exec"), ns)
            return ns["call"](args)  # type: ignore[operator]

    @requires_cuda_and_triton
    @parametrize(
        "cfg",
        [
            # coordinate_descent_tuning would have the template read the autotune cache
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "coordinate_descent_tuning": True,
            },
            {"combo_kernels": True},
        ],
        name_fn=lambda cfg: next(iter(cfg)),
    )
    def test_template_and_combo_kernels_run_standalone(self, cfg):
        # Both reach emit_triton_kernel_definition by their own route, with source of a
        # different shape from a pointwise kernel's: a Triton matmul template, and a
        # combo kernel with its module-level device functions.
        if "max_autotune" in cfg and not is_big_gpu():
            self.skipTest("Triton GEMM templates need a big GPU")

        def fn(a, b, x, y):
            return a @ b, x.sin(), y.cos()

        a, b = torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda")
        x, y = torch.randn(128, device="cuda"), torch.randn(96, device="cuda")
        result, code = _code_for(fn, a, b, x, y, readable_wrapper=True, **cfg)
        self.assertNotIn("async_compile.triton", code)
        marker = "triton_tem_" if "max_autotune" in cfg else "pid_offset"
        self.assertIn(marker, code)
        expected = fn(a, b, x, y)
        self.assertEqual(result, expected)
        with _no_runtime_tuning():
            self.assertEqual(self._run_standalone(code, [a, b, x, y]), expected)
        # A template is built with its one config, so it keeps its decorator unpinned.
        configs = code[code.index("\nKERNEL_CONFIGS = {") :].split("\n}\n")[0]
        for template in re.findall(r"^def (triton_tem_\w+)\(", code, re.MULTILINE):
            self.assertNotIn(repr(template), configs)

    @requires_cuda_and_triton
    @parametrize(
        "case",
        [
            ("persistent_reduction", _softmax, (64, 128)),
            ("reduction", lambda x: x.sum(-1), (64, 3000)),
            ("pointwise", lambda x: (x.sin() * 2).relu(), (4096,)),
            # its inductor_meta holds an AutotuneHint, which is not a literal
            (
                "bucketize",
                lambda x: torch.bucketize(x, torch.linspace(-1, 1, 8, device=x.device)),
                (4096,),
            ),
            ("subgraph", _cond_softmax, (64, 128)),
        ],
        name_fn=lambda case: case[0],
    )
    @config.patch(coordinate_descent_tuning=True, max_autotune_pointwise=True)
    def test_kernels_launch_with_their_compile_time_config(self, case):
        # A kernel's launch config decides its numerics (a reduction's block sizes set
        # its summation order), so the artifact fixes it: every kernel is pinned to the
        # config tuning chose at compile time, listed in KERNEL_CONFIGS, and nothing
        # benchmarks or coordinate-descends on first launch.
        _, fn, shape = case
        x = torch.randn(shape, device="cuda")
        # what tuning chose, read from the base class's scope, not from the table
        tuned: dict[str, object] = {}
        run_autotune_block = PythonWrapperCodegen.generate_and_run_autotune_block

        def record_tuned(self):
            scope = run_autotune_block(self)
            tuned.update(scope or {})
            return scope

        with mock.patch.object(
            PythonWrapperCodegen, "generate_and_run_autotune_block", record_tuned
        ):
            expected, code = _code_for(fn, x, readable_wrapper=True)
        decorators = re.findall(r"^@triton_heuristics\.(\w+)\(", code, re.MULTILINE)
        self.assertTrue(decorators)
        self.assertEqual(set(decorators), {"fixed_config"}, decorators)
        self.assertIn("\nKERNEL_CONFIGS = {\n", code)
        from torch._inductor.runtime.triton_heuristics import config_to_dict

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "artifact.py")
            with open(path, "w") as f:
                f.write(code)
            ns: dict[str, object] = {"__file__": path, "__name__": "_readable_artifact"}
            with _no_runtime_tuning():
                exec(compile(code, path, "exec"), ns)
                # the same configs as the compile, so the same bits
                self.assertEqual(ns["call"]([x])[0], expected, atol=0, rtol=0)  # type: ignore[operator]
                # -x takes the other torch.cond branch, so every kernel launches
                other = ns["call"]([-x])[0]  # type: ignore[operator]
                self.assertEqual(other, fn(-x), atol=1e-4, rtol=1e-4)
        kernel_configs = ns["KERNEL_CONFIGS"]
        for name, cfg in kernel_configs.items():  # type: ignore[attr-defined]
            (tuned_launcher,) = tuned[name].launchers  # type: ignore[attr-defined]
            self.assertEqual(config_to_dict(tuned_launcher.config), cfg, name)
            (launcher,) = ns[name].launchers  # type: ignore[attr-defined]
            self.assertEqual(config_to_dict(launcher.config), cfg, name)

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True)
    def test_dynamic_shape_kernel_keeps_its_pinned_config_at_every_size(self):
        # Tuned once on the size hints, it launches with that config at other sizes.
        def fn(x):
            return x.sum(-1)

        torch._dynamo.reset()
        x = torch.randn(64, 3000, device="cuda")
        with config.patch(readable_wrapper=True):
            _, codes = run_and_get_code(torch.compile(fn, dynamic=True), x)
        code = "\n".join(codes)
        self.assertEqual(
            re.findall(r"^@triton_heuristics\.(\w+)\(", code, re.MULTILINE),
            ["fixed_config"],
        )
        with _no_runtime_tuning():
            for shape in [(64, 3000), (17, 5000), (200, 129)]:
                y = torch.randn(shape, device="cuda")
                (out,) = self._run_standalone(code, [y, *shape])
                self.assertEqual(out, fn(y), atol=1e-3, rtol=1e-3)

    @requires_cuda_and_triton
    def test_triton_kernels_require_compile_time_autotuning(self):
        # Unset, the mode turns it on; turned off, the kernels could only be tuned on
        # first launch.
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(
            _softmax,
            x,
            readable_wrapper=True,
            **{"triton.autotune_at_compile_time": None},
        )
        self.assertIn("\nKERNEL_CONFIGS = {\n", code)
        with self.assertRaisesRegex(Exception, "triton.autotune_at_compile_time"):
            _code_for(
                _softmax,
                x,
                readable_wrapper=True,
                **{"triton.autotune_at_compile_time": False},
            )

    @requires_cuda_and_triton
    def test_compile_time_autotuning_is_on_before_decompositions(self):
        # repeat_interleave's data-dependent decomposition is skipped under
        # autotune_at_compile_time, whose block runs kernels on generated inputs.
        def fn(x, repeats):
            return torch.repeat_interleave(x, repeats, output_size=6)

        x = torch.randn(3, device="cuda")
        repeats = torch.tensor([1, 2, 3], device="cuda")
        result, code = _code_for(fn, x, repeats, readable_wrapper=True)
        self.assertEqual(result, fn(x, repeats))
        # decomposed, it would fuse into a Triton kernel
        self.assertIsNone(
            re.search(r"^def triton_\w*repeat_interleave", code, re.MULTILINE)
        )

    @requires_cuda_and_triton
    @config.patch(coordinate_descent_tuning=True, incremental_autotune=True)
    def test_user_kernel_does_not_read_the_autotune_cache(self):
        # A user kernel keeps filename=, so coordinate_descent_tuning in its meta would
        # have a cached best config replace its own.
        from torch.testing._internal.triton_utils import add_kernel

        def fn(x):
            out = torch.empty_like(x)
            add_kernel[(4,)](x, x, out, x.numel(), BLOCK_SIZE=64)
            return out

        x = torch.randn(256, device="cuda")
        expected, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("'coordinate_descent_tuning'", code)
        self.assertNotIn("'incremental_autotune': True", code)
        with _no_runtime_tuning():
            self.assertEqual(self._run_standalone(code, [x])[0], expected)

    @requires_cuda_and_triton
    def test_user_kernel_autotuned_over_several_configs_is_refused(self):
        # It stays a source string for AsyncCompile, whose autotuner would benchmark
        # its configs on first launch.
        from torch.testing._internal.triton_utils import add_kernel_autotuned

        def fn(x):
            out = torch.empty_like(x)
            add_kernel_autotuned[(4,)](x, x, out, x.numel())
            return out

        x = torch.randn(256, device="cuda")
        with self.assertRaisesRegex(Exception, "add_kernel_autotuned"):
            _code_for(fn, x, readable_wrapper=True)

    def test_multi_kernel_is_refused(self):
        # MultiKernelCall benchmarks its candidates on first launch.
        with self.assertRaisesRegex(Exception, "multi_kernel"):
            _code_for(
                torch.relu,
                torch.randn(8),
                readable_wrapper=True,
                **{"triton.multi_kernel": 1},
            )

    @requires_cuda_and_triton
    @parametrize("case", ["scan", "flex_attention"])
    def test_same_named_helpers_do_not_shadow(self, case):
        # Kernels define @triton.jit helpers under names unique only per kernel: scan
        # combine_fns are named by op sequence, and flex attention's template defines
        # forward_inner & co. Both kernels here define the same names, different bodies.
        if case == "scan":

            def fn(x, y):
                kw = {"dim": 0, "combine_mode": "pointwise"}
                a = associative_scan(lambda p, q: p + q + 1, x, **kw)
                return a, associative_scan(lambda p, q: p + q + 2, y, **kw)

            args = [torch.randn(64, device="cuda"), torch.randn(64, device="cuda")]
        else:
            # one tensor as q, k and v: _run_standalone passes args in graph order
            def fn(x):
                a = flex_attention(x, x, x, score_mod=lambda s, b, h, m, n: s * 2)
                return a, flex_attention(x, x, x, score_mod=lambda s, b, h, m, n: s + m)

            args = [torch.randn(1, 2, 128, 64, device="cuda")]
        result, code = _code_for(fn, *args, readable_wrapper=True)
        defs = re.findall(r"^def (\w+)\(", code, re.MULTILINE)
        self.assertEqual(len(defs), len(set(defs)), defs)
        expected = fn(*args)
        tol = {"atol": 2e-2, "rtol": 2e-2}
        self.assertEqual(result, expected, **tol)
        self.assertEqual(self._run_standalone(code, args), expected, **tol)

    @requires_cuda_and_triton
    def test_preamble_binds_only_what_the_graph_uses(self):
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        preamble = code.split("# kernel path:")[0]
        self.assertIn("empty_strided_cuda", preamble)
        self.assertIn("assert_size_stride", preamble)
        for unused in (
            "empty_strided_xpu",
            "empty_strided_mtia",
            "empty_strided_cpu_pinned",
            "alloc_from_pool",
            "maybe_profile",
            "run_intermediate_hooks",
            "import tempfile",
            "import random",
            "from ctypes import",
        ):
            self.assertNotIn(unused, preamble, f"{unused!r} kept but unused")

    def test_default_wrapper_emits_the_whole_preamble(self):
        # The preamble tables replaced two blobs for every python wrapper, not just the
        # readable one; the default wrapper must still write every entry.
        real = PythonWrapperCodegen.write_preamble_line
        with mock.patch.object(
            PythonWrapperCodegen, "write_preamble_line", autospec=True, side_effect=real
        ) as spy:
            _, code = _code_for(torch.relu, torch.randn(8), cpu_backend="cpp")
        wrapper = spy.call_args_list[0].args[0]
        table = wrapper._preamble_imports() + wrapper._preamble_bindings()
        written = {call.args[3] for call in spy.call_args_list}
        for _, line in table:
            self.assertIn(line, written)
            self.assertIn(line, code)

    def test_cpu_graph_does_not_bind_gpu_allocators(self):
        def fn(x):
            return (x + 1).relu().sum(0)

        x = torch.randn(1024)
        _, code = _code_for(fn, x, readable_wrapper=True, cpu_backend="cpp")
        self.assertIn("empty_strided_cpu", code)
        self.assertNotIn("empty_strided_cuda", code)
        self.assertNotIn("empty_strided_xpu", code)

    @requires_cuda_and_triton
    def test_async_compile_is_dropped_when_no_backend_needs_it(self):
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("AsyncCompile", code)
        self.assertNotIn("async_compile", code)
        # the separator that precedes the dropped wait/del goes with it
        self.assertNotIn("\n\n\n\n", code)

    def test_async_compile_survives_when_a_backend_needs_it(self):
        # C++ kernels cannot be hoisted -- the text is C++ and producing the value
        # needs a compiler invocation -- so the lifecycle has to stay for them. The
        # decision is per-graph, not per-mode.
        def fn(x):
            return (x + 1).relu().sum(0)

        x = torch.randn(1024)
        _, code = _code_for(fn, x, readable_wrapper=True, cpu_backend="cpp")
        self.assertIn("async_compile.cpp_pybinding(", code)
        for line in (
            "from torch._inductor.async_compile import AsyncCompile",
            "async_compile = AsyncCompile()",
            "async_compile.wait(globals())",
            "del async_compile",
        ):
            self.assertEqual(code.count(line), 1, line)
        # with its two blank separator lines, exactly as the default wrapper writes it
        self.assertIn("\n\n\nasync_compile.wait(globals())\ndel async_compile\n", code)

    @requires_cuda_and_triton
    def test_async_compile_survives_for_a_user_defined_triton_kernel(self):
        # A user @triton.jit kernel is still an async_compile.triton(...) source string,
        # the one Triton kernel that keeps the lifecycle alive in a readable module.
        from torch.testing._internal.triton_utils import add_kernel

        def fn(x):
            out = torch.empty_like(x)
            add_kernel[(4,)](x, x, out, x.numel(), BLOCK_SIZE=64)
            return out

        x = torch.randn(256, device="cuda")
        expected, code = _code_for(fn, x, readable_wrapper=True)
        self.assertEqual(expected, x + x)
        self.assertEqual(code.count("= async_compile.triton("), 1)
        for line in ("async_compile = AsyncCompile()", "async_compile.wait(globals())"):
            self.assertEqual(code.count(line), 1, line)
        self.assertEqual(self._run_standalone(code, [x])[0], expected)

    def test_cpp_kernels_in_root_and_subgraphs_run_standalone(self):
        # A subgraph's C++ kernel binds through the root's async_compile, and only the
        # root waits on it, so the lifecycle is decided once for the whole module.
        def fn(x):
            y = (x + 1).relu()
            pred = y.sum() > 0
            return torch.cond(pred, lambda t: (t * 2).sin(), lambda t: t.cos(), (y,))

        x = torch.randn(64, 128)
        expected, code = _code_for(fn, x, readable_wrapper=True, cpu_backend="cpp")
        self.assertEqual(expected, fn(x))
        # one kernel in the root and one per branch
        self.assertEqual(code.count("= async_compile.cpp_pybinding("), 3)
        self.assertEqual(code.count("async_compile.wait(globals())"), 1)
        self.assertEqual(self._run_standalone(code, [x])[0], expected)

    @requires_cuda_and_triton
    def test_a_binding_is_not_kept_alive_by_its_own_definition(self):
        # `_quantized = torch.ops._quantized` names itself on the right-hand side, so
        # any analysis that counts attribute names as uses can never drop it.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("_quantized", code)

    @requires_cuda_and_triton
    def test_a_name_mentioned_only_in_a_comment_is_not_a_use(self):
        # Inductor stamps each kernel with a provenance comment naming its source ops
        # ("Original ATen: [aten.mul, ...]"), which is not a use of the aten binding.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertIn("Original ATen: [aten.", code)
        self.assertNotIn("aten = torch.ops.aten", code)

    @requires_cuda_and_triton
    def test_a_name_used_only_inside_a_kernel_is_not_a_wrapper_use(self):
        # A kernel module supplies its own imports (`math as tl_math`) and carries
        # `'device': 0` in its metadata. Neither is a use of the wrapper's binding, and
        # once kernels are hoisted they sit in the same text as the wrapper's own code.
        x = torch.randn(64, 128, device="cuda")
        for fn in (_softmax, _cond_softmax):
            with self.subTest(fn.__name__):
                _, code = _code_for(fn, x, readable_wrapper=True)
                self.assertIn("math as tl_math", code)
                self.assertNotIn("\nimport math\n", code)
                self.assertNotIn("from torch import device, empty_strided", code)

    @requires_cuda_and_triton
    def test_names_the_wrapper_uses_are_kept(self):
        # The other direction: extern calls name `device(...)` and `inf` in the
        # wrapper's own code, so their imports must survive the trim.
        def fn(x):
            perm = torch.randperm(x.shape[0], device=x.device)
            return torch.cdist(x, x * 2, p=float("inf")).sum() + perm.sum()

        x = torch.randn(8, 4, device="cuda")
        expected, code = _code_for(fn, x, readable_wrapper=True)
        self.assertIn("device=device(type='cuda'", code)
        self.assertIn("from torch import device, empty_strided", code)
        self.assertIn("from math import inf, nan", code)
        self.assertEqual(self._run_standalone(code, [x])[0], expected)

    @requires_cuda_and_triton
    def test_no_stale_pointer_to_a_cache_file(self):
        # Inductor stamps each kernel "# kernel path: /tmp/torchinductor_.../x.py",
        # naming where the kernel WOULD have been compiled from. It is defined in this
        # file instead, and pointing a reader at a cache file is the confusion this mode
        # exists to remove.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("# kernel path:", code)
        # the rest of the provenance comment is still worth having
        self.assertIn("Original ATen:", code)

    @requires_cuda_and_triton
    def test_triton_is_not_imported_twice(self):
        # A hoisted kernel carries its own triton imports, so the wrapper's copy is dead
        # weight -- and `start_graph`/`end_graph` are only used under profile_bandwidth.
        x = torch.randn(64, 128, device="cuda")
        for fn in (_softmax, _cond_softmax):
            with self.subTest(fn.__name__):
                _, code = _code_for(fn, x, readable_wrapper=True)
                preamble = code.split("Original ATen:")[0]
                self.assertNotIn("import triton", preamble)
                self.assertNotIn("start_graph", preamble)
                # each kernel still supplies what it needs, and nothing else does
                kernels = len(re.findall(r"^def triton_\w+\(", code, re.MULTILINE))
                self.assertGreater(kernels, 0)
                for line in ("import triton\n", "import triton.language as tl\n"):
                    self.assertEqual(code.count(line), kernels, line)

    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=False)
        self.assertIn("async_compile.triton", code)
        self.assertIn("AsyncCompile()", code)

    def test_shadowing_kernel_names_are_refused(self):
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256)
        with self.assertRaisesRegex(Exception, "unique_kernel_names"):
            _code_for(
                fn, x, readable_wrapper=True, **{"triton.unique_kernel_names": False}
            )

    @parametrize("flag", ["benchmark_kernel", "benchmark_combo_kernel"])
    def test_benchmark_harness_is_refused(self, flag):
        # these append a get_args()/call()/__main__ harness to each kernel; at module
        # level those collide with each other and with the wrapper's own call.
        def fn(x):
            return (x * 2).relu()

        with self.assertRaisesRegex(Exception, flag):
            _code_for(fn, torch.randn(256), readable_wrapper=True, **{flag: True})

    @parametrize("flag", ["cpp_wrapper", "fx_wrapper"])
    def test_non_python_wrapper_is_refused(self, flag):
        # these select a different wrapper, which would silently drop readable_wrapper
        def fn(x):
            return (x * 2).relu()

        with self.assertRaisesRegex(Exception, "cpp_wrapper and fx_wrapper"):
            _code_for(fn, torch.randn(256), readable_wrapper=True, **{flag: True})

    def test_non_stock_python_wrapper_is_refused(self):
        # MTIA and out-of-tree backends register their own python wrapper, which this
        # mode cannot replace; keeping it would silently ignore the flag.
        class OutOfTreeWrapper(PythonWrapperCodegen):
            pass

        with patch_inductor_backend("cpu", python_wrapper_codegen=OutOfTreeWrapper):
            with self.assertRaisesRegex(Exception, "OutOfTreeWrapper"):
                _code_for(torch.relu, torch.randn(8), readable_wrapper=True)

    @config.patch(readable_wrapper=True)
    def test_registration_accessor_is_unaffected(self):
        # get_wrapper_codegen_for_device also reads back what a device registered (for
        # patch_inductor_backend's restore and capability probes), so the flag must not
        # leak into it, or a restore would register the readable class for good.
        init_backend_registration()
        self.assertIs(get_wrapper_codegen_for_device("cpu"), PythonWrapperCodegen)
        self.assertTrue(has_cpp_wrapper_for_device("cpu"))
        with patch_inductor_backend("cpu"):
            pass
        self.assertIs(get_wrapper_codegen_for_device("cpu"), PythonWrapperCodegen)

    def test_profile_bandwidth_output_is_refused(self):
        # profile_bandwidth_output runs the benchmark harness, which this mode drops.
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256)
        with self.assertRaisesRegex(Exception, "profile_bandwidth_output"):
            _code_for(
                fn, x, readable_wrapper=True, profile_bandwidth_output="unused.txt"
            )


instantiate_parametrized_tests(TestReadableWrapperCodegen)


if __name__ == "__main__":
    run_tests()
