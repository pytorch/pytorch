# Owner(s): ["module: inductor"]

import os
import re
import tempfile

import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
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
        self.assertTrue(re.search(r"^def triton_\w+\(", code, re.MULTILINE))

    @requires_cuda_and_triton
    @parametrize("autotune_at_compile_time", [None, True])
    def test_emitted_module_runs_standalone_and_matches_eager(
        self, autotune_at_compile_time
    ):
        # The point of the mode: the file on its own is the program. Compile it from a
        # real path -- @triton.jit resolves its own source by filename, so a hoisted
        # kernel cannot be exec'd from a bare string. autotune_at_compile_time=True is
        # what export_python compiles under.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        cfg = {"triton.autotune_at_compile_time": autotune_at_compile_time}
        expected, code = _code_for(fn, x, readable_wrapper=True, **cfg)
        self.assertEqual(expected, fn(x))
        # Only the compile-time autotune script, which execs its kernels, keeps the
        # AsyncCompile string form; the wrapper itself defines the kernel as code.
        self.assertEqual(
            code.count("= async_compile.triton("), 1 if autotune_at_compile_time else 0
        )
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
    def test_same_named_helpers_do_not_shadow(self):
        # Scan helpers are named by op sequence and numbered per kernel, so these two
        # combine_fns emit the same helper name with different constants.
        def fn(x, y):
            kw = {"dim": 0, "combine_mode": "pointwise"}
            a = associative_scan(lambda p, q: p + q + 1, x, **kw)
            return a, associative_scan(lambda p, q: p + q + 2, y, **kw)

        x, y = torch.randn(64, device="cuda"), torch.randn(64, device="cuda")
        result, code = _code_for(fn, x, y, readable_wrapper=True)
        helpers = re.findall(r"^def (_triton_helper_fn\w*)\(", code, re.MULTILINE)
        self.assertEqual(len(helpers), 2)
        self.assertEqual(len(helpers), len(set(helpers)), helpers)
        expected = fn(x, y)
        self.assertEqual(result, expected)
        self.assertEqual(self._run_standalone(code, [x, y]), expected)

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

    def test_cpu_graph_does_not_bind_gpu_allocators(self):
        def fn(x):
            return (x + 1).relu().sum(0)

        x = torch.randn(1024)
        _, code = _code_for(fn, x, readable_wrapper=True, cpu_backend="cpp")
        self.assertIn("empty_strided_cpu", code)
        self.assertNotIn("empty_strided_cuda", code)
        self.assertNotIn("empty_strided_xpu", code)

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
