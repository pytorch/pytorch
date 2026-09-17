# Owner(s): ["module: inductor"]

import os
import re
import tempfile

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _code_for(fn, *args, **config_kwargs):
    torch._dynamo.reset()
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn), *args)
    return result, "\n".join(codes)


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
    def test_async_compile_is_dropped_when_no_backend_needs_it(self):
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=True)
        self.assertNotIn("AsyncCompile()", code)
        self.assertNotIn("async_compile.wait", code)
        self.assertNotIn("del async_compile", code)

    def test_async_compile_survives_when_a_backend_needs_it(self):
        # C++ kernels cannot be hoisted -- the text is C++ and producing the value
        # needs a compiler invocation -- so the lifecycle has to stay for them. The
        # decision is per-graph, not per-mode.
        def fn(x):
            return (x + 1).relu().sum(0)

        x = torch.randn(1024)
        _, code = _code_for(fn, x, readable_wrapper=True)
        if "async_compile.cpp" not in code:
            self.skipTest("graph did not lower to a C++ kernel")
        self.assertIn("AsyncCompile()", code)
        self.assertIn("async_compile.wait", code)

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
    def test_emitted_module_runs_standalone_and_matches_eager(self):
        # The point of the mode: the file on its own is the program. Compile it from a
        # real path -- @triton.jit resolves its own source by filename, so a hoisted
        # kernel cannot be exec'd from a bare string.
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        expected, code = _code_for(fn, x, readable_wrapper=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "artifact.py")
            with open(path, "w") as f:
                f.write(code)
            ns: dict[str, object] = {"__file__": path, "__name__": "_readable_artifact"}
            with open(path) as f:
                exec(compile(f.read(), path, "exec"), ns)
            got = ns["call"]([x])  # type: ignore[operator]
        self.assertEqual(got[0], expected)

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
    def test_default_wrapper_still_uses_async_compile(self):
        def fn(x):
            return torch.softmax(x * 2, dim=-1)

        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(fn, x, readable_wrapper=False)
        self.assertIn("async_compile.triton", code)
        self.assertIn("AsyncCompile()", code)

    @requires_cuda_and_triton
    def test_shadowing_kernel_names_are_refused(self):
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256, device="cuda")
        with self.assertRaisesRegex(Exception, "unique_kernel_names"):
            _code_for(
                fn, x, readable_wrapper=True, **{"triton.unique_kernel_names": False}
            )

    @requires_cuda_and_triton
    def test_benchmark_kernel_is_refused(self):
        # benchmark_kernel appends a get_args()/call()/__main__ harness to each kernel;
        # at module level those collide with each other and with the wrapper's own call.
        def fn(x):
            return (x * 2).relu()

        x = torch.randn(256, device="cuda")
        with self.assertRaisesRegex(Exception, "benchmark_kernel"):
            _code_for(fn, x, readable_wrapper=True, benchmark_kernel=True)


if __name__ == "__main__":
    run_tests()
