# Owner(s): ["module: inductor"]

import os
import re
import subprocess
import sys
import tempfile

import torch
from torch._inductor import config
from torch._inductor.codegen.wrapper import (
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


def _code_for(fn, *args, **config_kwargs):
    torch._dynamo.reset()
    with config.patch(**config_kwargs):
        result, codes = run_and_get_code(torch.compile(fn), *args)
    return result, "\n".join(codes)


def _run_from_file(code, args):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "module.py")
        with open(path, "w") as f:
            f.write(code)
        ns = {"__file__": path, "__name__": "_wrapper_preamble"}
        exec(compile(code, path, "exec"), ns)
        return ns["call"](args)


def _softmax(x):
    return torch.softmax(x * 2, dim=-1)


# A graph partition's code, like a torch.cond branch's, is a subgraph's. Every op of
# a graph that has no CPU op lands in partition_0.
_PARTITIONED = {"triton.cudagraphs": True, "graph_partition": True}


def _cond_softmax(x):
    # Kernels in both the root and the branch subgraphs, whose text the root splices in.
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


class TestWrapperPreamble(TestCase):
    """The python wrapper imports and binds only the names its module uses."""

    def test_cpu_graph_binds_only_what_it_uses(self):
        def fn(x):
            return (x + 1).relu().sum(0)

        x = torch.randn(1024)
        result, code = _code_for(fn, x)
        self.assertIn("empty_strided_cpu = ", code)
        for unused in (
            "empty_strided_cuda",
            "empty_strided_xpu",
            "empty_strided_cpu_pinned",
            "alloc_from_pool",
            "run_intermediate_hooks",
            "import tempfile",
            "import random",
            "from ctypes import",
        ):
            self.assertNotIn(unused, code, f"{unused!r} kept but unused")
        self.assertEqual(_run_from_file(code, [x])[0], result)

    def _wrapper(self):
        return self._root_and_subgraph()[0]

    def _root_and_subgraph(self):
        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            root = PythonWrapperCodegen()
            return root, SubgraphPythonWrapperCodegen("subgraph_0", root)

    def test_a_dotted_import_is_kept_when_its_first_component_is_used(self):
        wrapper, buf = self._wrapper(), IndentedBuffer()
        wrapper.write_if_used(buf, "import os.path")
        wrapper.write_if_used(buf, "import xml.dom as dom")
        buf.writeline("p = os.path.join('a')")
        wrapper.scan_for_used_names(buf)
        self.assertEqual(buf.getvalue(), "import os.path\np = os.path.join('a')\n")

    def test_a_kept_binding_keeps_the_import_its_right_hand_side_uses(self):
        wrapper, buf = self._wrapper(), IndentedBuffer()
        for line in ("import os", "import sys", "sep = os.sep", "path = sys.path"):
            wrapper.write_if_used(buf, line)
        buf.writeline("print(sep)")
        wrapper.scan_for_used_names(buf)
        self.assertEqual(buf.getvalue(), "import os\nsep = os.sep\nprint(sep)\n")

    def test_a_line_with_explicit_names_is_not_read_as_a_binding(self):
        # Kept for its names, a line uses all it reads, x and os included; torch among
        # its names does not make it unconditional.
        wrapper, buf = self._wrapper(), IndentedBuffer()
        for line in ("import os", "import sys", "x = sys.path"):
            wrapper.write_if_used(buf, line)
        wrapper.write_if_used(buf, "y, z = os.sep, x", names=("y", "z"))
        wrapper.write_if_used(buf, "torch.manual_seed(0)", names=("torch", "seed"))
        buf.writeline("print(y)")
        wrapper.scan_for_used_names(buf)
        expected = "import os\nimport sys\nx = sys.path\ny, z = os.sep, x\nprint(y)\n"
        self.assertEqual(buf.getvalue(), expected)
        exec(buf.getvalue(), {})

    def test_a_name_only_a_subgraph_uses_keeps_the_lines_it_needs(self):
        # The subgraph's code, spliced into the root unscanned, uses sep, whose binding
        # in turn keeps import os.
        (root, subgraph), buf = self._root_and_subgraph(), IndentedBuffer()
        for line in ("import os", "import sys", "sep = os.sep"):
            root.write_if_used(buf, line)
        subgraph_code = IndentedBuffer()
        subgraph_code.writeline("print(sep)")
        subgraph.scan_for_used_names(subgraph_code)
        root.scan_for_used_names(buf)
        self.assertEqual(buf.getvalue(), "import os\nsep = os.sep\n")

    def test_a_subgraph_cannot_write_a_line_kept_only_if_used(self):
        (_, subgraph), buf = self._root_and_subgraph(), IndentedBuffer()
        subgraph.write_if_used(buf, "import os")
        with self.assertRaisesRegex(AssertionError, "subgraph_0 wrote a line"):
            subgraph.scan_for_used_names(buf)

    def test_the_scan_reads_the_text_that_is_emitted(self):
        # getrawvalue would join `y = a\` and `os.sep` into `y = aos.sep`.
        wrapper, buf = self._wrapper(), IndentedBuffer()
        wrapper.write_if_used(buf, "import os")
        buf.writelines(["y = a\\", "os.sep"])
        wrapper.scan_for_used_names(buf)
        self.assertEqual(buf.getvalue(), "import os\ny = a\\\nos.sep\n")

    def test_blank_lines_before_an_omitted_block_survive_a_splice(self):
        wrapper, buf = self._wrapper(), IndentedBuffer()
        for name in ("a", "b"):
            wrapper.write_omitted_from_scan(buf, f"\n\ndef {name}():\n    pass\n")
        module = IndentedBuffer()
        module.splice(buf)
        expected = "\n\ndef a():\n    pass\n\n\ndef b():\n    pass\n"
        self.assertEqual(module.getvalue(), expected)

    def test_an_omitted_block_cannot_be_indented(self):
        wrapper, buf = self._wrapper(), IndentedBuffer()
        with buf.indent(), self.assertRaisesRegex(AssertionError, "indent 0"):
            wrapper.write_omitted_from_scan(buf, "x = 1\ny = 2\n")

    def test_a_line_whose_bindings_cannot_be_read_raises(self):
        wrapper, buf = self._wrapper(), IndentedBuffer()
        for line in ("from m import (a, b)", "from m import *", "del x"):
            with self.assertRaisesRegex(AssertionError, "cannot tell what"):
                wrapper.write_if_used(buf, line)

    @requires_gpu_and_triton
    def test_a_binding_is_not_kept_alive_by_its_own_definition(self):
        # `_quantized = torch.ops._quantized` names itself on the right-hand side.
        _, code = _code_for(_softmax, torch.randn(64, 128, device=GPU_TYPE))
        self.assertNotIn("_quantized", code)

    @requires_gpu_and_triton
    def test_a_name_mentioned_only_in_a_comment_is_not_a_use(self):
        _, code = _code_for(_softmax, torch.randn(64, 128, device=GPU_TYPE))
        self.assertIn("Original ATen: [aten.", code)
        self.assertNotIn("aten = torch.ops.aten", code)

    @requires_gpu_and_triton
    def test_a_name_used_only_inside_a_kernel_is_not_a_wrapper_use(self):
        # A module-level kernel brings its own imports (`math as tl_math`, triton) and
        # carries `'device': 0` in its metadata. None of those is a use of the wrapper's
        # binding.
        x = torch.randn(64, 128, device=GPU_TYPE)
        for name, fn, cfg in (
            ("_softmax", _softmax, {}),
            ("_cond_softmax", _cond_softmax, {}),
            ("partition", _softmax, _PARTITIONED),
        ):
            with self.subTest(name):
                result, code = _code_for(fn, x, **cfg)
                if cfg:
                    self.assertIn("def partition_0(", code)
                self.assertIn("math as tl_math", code)
                self.assertNotIn("\nimport math\n", code)
                self.assertNotIn("start_graph", code)
                kernels = len(re.findall(r"^def triton_\w+\(", code, re.MULTILINE))
                self.assertGreater(kernels, 0)
                for line in ("import triton\n", "import triton.language as tl\n"):
                    self.assertEqual(code.count(line), kernels, line)
                self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_gpu_and_triton
    def test_names_the_wrapper_uses_are_kept(self):
        # Extern calls name `device(...)` and `inf` only through repr().
        def fn(x):
            perm = torch.randperm(x.shape[0], device=x.device)
            return torch.cdist(x, x * 2, p=float("inf")).sum() + perm.sum()

        x = torch.randn(8, 4, device=GPU_TYPE)
        result, code = _code_for(fn, x)
        self.assertIn(f"device=device(type='{GPU_TYPE}'", code)
        self.assertIn("from torch import device, empty_strided", code)
        self.assertIn("from math import inf, nan", code)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    def _check_mm_only_in_a_subgraph(self, fn, x, **cfg):
        # Only the subgraph calls extern_kernels.mm, and the root does not scan
        # subgraphs' code.
        result, code = _code_for(fn, x, **cfg)
        self.assertIn("extern_kernels.mm(", code)
        self.assertNotIn("extern_kernels.mm(", code.split("def call(")[-1])
        self.assertIn("import extern_kernels", code)
        self.assertEqual(_run_from_file(code, [x])[0], result)
        return code

    def test_a_name_used_only_in_a_cond_branch_is_kept(self):
        def fn(x):
            return torch.cond(x.sum() > 0, lambda t: t @ t, lambda t: t.cos(), (x,))

        # Positive, so the predicate takes the branch that calls mm.
        self._check_mm_only_in_a_subgraph(fn, torch.rand(16, 16))

    @requires_gpu_and_triton
    def test_a_name_used_only_in_a_graph_partition_is_kept(self):
        x = torch.randn(16, 16, device=GPU_TYPE)
        code = self._check_mm_only_in_a_subgraph(lambda x: x @ x, x, **_PARTITIONED)
        self.assertIn("def partition_0(", code)

    def test_a_graph_that_compiles_nothing_has_no_async_compile(self):
        def fn(a, b):
            return a @ b

        a, b = torch.randn(8, 8), torch.randn(8, 8)
        result, code = _code_for(fn, a, b)
        self.assertIn("extern_kernels.mm(", code)
        self.assertNotIn("async_compile", code)
        self.assertNotIn("AsyncCompile", code)
        self.assertEqual(_run_from_file(code, [a, b])[0], result)

    def test_kernels_only_a_subgraph_compiles_keep_async_compile_wait(self):
        # The root compiles nothing; only the branches' C++ kernels name async_compile.
        def fn(p, x):
            return torch.cond(p, lambda t: t.cos(), lambda t: t.sin(), (x,))

        p, x = torch.tensor(True), torch.randn(8)
        result, code = _code_for(fn, p, x)
        self.assertNotIn("cpp_fused_", code.split("def call(")[-1])
        self.assertIn("async_compile.wait(globals())\ndel async_compile", code)
        self.assertEqual(_run_from_file(code, [p, x])[0], result)

    @requires_gpu_and_triton
    def test_module_level_kernels_keep_async_compile_wait(self):
        # Nothing but the wait names async_compile, and the wait is what compiles the
        # kernels when the module runs on its own. A subgraph's kernels, a graph
        # partition's included, go into the kernel_sources it shares with the root.
        x = torch.randn(64, 128, device=GPU_TYPE)
        for name, fn, cfg in (
            ("_softmax", _softmax, {}),
            ("_cond_softmax", _cond_softmax, {}),
            ("partition", _softmax, _PARTITIONED),
        ):
            with self.subTest(name):
                _, code = _code_for(fn, x, **cfg)
                if cfg:
                    self.assertIn("def partition_0(", code)
                self.assertEqual(code.count("async_compile."), 1)
                self.assertIn("async_compile = AsyncCompile()", code)
                self.assertIn("async_compile.wait(globals())\ndel async_compile", code)

    def test_a_name_used_only_in_a_nested_subgraph_is_kept(self):
        # Only the inner true branch calls extern_kernels.mm. Its code is spliced into
        # the outer branch's, which does not scan it either.
        def fn(x):
            def outer(t):
                return torch.cond(t.mean() > 0, lambda u: u @ u, lambda u: -u, (t,))

            return torch.cond(x.sum() > 0, outer, lambda t: t.cos(), (x,))

        # Positive, so both predicates take the branch that calls mm.
        x = torch.rand(16, 16)
        result, code = _code_for(fn, x)
        # Each branch is a module-level function. The one that calls mm is called only
        # from another branch, the outer one, which call() calls.
        defs = dict(re.findall(r"(?ms)^def (\w+)\((.*?)(?=^\S|\Z)", code))
        (inner,) = [name for name, body in defs.items() if "extern_kernels.mm(" in body]
        callers = [n for n, body in defs.items() if f"{inner}(" in body]
        self.assertEqual(len(callers), 1, callers)
        self.assertNotIn(f"{inner}(", code.split("def call(")[-1])
        self.assertIn(f"{callers[0]}(", code.split("def call(")[-1])
        self.assertIn("import extern_kernels", code)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_gpu_and_triton
    def test_profile_bandwidth_keeps_start_graph(self):
        x = torch.randn(64, 128, device=GPU_TYPE)
        result, code = _code_for(_softmax, x, profile_bandwidth=True)
        self.assertIn("start_graph", code.split("def call(")[0])
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_gpu_and_triton
    def test_autotune_at_compile_time_runs_from_file(self):
        x = torch.randn(64, 128, device=GPU_TYPE)
        at_compile_time = {"triton.autotune_at_compile_time": True}
        result, code = _code_for(_softmax, x, **at_compile_time)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_gpu_and_triton
    def test_benchmark_harness_runs(self):
        # The harness is appended to the module last, and is run as a script.
        x = torch.randn(64, 128, device=GPU_TYPE)
        _, code = _code_for(_softmax, x)
        self.assertIn("def benchmark_compiled_module(", code)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "module.py")
            with open(path, "w") as f:
                f.write(code)
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
            subprocess.check_call([sys.executable, path], env=env)


if __name__ == "__main__":
    run_tests()
