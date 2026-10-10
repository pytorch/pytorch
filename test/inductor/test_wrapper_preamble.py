# Owner(s): ["module: inductor"]

import os
import re
import subprocess
import sys
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

    @requires_cuda_and_triton
    def test_a_binding_is_not_kept_alive_by_its_own_definition(self):
        # `_quantized = torch.ops._quantized` names itself on the right-hand side.
        _, code = _code_for(_softmax, torch.randn(64, 128, device="cuda"))
        self.assertNotIn("_quantized", code)

    @requires_cuda_and_triton
    def test_a_name_mentioned_only_in_a_comment_is_not_a_use(self):
        _, code = _code_for(_softmax, torch.randn(64, 128, device="cuda"))
        self.assertIn("Original ATen: [aten.", code)
        self.assertNotIn("aten = torch.ops.aten", code)

    @requires_cuda_and_triton
    def test_a_name_used_only_inside_a_kernel_is_not_a_wrapper_use(self):
        # A module-level kernel brings its own imports (`math as tl_math`, triton) and
        # carries `'device': 0` in its metadata. None of those is a use of the wrapper's
        # binding.
        x = torch.randn(64, 128, device="cuda")
        result, code = _code_for(_softmax, x)
        self.assertIn("math as tl_math", code)
        self.assertNotIn("\nimport math\n", code)
        self.assertNotIn("start_graph", code)
        kernels = len(re.findall(r"^def triton_\w+\(", code, re.MULTILINE))
        self.assertGreater(kernels, 0)
        for line in ("import triton\n", "import triton.language as tl\n"):
            self.assertEqual(code.count(line), kernels, line)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_cuda_and_triton
    def test_names_the_wrapper_uses_are_kept(self):
        # Extern calls name `device(...)` and `inf` only through repr().
        def fn(x):
            perm = torch.randperm(x.shape[0], device=x.device)
            return torch.cdist(x, x * 2, p=float("inf")).sum() + perm.sum()

        x = torch.randn(8, 4, device="cuda")
        result, code = _code_for(fn, x)
        self.assertIn("device=device(type='cuda'", code)
        self.assertIn("from torch import device, empty_strided", code)
        self.assertIn("from math import inf, nan", code)
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_cuda_and_triton
    def test_profile_bandwidth_keeps_start_graph(self):
        x = torch.randn(64, 128, device="cuda")
        result, code = _code_for(_softmax, x, profile_bandwidth=True)
        self.assertIn("start_graph", code.split("def call(")[0])
        self.assertEqual(_run_from_file(code, [x])[0], result)

    @requires_cuda_and_triton
    def test_benchmark_harness_runs(self):
        # The harness is appended to the module last, and is run as a script.
        x = torch.randn(64, 128, device="cuda")
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
