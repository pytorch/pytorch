# Owner(s): ["module: inductor"]

import os
import re
import tempfile
from unittest import mock

import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor import config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_big_gpu, run_and_get_code
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


def _cond_softmax(x):
    # Kernels in both the root and the branch subgraphs, whose text the root splices in.
    return torch.cond(
        x.sum() > 0, lambda t: torch.softmax(t * 2, dim=-1), lambda t: t.cos(), (x,)
    )


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


class TestDefaultWrapper(TestCase):
    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        self.assertIn("async_compile.triton(", code)


instantiate_parametrized_tests(TestModuleLevelKernels)


if __name__ == "__main__":
    run_tests()
