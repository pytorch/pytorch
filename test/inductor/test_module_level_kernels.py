# Owner(s): ["module: inductor"]

import re

import torch
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


class TestModuleLevelKernels(TestCase):
    @requires_cuda_and_triton
    @parametrize("fn", [_softmax, _cond_softmax])
    def test_kernels_are_defined_at_module_level(self, fn):
        x = torch.randn(64, 128, device="cuda")
        result, code = _code_for(fn, x, **{"triton.module_level_kernels": True})
        self.assertEqual(result, fn(x))
        self.assertNotIn("async_compile.triton", code)
        self.assertTrue(re.search(r"^@triton_heuristics\.\w+\(", code, re.MULTILINE))
        kernels = re.findall(r"^def (triton_\w+)\(", code, re.MULTILINE)
        self.assertTrue(kernels, code)
        self.assertEqual(len(kernels), len(set(kernels)), kernels)
        # Unlike readable_wrapper, the provenance comment keeps the kernel's cache file.
        self.assertIn("# kernel path:", code)

    @requires_cuda_and_triton
    def test_default_wrapper_still_uses_async_compile(self):
        x = torch.randn(64, 128, device="cuda")
        _, code = _code_for(_softmax, x)
        self.assertIn("async_compile.triton(", code)

    @requires_cuda_and_triton
    @parametrize(
        "patch",
        [
            {"triton.unique_kernel_names": False},
            {"benchmark_kernel": True},
            {"benchmark_combo_kernel": True},
        ],
    )
    def test_shadowing_configs_are_refused(self, patch):
        x = torch.randn(64, 128, device="cuda")
        with self.assertRaisesRegex(Exception, "module_level_kernels"):
            _code_for(_softmax, x, **{"triton.module_level_kernels": True}, **patch)


instantiate_parametrized_tests(TestModuleLevelKernels)


if __name__ == "__main__":
    run_tests()
