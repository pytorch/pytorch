# Owner(s): ["module: inductor"]
import importlib
from typing import Any, Callable, Optional
from unittest import skipIf

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorTest(InductorTestCase):
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: Optional[dict] = None,
        config_patches: Optional[dict] = None,
    ):
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}
        if config_patches is None:
            config_patches = {}

        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        with config.patch(config_patches):
            compiled = torch.compile(func, backend="inductor", **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual))

        return result, code

    def count_code(self, substr: str, code: list[str], expected: Optional[int]):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @parametrize("force_pointwise_cat", [False, True])
    def test_force_pointwise_cat(self, force_pointwise_cat: bool):
        def func(a, b):
            return torch.cat([a + 1, b + 2], dim=0)

        a = torch.randn(1024, device=torch.device("cpu"))
        b = torch.randn(1024, device=torch.device("cpu"))
        config_patches = {
            "force_pointwise_cat": force_pointwise_cat,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )

        reinterpret_call = (
            "= reinterpret_tensor_wrapper("
            if config.cpp_wrapper
            else "= reinterpret_tensor("
        )
        if force_pointwise_cat:
            self.count_code(reinterpret_call, code, 0)
        else:
            self.count_code(reinterpret_call, code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    def test_kernel_fusion_thresholds(self):
        def func(a, b):
            tmp0 = a + 1
            tmp1 = tmp0 + 2
            tmp2 = tmp1 + 3
            tmp3 = tmp2 + b
            return tmp0, tmp2, tmp3

        a = torch.randn(1024, device=torch.device(GPU_TYPE))
        b = torch.randn(1024, device=torch.device(GPU_TYPE))
        config_patches = {
            "max_fusion_size": 1,
            "realize_reads_threshold": 1,
            "realize_opcount_threshold": 1,
            "inplace_buffers": False,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )
        self.count_code("@triton.jit", code, 3)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or HAS_CPU:
        run_tests(needs="filelock")
