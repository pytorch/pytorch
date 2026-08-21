# Owner(s): ["module: inductor"]
import importlib
import unittest.mock
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorGeneric(InductorTestCase):
    hw_classification = HardwareClassification.GENERIC

    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: dict | None = None,
        config_patches: dict | None = None,
        atol: float | None = 1e-05,
        rtol: float | None = 1e-08,
    ):
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

        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual, atol=atol, rtol=rtol))

        return result, code

    def count_code(self, substr: str, code: list[str], expected: int | None):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @parametrize("force_pointwise_cat", [False, True])
    def test_force_pointwise_cat(self, force_pointwise_cat: bool):
        def func(a, b):
            return torch.cat([a + 1, b + 2], dim=0)

        a = torch.randn(1024)
        b = torch.randn(1024)
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


class CodegenInductorTest(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: dict | None = None,
        config_patches: dict | None = None,
        atol: float | None = 1e-05,
        rtol: float | None = 1e-08,
    ):
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

        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual, atol=atol, rtol=rtol))

        return result, code

    def count_code(self, substr: str, code: list[str], expected: int | None):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @onlyAccelerator
    def test_cse_make_block_ptr_reduction(self, device):
        if self.device_type == "mps":
            self.skipTest("Triton is not available for MPS")

        def func(a, b):
            tmp0 = a * b
            tmp1 = a + b
            c = tmp0 + tmp1
            return c.sum(dim=0)

        config_patches = {
            "triton.use_block_ptr": True,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
            "split_reductions": False,
        }
        a = torch.randn((512, 4096), device=device)
        b = torch.randn((512, 4096), device=device)
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
            atol=1e-4,
        )
        self.count_code("= tl.make_block_ptr(in_ptr", code, 2)
        self.count_code("= tl.load(block_ptr", code, 2)

    @onlyAccelerator
    def test_block_ptr_falls_back_when_api_missing(self, device):
        if self.device_type == "mps":
            self.skipTest("Triton is not available for MPS")

        def func(a, b):
            tmp0 = a * b
            tmp1 = a + b
            c = tmp0 + tmp1
            return c.sum(dim=0)

        config_patches = {
            "triton.use_block_ptr": True,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
            "split_reductions": False,
            "force_disable_caches": True,
        }
        a = torch.randn((512, 4096), device=device)
        b = torch.randn((512, 4096), device=device)
        with unittest.mock.patch(
            "torch._inductor.codegen.triton_utils.has_triton_block_ptr",
            lambda: False,
        ):
            _, code = self.run_and_compare(
                func,
                a,
                b,
                config_patches=config_patches,
                atol=1e-4,
            )
        self.count_code("tl.make_block_ptr", code, 0)

    @onlyAccelerator
    @parametrize("disable_welford_reduction", [True, False])
    def test_disable_welford_reduction(self, disable_welford_reduction: bool, device):
        if self.device_type == "mps":
            self.skipTest("Triton is not available for MPS")

        def func(x):
            return torch.var_mean(x, dim=1)

        x = torch.randn((4, 65536), device=device)
        config_patches = {
            "mtia.disable_welford_reduction": disable_welford_reduction,
        }
        _, code = self.run_and_compare(
            func,
            x,
            config_patches=config_patches,
            atol=1e-2,
            rtol=1e-4,
        )

        welford_count = sum(prog.count("triton_helpers.welford") for prog in code)
        if disable_welford_reduction:
            self.assertEqual(welford_count, 0)
        else:
            self.assertGreater(welford_count, 0)

    @onlyAccelerator
    def test_kernel_fusion_thresholds(self, device):
        if self.device_type == "mps":
            self.skipTest("Triton is not available for MPS")

        def func(a, b):
            tmp0 = a + 1
            tmp1 = tmp0 + 2
            tmp2 = tmp1 + 3
            tmp3 = tmp2 + b
            return tmp0, tmp2, tmp3

        a = torch.randn(1024, device=device)
        b = torch.randn(1024, device=device)
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


instantiate_device_type_tests(
    CodegenInductorTest, globals(), allow_xpu=True, except_for="cpu"
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
