# Owner(s): ["module: inductor"]
import importlib
from collections.abc import Callable
from typing import Any
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
from torch.testing._internal.triton_utils import requires_cuda_and_triton


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorTest(InductorTestCase):
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: dict | None = None,
        config_patches: dict | None = None,
        atol: float | None = 1e-05,
        rtol: float | None = 1e-08,
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
    def test_cse_make_block_ptr_reduction(self):
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
        a = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        b = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
            atol=1e-4,
        )
        self.count_code("= tl.make_block_ptr(in_ptr", code, 2)
        self.count_code("= tl.load(block_ptr", code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    @parametrize("disable_welford_reduction", [True, False])
    def test_disable_welford_reduction(self, disable_welford_reduction: bool):
        def func(x):
            return torch.var_mean(x, dim=1)

        # Use a reduction larger than the CUDA two-step variance threshold to
        # force codegen to prefer Welford reduction, in order to test
        # effectiveness of config flag disable_welford_reduction.
        # This test should run fine on GPU as the configuration is not specific to MTIA backend.
        x = torch.randn((4, 65536), device=torch.device(GPU_TYPE))
        config_patches = {
            "mtia.disable_welford_reduction": disable_welford_reduction,
            "triton.two_pass_variance_l2_fraction": 0.0,
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

    @requires_cuda_and_triton
    @parametrize(
        "two_pass_variance_l2_fraction, expect_welford",
        [(0.0, True), (1e-12, True), (1.0, False)],
    )
    def test_l2_cache_aware_two_step_variance(
        self, two_pass_variance_l2_fraction: float, expect_welford: bool
    ):
        def func(x):
            return torch.var_mean(x, dim=1)

        device = torch.device(GPU_TYPE)
        device_props = torch.cuda.get_device_properties(device)
        outer_dim = max(1024, device_props.multi_processor_count * 2 * 32)
        min_reduction_dim = 64
        max_l2_reduction_dim = device_props.L2_cache_size // (
            outer_dim * torch.empty((), dtype=torch.float32).element_size()
        )
        if max_l2_reduction_dim < min_reduction_dim:
            self.skipTest("CUDA device L2 is too small for a non-split Welford test")
        reduction_dim = min(2048, max_l2_reduction_dim)
        x = torch.randn((outer_dim, reduction_dim), device=device)
        config_patches = {
            "triton.two_pass_variance_l2_fraction": two_pass_variance_l2_fraction,
        }
        _, code = self.run_and_compare(
            func,
            x,
            config_patches=config_patches,
            atol=1e-2,
            rtol=1e-4,
        )

        welford_count = sum(prog.count("triton_helpers.welford") for prog in code)
        if expect_welford:
            self.assertGreater(welford_count, 0)
        else:
            self.assertEqual(welford_count, 0)

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
