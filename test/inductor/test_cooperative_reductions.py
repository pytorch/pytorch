# Owner(s): ["module: inductor"]
import unittest
from typing import Any, Dict, List, Type

import sympy

import torch
import torch._inductor
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import FixedTritonConfig, TritonKernel
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import assert_close
from torch.testing._internal.common_cuda import IS_SM89
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


@config.patch(
    {
        "triton.cooperative_reductions": True,
        "triton.force_cooperative_reductions": True,
    }
)
@instantiate_parametrized_tests
class CooperativeReductionTests(TestCase):
    def setUp(self):
        super().setUp()
        torch._inductor.metrics.generated_kernel_count = 0
        torch._dynamo.reset()

    def run_and_check(self, fn, args, dtype=None, *, expect_kernel_count=1):
        # Define fixed tolerances
        RTOL = 1e-5
        ATOL = 1e-6

        # calculate reference value in higher precision when input dtype is float16
        ref_dtype = dtype
        if dtype == torch.float16:
            ref_dtype = torch.float64

        # Cast to the determined reference dtype
        args_ref = [tensor.to(ref_dtype) for tensor in args]

        # Calculate expected output
        raw_expected = fn(*args_ref)

        if isinstance(raw_expected, (tuple, list)):
            # If it's a tuple or list, apply .to(dtype) to each tensor within it
            # Also, handle cases where dtype might not be provided (e.g., for bool reductions)
            if dtype is not None:
                expected = type(raw_expected)(
                    [
                        t.to(dtype) if isinstance(t, torch.Tensor) else t
                        for t in raw_expected
                    ]
                )
            else:
                expected = type(raw_expected)(
                    [
                        t.to(torch.float64) if isinstance(t, torch.Tensor) else t
                        for t in raw_expected
                    ]
                )
        else:
            # If it's a single tensor
            if dtype is not None:
                expected = raw_expected.to(dtype)
            else:
                expected = raw_expected.to(torch.float64)

        fn_compiled = torch.compile(fn, fullgraph=True)
        result, (source_code,) = run_and_get_code(fn_compiled, *args)

        # For comparison, ensure result is also a tuple/list if expected is
        if isinstance(expected, (tuple, list)):
            if isinstance(result, torch.Tensor):
                result = (result,)
            elif not isinstance(result, type(expected)):
                result = type(expected)(result)

            if dtype is not None:
                result = type(result)(
                    [t.to(dtype) if isinstance(t, torch.Tensor) else t for t in result]
                )
            else:
                result = type(result)(
                    [
                        t.to(torch.float64) if isinstance(t, torch.Tensor) else t
                        for t in result
                    ]
                )
        else:
            if dtype is not None and isinstance(result, torch.Tensor):
                result = result.to(dtype)
            elif isinstance(result, torch.Tensor):
                result = result.to(torch.float64)

        # Apply assert_close with fixed tolerances for tensor comparisons
        if isinstance(result, torch.Tensor) and isinstance(expected, torch.Tensor):
            assert_close(result, expected, rtol=RTOL, atol=ATOL)
        elif isinstance(result, (tuple, list)) and isinstance(expected, (tuple, list)):
            # Iterate through elements for comparison
            for r_item, e_item in zip(result, expected):
                if isinstance(r_item, torch.Tensor) and isinstance(
                    e_item, torch.Tensor
                ):
                    assert_close(r_item, e_item, rtol=RTOL, atol=ATOL)
                else:
                    # Fallback to assertEqual for non-tensor elements (e.g., bool, int)
                    self.assertEqual(r_item, e_item)
        else:
            # Fallback to assertEqual for other types not handled by assert_close
            self.assertEqual(result, expected)

        if "@triton_heuristics.fixed_config" in source_code:
            self.assertIn("cooperative_reduction_grid", source_code)
        else:
            self.assertIn("@triton_heuristics.cooperative_reduction", source_code)
        if "async_compile.multi_kernel" not in source_code:
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expect_kernel_count
            )
        return source_code
    @parametrize(
        "name",
        [
            "sum",
            "mean",
            "prod",
            "amin",
            "amax",
            "min",
            "max",
            "var_mean",
            "std",
            "softmax",
        ],
    )
    @parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_reduction_fns(self, name, dtype):
        if IS_SM89 and dtype == torch.float64 and name in ["std", "var_mean"]:
            raise unittest.SkipTest("Timeouts on SM89")

        def fn(x, y):
            return reduction_fn(x + y, dim=-1)

        reduction_fn = getattr(torch, name)
        args = [torch.randn(1, 1024**2, device="cuda", dtype=dtype) for _ in range(2)]
        self.run_and_check(fn, args)

    def test_bool_reduction_fns(self):
        def fn(x, y):
            return [
                torch.any(x == y),
                torch.all(x == y),
                torch.any(x != y),
                torch.all(x != y),
                torch.any(x < y),
                torch.all(x > y),
            ]

        args = [torch.randn(1024, device="cuda") for _ in range(2)]
        source_code = self.run_and_check(fn, args)
        if "async_compile.multi_kernel" in source_code:
            return
        before, after = source_code.split("triton_helpers.x_grid_barrier")
        self.assertEqual(before.count("if rsplit_id == ("), 0)
        self.assertEqual(after.count("if rsplit_id == ("), 6)

    @parametrize("bs", [1, 2, 5, 15])
    @parametrize("count", [1024**2 + 1, 1024**2 - 1, 1024])
    def test_non_power_of_2(self, bs, count):
        def fn(x):
            return x.mean(), x.std() + x.min()

        args = [torch.randn([bs, count], device="cuda")]
        self.run_and_check(fn, args)

    def test_chained_reductions(self):
        def fn(x):
            for _ in range(8):
                x = x + torch.softmax(x, 1)
            return x

        args = [torch.randn(4, 100000, device="cuda")]
        source_code = self.run_and_check(fn, args)
        if "async_compile.multi_kernel" in source_code:
            return
        self.assertEqual(source_code.count("triton_helpers.x_grid_barrier"), 16)
        self.assertEqual(source_code.count("empty_strided_cuda"), 5)

    def test_reduce_split(self):
        def fn(a, b):
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device="cuda"),
            torch.rand(20, 20, device="cuda"),
        ]
        self.run_and_check(fn, inps, expect_kernel_count=2)


@config.patch("triton.persistent_reductions", not config.triton.persistent_reductions)
class NoPersistCooperativeReductionTests(CooperativeReductionTests):
    pass


@config.patch("triton.multi_kernel", int(not config.triton.multi_kernel))
class MultiKernelCooperativeReductionTests(CooperativeReductionTests):
    pass


@config.patch(
    {
        "triton.cooperative_reductions": True,
    }
)
@instantiate_parametrized_tests
class TestFixedConfigs(TestCase):
    @parametrize(
        "persistent,cooperative,cfg",
        [
            (False, False, {"XBLOCK": 1, "RBLOCK": 128}),
            (False, False, {"XBLOCK": 2, "RBLOCK": 128}),
            (True, False, {"XBLOCK": 1}),
            (True, False, {"XBLOCK": 2}),
            (False, True, {"XBLOCK": 1, "RBLOCK": 128, "RSPLIT": 16}),
            (False, True, {"XBLOCK": 2, "RBLOCK": 128, "RSPLIT": 16}),
            (True, True, {"XBLOCK": 1, "RSPLIT": 16}),
            (True, True, {"XBLOCK": 2, "RSPLIT": 16}),
        ],
    )
    def test_fixed_configs(self, persistent, cooperative, cfg):
        class MyHeuristics(InductorChoices):
            def triton_kernel_kwargs(
                self,
                kernel_cls: Type[TritonKernel],
                features: SIMDKernelFeatures,
                groups: List[sympy.Expr],
                kernel_kwargs: Dict[str, Any],
            ) -> Dict[str, Any]:
                return {
                    **kernel_kwargs,
                    "override_cooperative_reduction": cooperative,
                    "override_persistent_reduction": persistent,
                    "fixed_config": FixedTritonConfig(cfg),
                }

        def fn(x):
            return torch.softmax(x + 1, dim=-1) + x

        args = [torch.randn(8, 8000, device="cuda")]
        with torch._inductor.virtualized.V.set_choices_handler(MyHeuristics()):
            expected = fn(*args)
            fn = torch.compile(fn, fullgraph=True)
            result, (source_code,) = run_and_get_code(fn, *args)
            self.assertEqual(result, expected)
            self.assertIn("@triton_heuristics.fixed_config(", source_code)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests(needs="filelock")
