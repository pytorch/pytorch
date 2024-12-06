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

    def run_and_check(self, fn, args, *, expect_kernel_count=1):
        expected = fn(*args)
        fn = torch.compile(fn, fullgraph=True)
        result, (source_code,) = run_and_get_code(fn, *args)
        self.assertEqual(result, expected)
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

    def test_no_redundant_assignment(self):
        def fn(x):
            return x + 2

        args = [torch.rand([4]).to(dtype=torch.complex64)]
        expected = fn(*args)
        fn = torch.compile(fn, fullgraph=True)
        result, (source_code,) = run_and_get_code(fn, *args)
        self.assertEqual(result, expected)
        if "async_compile.multi_kernel" in source_code:
            return
        self.assertEqual(source_code.count("buf1 = buf0"), 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests(needs="filelock")
