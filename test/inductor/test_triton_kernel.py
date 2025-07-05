# Owner(s): ["module: inductor"]
import unittest
from typing import Any

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


class TestingHeuristics(InductorChoices):
    def __init__(self, *, cooperative: bool, persistent: bool, cfg: dict[str, int]):
        super().__init__()
        self.cooperative = cooperative
        self.persistent = persistent
        self.cfg = cfg
        self.call_count = 0

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self.call_count += 1
        return {
            **kernel_kwargs,
            "override_cooperative_reduction": self.cooperative,
            "override_persistent_reduction": self.persistent,
            "fixed_config": FixedTritonConfig(self.cfg),
        }


@config.patch(
    {
        "triton.cooperative_reductions": True,
    }
)
@instantiate_parametrized_tests
class TestFixedConfigs(TestCase):
    def _check(self, fn, args, *, persistent=False, cooperative=True, cfg):
        expected = fn(*args)
        heuristic = TestingHeuristics(
            persistent=persistent, cooperative=cooperative, cfg=cfg
        )
        with torch._inductor.virtualized.V.set_choices_handler(heuristic):
            result, (source_code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), *args
            )
        self.assertEqual(result, expected)
        self.assertEqual(heuristic.call_count, 1)
        self.assertIn("@triton_heuristics.fixed_config(", source_code)

    @parametrize(
        "x_size,y_size",
        [
            (2, 5001),
            (5001, 2),
            (21, 432),
            (5000, 5001),
            (3214, 5001),
            (1, 1),
            (2, 2),
            (0, 0),
            (123, 321),
            (222, 122),
            (32, 231),
        ],
    )
    def test_fixed_configs(self,x_size,y_size):
        @torch.compile(dynamic=True)
        def fn(x):
            return torch.max(x, -1)
        new_fn = torch.compile(fn)
        input_tensor = torch.randn([x_size,y_size]).to(device="cuda:0")
        result = new_fn(input_tensor)
        self.assertEqual(np.max(result.values.cpu().numpy()),np.max(input_tensor.cpu().numpy()))
        


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests(needs="filelock")
