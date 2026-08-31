# Owner(s): ["module: inductor"]
import math

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCPU,
    dtypesIfCUDA,
    dtypesIfXPU,
    instantiate_device_type_tests,
    onlyCPU,
    skipXPUIf,
)


class LowPrecisionTest(TestCase):
    @skipXPUIf(
        True,
        "truncf/fpext barriers may be folded (intel-xpu-backend-for-triton#7491)",
    )
    @dtypesIfCPU(torch.bfloat16)
    @dtypesIfCUDA(torch.bfloat16, torch.float16)
    @dtypesIfXPU(torch.bfloat16, torch.float16)
    @dtypes(torch.bfloat16, torch.float16)
    @config.patch(emulate_precision_casts=False)
    def test_pointwise_rounds_before_comparison(self, device, dtype):
        torch.manual_seed(0)
        values = torch.randn((6, 4, 1, 33), device=device, dtype=dtype)
        scale = math.sqrt(32.0)

        def producer(x):
            return ((x / scale) * 0.7) * 0.9

        scaled = producer(values)
        maximum = scaled.amax(dim=-1, keepdim=True)

        def fn(maximum, values):
            scaled = producer(values)
            return (maximum == scaled).sum(dim=-1, keepdim=True)

        expected = fn(maximum, values)
        fused = ((values.float() / scale) * 0.7) * 0.9
        fused_result = (maximum.float() == fused).sum(dim=-1, keepdim=True)
        self.assertNotEqual(fused_result, expected)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), maximum, values
        )
        self.assertEqual(actual, expected)

        def compare_realized(maximum, scaled):
            return (maximum == scaled).sum(dim=-1, keepdim=True)

        actual, (realized_code,) = run_and_get_code(
            torch.compile(compare_realized, backend="inductor", fullgraph=True),
            maximum,
            scaled,
        )
        self.assertEqual(actual, compare_realized(maximum, scaled))

        if device == "cuda":
            lowp_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
            self.assertEqual(code.count(f".to(tl.{lowp_name})"), 3)
            self.assertNotIn(f".to(tl.{lowp_name})", realized_code)

    @onlyCPU
    @dtypes(torch.float16)
    @config.patch(emulate_precision_casts=False)
    def test_saved_backward_comparison_does_not_round_forward(self, device, dtype):
        input = torch.tensor(
            [0.09051513671875, -8.1796875],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        target = torch.tensor([1.0, -1.0], device=device, dtype=dtype)

        def fn(input, target):
            return torch.nn.functional.hinge_embedding_loss(
                input,
                target,
                margin=-7.167470387213217,
                reduction="sum",
            )

        expected = fn(input.float(), target.float()).to(dtype)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(input, target)
        self.assertEqual(actual, expected, atol=0, rtol=0)


instantiate_device_type_tests(
    LowPrecisionTest,
    globals(),
    only_for=("cpu", "cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
