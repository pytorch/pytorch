# Owner(s): ["oncall: cpu inductor"]

import shutil
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import HardwareClassification, IS_WINDOWS


class TestWindowsInt8WeightOnlyGemm(TestCase):
    hw_classification = HardwareClassification.CPU

    @unittest.skipUnless(IS_WINDOWS, "requires Windows CPU Inductor")
    @config.patch(
        freezing=True,
        max_autotune=True,
        max_autotune_gemm_backends="CPP",
        **{"cpp.simdlen": 256, "cpp.weight_prepack": True},
    )
    def test_frozen_int8_weight_only_gemm(self):
        if shutil.which("cl") is None:
            self.skipTest("requires MSVC in PATH")
        if not torch.cpu._is_avx2_supported():
            self.skipTest("requires AVX2")

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                rng = torch.Generator().manual_seed(2026)
                self.register_buffer(
                    "weight",
                    torch.randint(
                        -127, 128, (3072, 1024), dtype=torch.int8, generator=rng
                    ),
                )
                # A power-of-two scale avoids introducing rounding in scaling.
                self.register_buffer("scale", torch.full((3072,), 0.015625).bfloat16())

            def forward(self, x):
                return torch.mm(x, self.weight.t().to(x.dtype)) * self.scale

        model = Linear().eval()
        x = torch.randn(8, 1024, generator=torch.Generator().manual_seed(2027))
        x = x.bfloat16()
        with torch.inference_mode():
            expected = x.double() @ model.weight.double().t() * model.scale.double()
            before = counters["inductor"]["cpp_templated_kernel_counter"]
            actual, code = run_and_get_code(torch.compile(model, fullgraph=True), x)

        self.assertGreater(counters["inductor"]["cpp_templated_kernel_counter"], before)
        source = "\n".join(code)
        self.assertRegex(
            source, r"_mm_prefetch\s*\(\s*reinterpret_cast<const char\s*\*>\s*\(\s*B\b"
        )
        self.assertRegex(source, r"convert_to_int32\s*<\s*int8_t\s*>")
        self.assertNotIn("extern_kernels._weight_int8pack_mm", source)
        torch.testing.assert_close(actual, expected.bfloat16(), atol=0.025, rtol=0.016)


if __name__ == "__main__":
    run_tests()
