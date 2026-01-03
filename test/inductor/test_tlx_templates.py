# Owner(s): ["module: inductor"]
import unittest
from functools import reduce

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._triton import has_datacenter_blackwell_tma_device


def has_tlx() -> bool:
    """Check if TLX (Triton Language eXtensions) is available."""
    try:
        import triton.language.extra.tlx  # noqa: F401

        return True
    except ImportError:
        return False


torch.set_float32_matmul_precision("high")


@instantiate_parametrized_tests
class TestTLXTemplates(TestCase):
    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device() or not config.is_fbcode(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipIf(not has_tlx(), "TLX not available")
    @parametrize("template", ("blackwell_gemm_clc", "blackwell_gemm_2cta"))
    @parametrize("dynamic", (False, True))
    @parametrize("epilogue_subtile", (False, True))
    def test_tlx_mm(
        self,
        template: str,
        dynamic: bool,
        epilogue_subtile: bool,
    ):
        a_transposed: bool = False
        b_transposed: bool = False

        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            # a = a.repeat(8, 8)
            # b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        def next_multiple_16(a: int) -> int:
            return ((a + 15) // 16) * 16

        M, N, K = 4096, 4096, 4096
        a_shape = (K, M) if a_transposed else (M, K)
        a_stride = (
            (next_multiple_16(M), 1) if a_transposed else (next_multiple_16(K), 1)
        )
        a = torch.empty_strided(a_shape, a_stride, dtype=torch.float16).to(GPU_TYPE)
        a[:] = torch.randn(a_shape, dtype=torch.float16)
        a = a.to(GPU_TYPE)
        b_shape = (N, K) if b_transposed else (K, N)
        b_stride = (
            (next_multiple_16(K), 1) if a_transposed else (next_multiple_16(N), 1)
        )
        b = torch.empty_strided(b_shape, b_stride, dtype=torch.float16)
        b[:] = torch.randn(b_shape, dtype=torch.float16)
        b = b.to(GPU_TYPE)

        with config.patch(
            {
                "force_disable_caches": True,
                "enable_caching_generated_triton_templates": False,
                "triton.enable_tlx_templates": True,
                "max_autotune": True,
                "test_configs.autotune_choice_name_regex": template,
                "triton.enable_epilogue_subtiling": epilogue_subtile,
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected)
        expected_strings = {
            "blackwell_gemm_clc": [
                "tlx.clc_create_context",
                "tlx.clc_producer",
                "tlx.clc_consumer",
            ],
            "blackwell_gemm_2cta": [
                "'ctas_per_cga': (2, 1, 1)",
                "tlx.cluster_cta_rank",
                "pred=pred_cta0",
                "two_ctas=True",
            ],
        }
        assert template in expected_strings
        file_check = reduce(
            lambda fc, s: fc.check(s), expected_strings[template], FileCheck()
        )
        file_check.run(code[0])


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
