# Owner(s): ["module: inductor"]
import math
import unittest
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.kernel.decompose_k import (
    append_blackwell_decompose_k_partial_choice,
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    blackwell_decomposeK,
)
from torch._inductor.lowering import lowerings
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


K_SPLIT = 8


@torch.library.custom_op(
    "inductor_test::blackwell_decompose_k_partial", mutates_args={}
)
def blackwell_decompose_k_partial(
    a: torch.Tensor, b: torch.Tensor, two_ctas: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    m_pad = math.ceil(m / 128) * 128
    block_k = 64 if two_ctas else 128
    k_part = math.ceil(math.ceil(k / K_SPLIT) / block_k) * block_k
    out = torch.zeros((K_SPLIT, m_pad, n), device=a.device, dtype=torch.float32)
    for split in range(K_SPLIT):
        begin = split * k_part
        end = min(begin + k_part, k)
        if begin < end:
            out[split, :m] = torch.mm(
                a[:, begin:end], b[begin:end], out_dtype=torch.float32
            )
    return out.view(K_SPLIT * m_pad, n)


@blackwell_decompose_k_partial.register_fake
def _(a: torch.Tensor, b: torch.Tensor, two_ctas: bool) -> torch.Tensor:
    del two_ctas
    m_pad = math.ceil(a.shape[0] / 128) * 128
    return a.new_empty((K_SPLIT * m_pad, b.shape[1]), dtype=torch.float32)


@unittest.skipUnless(
    HAS_GPU and SM100OrLater,
    "requires NVIDIA SM100+",
)
class TestBlackwellDecomposeKPartial(TestCase):
    def _run(self, two_ctas: bool) -> None:
        config_index = 1 if two_ctas else 0
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]

        def lowering(a, b, two_ctas_arg):
            if bool(two_ctas_arg) != two_ctas:
                raise AssertionError("unexpected 2CTA specialization")
            m, _ = map(int, a.get_size())
            n = int(b.get_size()[1])
            m_tiles = math.ceil(m / partial_config.block_m)
            if partial_config.two_ctas:
                m_tiles = math.ceil(m_tiles / 2) * 2
            m_pad = m_tiles * partial_config.block_m
            layout = ir.FixedLayout(
                a.get_device(),
                torch.float32,
                [K_SPLIT * m_pad, n],
                [n, 1],
            )
            choices = []
            append_blackwell_decompose_k_partial_choice(
                choices,
                (a, b),
                layout,
                k_split=K_SPLIT,
                config=partial_config,
            )
            return choices[0].output_node()

        m, k, n = 256, 8193, 128
        a_storage = torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.T
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def fn(x, y):
            partial = blackwell_decompose_k_partial(x, y, two_ctas)
            return partial.view(K_SPLIT, m, n).sum(0).to(torch.bfloat16)

        with (
            mock.patch.dict(
                lowerings,
                {
                    torch.ops.inductor_test.blackwell_decompose_k_partial.default: lowering
                },
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)

        expected = a @ b
        torch.testing.assert_close(actual, expected, atol=16.0, rtol=1e-1)
        self.assertIn("make_tensor_descriptor", codes[0])
        self.assertIn("make_tensor_descriptor(out_ptr0, shape=[2048, 128]", codes[0])
        self.assertIn("BATCH_SIZE : tl.constexpr = 8", codes[0])
        self.assertIn("DESCRIPTOR_K : tl.constexpr = 8193", codes[0])
        k_part = math.ceil(math.ceil(k / K_SPLIT) / partial_config.block_k)
        k_part *= partial_config.block_k
        self.assertIn(f"K_BATCH_OFFSET : tl.constexpr = {k_part}", codes[0])
        self.assertIn(
            f"K_TILES : tl.constexpr = {k_part // partial_config.block_k}", codes[0]
        )
        self.assertNotIn("DECOMPOSE_K", codes[0])
        self.assertEqual("TWO_CTAS : tl.constexpr = True" in codes[0], two_ctas)

    def test_1cta(self):
        self._run(False)

    def test_2cta(self):
        self._run(True)

    def test_tuned_mm_choice_smoke(self):
        m, k, n = 256, 8193, 128
        a_storage = torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.T
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="ATEN,TRITON",
            compile_threads=1,
            assume_aligned_inputs=True,
            **{
                "triton.enable_template_tma_store": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_blackwell_decompose_k_partial": True,
                "triton.num_decompose_k_splits": 4,
                "triton.use_tensor_descriptor": True,
                "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
            },
        ):
            actual = torch.compile(lambda x, y: x @ y, fullgraph=True)(a, b)

        torch.testing.assert_close(actual, a @ b, atol=16.0, rtol=1e-1)

    def test_complete_plan_direct(self):
        m, k, n = 256, 8193, 128
        a_storage = torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.T
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        with config.patch(
            compile_threads=1,
            **{
                "triton.enable_template_tma_store": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.use_tensor_descriptor": True,
            },
        ):
            actual = torch.compile(
                lambda x, y: blackwell_decomposeK(x, y, 33, 0), fullgraph=True
            )(a, b)
        torch.testing.assert_close(actual, a @ b, atol=16.0, rtol=1e-1)


if __name__ == "__main__":
    if HAS_GPU and HAS_CPU:
        run_tests()
