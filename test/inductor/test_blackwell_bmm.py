# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.autows_utils import meta_ws_enabled
from torch._inductor.kernel.bmm import (
    append_blackwell_bmm_choice,
    BlackwellBMMConfig,
)
from torch._inductor.lowering import lowerings
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


@torch.library.custom_op("inductor_test::blackwell_bmm", mutates_args={})
def blackwell_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b)


@blackwell_bmm.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], a.shape[1], b.shape[2]))


@torch.library.custom_op("inductor_test::blackwell_bmm_flat", mutates_args={})
def blackwell_bmm_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b).flatten(0, 1)


@blackwell_bmm_flat.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0] * a.shape[1], b.shape[2]))


@unittest.skipUnless(HAS_GPU and SM100OrLater, "requires NVIDIA SM100+")
class TestBlackwellBMM(TestCase):
    def _run(self, broadcast_b: bool) -> None:
        bsz, m, k, n = 3, 256, 8193, 128
        a_storage = torch.randn(bsz, k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.transpose(1, 2)
        if broadcast_b:
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            b = b.unsqueeze(0).expand(bsz, -1, -1)
        else:
            b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
        template_config = BlackwellBMMConfig(
            block_m=128,
            block_n=128,
            block_k=128,
            num_stages=3,
            epilogue_subtile=1,
        )

        def lowering(a_node, b_node):
            layout = ir.FixedLayout(
                a_node.get_device(),
                a_node.get_dtype(),
                [bsz, m, n],
                [m * n, n, 1],
            )
            choices = []
            append_blackwell_bmm_choice(
                choices, (a_node, b_node), layout, config=template_config
            )
            return choices[0].output_node()

        with (
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(
                    lambda x, y: blackwell_bmm(x, y), fullgraph=True
                ),
                a,
                b,
            )

        torch.testing.assert_close(actual, torch.bmm(a, b), atol=16.0, rtol=1e-1)
        self.assertIn("make_tensor_descriptor", codes[0])
        self.assertNotIn("two_ctas=True", codes[0])
        if meta_ws_enabled():
            self.assertIn("data_partition_factor=", codes[0])
        else:
            self.assertNotIn("data_partition_factor", codes[0])

    def test_1cta_strided_batch(self):
        self._run(False)

    def test_1cta_broadcast_batch_stride(self):
        self._run(True)

    @unittest.skipUnless(meta_ws_enabled(), "2CTA Blackwell BMM requires MetaWS")
    def test_2cta_flat_output(self):
        bsz, m, k, n = 2, 256, 8193, 128
        a_storage = torch.randn(bsz, k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.transpose(1, 2)
        b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
        template_config = BlackwellBMMConfig(
            block_m=128,
            block_n=128,
            block_k=64,
            num_stages=4,
            epilogue_subtile=1,
            two_ctas=True,
        )

        def lowering(a_node, b_node):
            layout = ir.FixedLayout(
                a_node.get_device(),
                a_node.get_dtype(),
                [bsz * m, n],
                [n, 1],
            )
            choices = []
            append_blackwell_bmm_choice(
                choices, (a_node, b_node), layout, config=template_config
            )
            return choices[0].output_node()

        with (
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm_flat.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(blackwell_bmm_flat, fullgraph=True), a, b
            )

        torch.testing.assert_close(
            actual.view(bsz, m, n), torch.bmm(a, b), atol=16.0, rtol=1e-1
        )
        self.assertIn("TWO_CTAS : tl.constexpr = True", codes[0])
        self.assertIn("ctas_per_cga=(2, 1, 1)", codes[0])
        self.assertIn("make_tensor_descriptor(out_ptr0", codes[0])

    def test_tuned_bmm_choice_smoke(self):
        bsz, m, k, n = 3, 256, 8193, 128
        a_storage = torch.randn(bsz, k, m, device="cuda", dtype=torch.bfloat16)
        a = a_storage.transpose(1, 2)
        b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="ATEN,TRITON",
            compile_threads=1,
            **{
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_blackwell_bmm_template": True,
            },
        ):
            actual = torch.compile(torch.bmm, fullgraph=True)(a, b)
        torch.testing.assert_close(actual, torch.bmm(a, b), atol=16.0, rtol=1e-1)


if __name__ == "__main__":
    if HAS_GPU and HAS_CPU:
        run_tests()
