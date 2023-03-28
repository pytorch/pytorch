# Owner(s): ["module: inductor"]
import itertools
import math
import unittest

import torch
import torch._inductor.config
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_SDPA, TEST_CUDA
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


def skips(fn):
    fn = unittest.skipIf(not TEST_CUDA, "CUDA not available")(fn)
    fn = unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )(fn)
    return fn


class TestSDPAPatternRewriter(TestCase):
    def _check_common(self, dot_prod_attention, args1=None):
        tensor_shape = (4, 2, 16, 32)
        if args1 is None:
            args1 = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
        args2 = [*map(torch.clone, args1)]

        for training in [False, True]:
            for x in itertools.chain(args1[:3], args2[:3]):
                x.requires_grad = training

            result1 = dot_prod_attention(*args1)
            result2, (source_code,) = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True), *args2
            )
            self.assertIn("aten._scaled_dot_product_efficient_attention", source_code)
            self.assertEqual(result1, result2)

            if training:
                result1.sum().backward()
                result2.sum().backward()

                self.assertEqual(args1[0].grad, args2[0].grad)
                self.assertEqual(args1[1].grad, args2[1].grad)
                self.assertEqual(args1[2].grad, args2[2].grad)

    @skips
    def test_sdpa_rewriter_1(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        self._check_common(dot_prod_attention)

    @skips
    def test_pattern_fails_with_reuse(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """

        @torch.compile(fullgraph=True)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            attn_weights = (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
            )
            return attn_weights.matmul(value), attn_weights

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)

    @skips
    def test_sdpa_rewriter_2(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        self._check_common(dot_prod_attention)

    @skips
    @unittest.skip("need to fix issue with dropout")
    def test_sdpa_rewriter_3(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                p=0.4,
                training=True,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention)

    @skips
    @unittest.skip("need to fix issue with dropout")
    def test_sdpa_rewriter_4(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                p=0.2,
                training=True,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention)

    @skips
    @unittest.skip("this pattern just gets re-expanded in dispatcher")
    def test_sdpa_rewriter_5(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(0.1)
                .masked_fill(attn_mask, float("-inf"))
                .softmax(dim=-1)
                .matmul(value)
            )

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn((1, 1, 8, 8), device="cuda") > 0,
        ]
        self._check_common(dot_prod_attention, args)

    @skips
    @unittest.skip("need to fix issue with dropout")
    def test_sdpa_rewriter_6(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1))
                .div(0.1)
                .masked_fill(attn_mask, float("-inf"))
                .softmax(dim=-1),
                p=0.3,
                training=True,
                inplace=False,
            ).matmul(value)

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn((1, 1, 8, 8), device="cuda") > 0,
        ]
        self._check_common(dot_prod_attention, args)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
