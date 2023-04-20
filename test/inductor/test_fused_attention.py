# Owner(s): ["module: inductor"]
import itertools
import math

import torch
import torch._inductor.config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_SDPA
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestSDPAPatternRewriter(TestCase):
    @config.patch(fallback_random=True, lowmem_dropout=False)
    def _check_common(self, dot_prod_attention, args1=None, contains=True):
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

            torch.manual_seed(1234)
            result1 = dot_prod_attention(*args1)

            counters.clear()
            torch.manual_seed(1234)
            result2, (source_code,) = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True), *args2
            )
            self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product_efficient_attention", source_code
                )
            self.assertEqual(result1, result2)

            if training:
                result1.sum().backward()
                result2.sum().backward()

                self.assertEqual(args1[0].grad, args2[0].grad)
                self.assertEqual(args1[1].grad, args2[1].grad)
                self.assertEqual(args1[2].grad, args2[2].grad)

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

        self._check_common(dot_prod_attention, contains=False)

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

        self._check_common(dot_prod_attention, contains=False)

    def test_sdpa_rewriter_5(self):
        def sfdp_pattern_5(query, key, value):
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        self._check_common(sfdp_pattern_5, contains=False)

    def test_sdpa_rewriter_6(self):
        def sfdp_pattern_6(query, key, value):
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            attn_weight = torch.dropout(attn_weight, 0.5, True)
            return attn_weight @ value

        self._check_common(sfdp_pattern_6, contains=False)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA and PLATFORM_SUPPORTS_FUSED_SDPA:
        run_tests()
