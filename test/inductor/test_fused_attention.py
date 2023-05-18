# Owner(s): ["module: inductor"]
import itertools
import math

import torch
import torch._inductor.config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_SDPA,
    SM80OrLater,
)
from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestSDPAPatternRewriter(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    @config.patch(fallback_random=True, lowmem_dropout=False)
    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
    ):
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
        args2 = self._clone_inputs(args1)

        for training in [False, True]:
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            torch.manual_seed(1234)
            result1 = dot_prod_attention(*args1)

            counters.clear()
            torch.manual_seed(1234)
            result2, (source_code,) = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True), *args2
            )
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product",
                    source_code,
                )
            self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if isinstance(arg1, torch.Tensor) and arg1.is_floating_point():
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=1.3e-6)

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

    @config.patch(fallback_random=True, lowmem_dropout=False)
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
        def sfdp_pattern_5_v1(query, key, value):
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

        def sfdp_pattern_5_v2(query, key, value):
            # https://github.com/pytorch/pytorch/issues/100318.
            attn_mask = torch.zeros(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).bool()
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        self._check_common(sfdp_pattern_5_v1, contains=False)
        self._check_common(sfdp_pattern_5_v2, contains=False)

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

    def test_sdpa_rewriter_7(self):
        def sfdp_pattern_7(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very small dropout to make sure test passes
            attn_weight = torch.dropout(attn_weight, 0.0001, True)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_7, args, contains=SM80OrLater)

    def test_sdpa_rewriter_8(self):
        def sfdp_pattern_8(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_8, args)

    def test_sdpa_rewriter_9(self):
        def sfdp_pattern_9(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very low dropout to make test pass
            attn_weight = torch.dropout(attn_weight, 0.0001, True)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_9, args, contains=SM80OrLater)

    def test_sdpa_rewriter_10(self):
        def sfdp_pattern_10(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.empty((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_10, args)

    @config.patch(fallback_random=True, lowmem_dropout=False)
    def test_pattern_fails_with_tensor_factor(self):
        # https://github.com/pytorch/pytorch/issues/99124
        class Model(torch.nn.Module):
            def __init__(self, is_inv_factor):
                super(Model, self).__init__()
                self.is_inv_factor = is_inv_factor

            def forward(self, query, key, value, scale_factor) -> torch.Tensor:
                y = torch.matmul(query, key.transpose(-2, -1))
                if self.is_inv_factor:
                    y = y.div(scale_factor)
                else:
                    y = y.mul(scale_factor)
                return y.softmax(dim=-1).matmul(value)

        tensor_shape = (2, 4, 4, 4)
        for is_inv_factor in [True, False]:
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((4, 1, 1), device="cuda"),
            ]
            model = Model(is_inv_factor).eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    def test_pattern_fails_with_unsupported_mask(self):
        # https://github.com/pytorch/pytorch/issues/100315
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super(Model, self).__init__()

            def forward(self, query, key, value, attn_mask) -> torch.Tensor:
                attn_weight = torch.softmax(
                    query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
                    + attn_mask,
                    dim=-1,
                )
                return attn_weight @ value

        tensor_shape = (2, 4, 4, 4)

        upsupported_masks = [
            torch.randn((2, 4, 4, 4), device="cuda").to(dtype=torch.int),
            2.0,
        ]
        for atte_mask in upsupported_masks:
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                atte_mask,
            ]
            model = Model().eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA and PLATFORM_SUPPORTS_FUSED_SDPA and not TEST_WITH_ROCM:
        run_tests()
