# Owner(s): ["module: inductor"]
import functools
import itertools
import math

import torch
import torch._inductor.config
import torch.utils.checkpoint
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM80OrLater,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args)

    return inner


class TestSDPAPatternRewriterTemplate(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
        check_train=True,
        override_check_equal=False,
        dtype=torch.float,
        rtol=1.3e-6,
    ):
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        for training in [False, True] if check_train else [False]:
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            counters.clear()
            torch.manual_seed(1234)
            result2, (source_code,) = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True),
                *(args2 + dropout_arg),
            )
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product",
                    source_code,
                )

            # some tests configured with very low dropout where we still want to check equality
            if not has_dropout or override_check_equal:
                self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and (not has_dropout or override_check_equal)
                    ):
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=rtol)

    @skipIfRocm
    def _test_sdpa_rewriter_1(self):
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

        for dtype in [torch.float, torch.half]:
            if self.device == "cpu" and dtype == torch.half:
                continue
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            self._check_common(dot_prod_attention, dtype=dtype, atol=0.001, rtol=rtol)
            self._check_common(
                checkpoint_wrapper(dot_prod_attention),
                dtype=dtype,
                atol=0.001,
                rtol=rtol,
            )

    def _test_pattern_fails_with_reuse(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """

        @skipIfRocm
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
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)

    @skipIfRocm
    def _test_sdpa_rewriter_2(self):
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
        self._check_common(checkpoint_wrapper(dot_prod_attention))

    def _test_sdpa_rewriter_3(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training: bool
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_4(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                p=0.2,
                inplace=False,
                training=training,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_5(self):
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
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v1), contains=False)
        self._check_common(sfdp_pattern_5_v2, contains=False)
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v2), contains=False)

    @skipIfRocm
    def _test_sdpa_rewriter_6(self):
        def sfdp_pattern_6(query, key, value, training):
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
            attn_weight = torch.nn.functional.dropout(attn_weight, 0.5, training)
            return attn_weight @ value

        self._check_common(sfdp_pattern_6, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_6), contains=False, has_dropout=True
        )

    @skipIfRocm
    def _test_sdpa_rewriter_7(self):
        def sfdp_pattern_7(query, key, value, training):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # Set to False
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_7,
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_7),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

    @skipIfRocm
    def _test_sdpa_rewriter_8(self):
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
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(sfdp_pattern_8, args, atol=2e-3)

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_8), args, atol=2e-3)

    @skipIfRocm
    def _test_sdpa_rewriter_9(self):
        def sfdp_pattern_9(query, key, value, training):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very low dropout to make test pass
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_9,
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_9),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

    @skipIfRocm
    def _test_sdpa_rewriter_10(self):
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
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(sfdp_pattern_10, args, atol=2e-3)

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_10), args, atol=2e-3)

    def _test_pattern_fails_with_tensor_factor(self):
        # https://github.com/pytorch/pytorch/issues/99124
        class Model(torch.nn.Module):
            def __init__(self, is_inv_factor):
                super().__init__()
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
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn((4, 1, 1), device=self.device),
            ]
            model = Model(is_inv_factor).eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    def _test_pattern_fails_with_unsupported_mask(self):
        # https://github.com/pytorch/pytorch/issues/100315
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, query, key, value, attn_mask) -> torch.Tensor:
                attn_weight = torch.softmax(
                    query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
                    + attn_mask,
                    dim=-1,
                )
                return attn_weight @ value

        tensor_shape = (2, 4, 4, 4)

        upsupported_masks = [
            torch.randn((2, 4, 4, 4), device=self.device).to(dtype=torch.int),
            2.0,
        ]
        for atte_mask in upsupported_masks:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                atte_mask,
            ]
            model = Model().eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    @skipIfRocm
    def _test_sdpa_rewriter_11(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        self._check_common(dot_prod_attention)

    def _test_sdpa_rewriter_12(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            return torch.nn.functional.dropout(
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(v),
                p=0.4,
                training=training,
                inplace=False,
            )

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

    @skipIfRocm
    def _test_sdpa_prev_13(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        self._check_common(dot_prod_attention, check_train=False)
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)

    @skipIfRocm
    def _test_sdpa_prev_14(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        self._check_common(dot_prod_attention, check_train=False)
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)

    @skipIfRocm
    def _test_sdpa_prev_15(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(v)
            )

        self._check_common(dot_prod_attention, check_train=False)

    @skipIfRocm
    def _test_sdpa_rewriter_13(self, dtype):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
            attn_weight = torch.nn.functional.dropout(
                attn_weight, p=0.5, training=training
            )
            return torch.bmm(attn_weight, value)

        tensor_shape = (4, 8, 16)
        args = [
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
        ]

        self._check_common(
            dot_prod_attention,
            check_train=False,
            args1=args,
            has_dropout=True,
            override_check_equal=True,
            atol=1e-2,
            rtol=1e-2,
        )


if HAS_CUDA and PLATFORM_SUPPORTS_FUSED_ATTENTION:

    class SDPAPatternRewriterCudaTests(TestSDPAPatternRewriterTemplate):
        device = "cuda"
        test_sdpa_rewriter_1_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        )
        test_pattern_fails_with_reuse_cuda = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        )
        test_sdpa_rewriter_3_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_3
        )
        test_sdpa_rewriter_4_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_4
        )
        test_sdpa_rewriter_5_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        )
        test_sdpa_rewriter_6_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_6
        )
        test_sdpa_rewriter_7_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_7
        )
        test_sdpa_rewriter_8_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_8
        )
        test_sdpa_rewriter_9_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_9
        )
        test_sdpa_rewriter_10_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_10
        )
        test_pattern_fails_with_tensor_factor_cuda = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_cuda = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_cuda = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_cuda = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_cuda = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_cuda = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        test_sdpa_rewriter_13_cuda = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.half
        )


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        device = "cpu"
        test_sdpa_rewriter_1_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        test_pattern_fails_with_reuse_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        test_sdpa_rewriter_5_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        test_pattern_fails_with_tensor_factor_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        test_sdpa_rewriter_13_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.float32
        )


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
