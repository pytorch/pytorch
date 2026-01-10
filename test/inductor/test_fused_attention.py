# Owner(s): ["module: inductor"]
import functools
import itertools
import math

import torch
import torch._inductor.config
import torch.utils.checkpoint
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM80OrLater,
)
from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


class TestSDPAPatternRewriterTemplate(TestCase):
    use_static_shapes = True

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
            if training and self.device == "xpu":
                # Intel GPU have not implemented sdpa backward yet mode.
                # TODO: remove this when sdpa backward is implemented for XPU.
                continue
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            if not self.use_static_shapes:
                torch._dynamo.mark_dynamic(args2[0], 0)
                torch._dynamo.mark_dynamic(args2[1], 0)
                torch._dynamo.mark_dynamic(args2[2], 0)

            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            counters.clear()
            torch.manual_seed(1234)
            result2, source_code = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True),
                *(args2 + dropout_arg),
            )
            source_code = "\n".join(source_code)
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
            atol = 0.001
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            if self.device in ["cpu", "xpu"] and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            self._check_common(dot_prod_attention, dtype=dtype, atol=atol, rtol=rtol)
            self._check_common(
                checkpoint_wrapper(dot_prod_attention),
                dtype=dtype,
                atol=atol,
                rtol=rtol,
            )

    @torch._inductor.config.patch("freezing", True)
    def _test_sdpa_rewriter_1_freezing(self):
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

        for dtype in [torch.half]:
            atol = 0.001
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            if self.device in ["cpu", "xpu"] and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            with torch.no_grad():
                self._check_common(
                    dot_prod_attention,
                    dtype=dtype,
                    atol=atol,
                    rtol=rtol,
                    check_train=False,
                )

    def _test_insignificant_strides(self):
        if self.device == "xpu":
            self.skipTest(
                "The operator 'aten::_scaled_dot_product_efficient_attention'"
                " is not currently implemented for the XPU device. "
            )
        f32 = torch.float32

        # repro taken from https://github.com/pytorch/pytorch/issues/124289
        # constant_pad_nd is a single element tensor that gets expanded

        def forward(
            permute_3: "f32[1, 32, 1, 128]",
            permute_4: "f32[1, 32, 1, 128]",
            permute_5: "f32[1, 32, 1, 128]",
            permute_6: "f32[1, 1, 64]",
            mul_2: "f32[1, 1, 1, 1]",
        ):
            cat = torch.ops.aten.cat.default([permute_6, permute_6], 2)
            permute_6 = None
            cos = torch.ops.aten.cos.default(cat)
            sin = torch.ops.aten.sin.default(cat)
            unsqueeze_10 = torch.ops.aten.unsqueeze.default(cos, 1)
            cos = None
            unsqueeze_11 = torch.ops.aten.unsqueeze.default(sin, 1)
            sin = None
            mul_5 = torch.ops.aten.mul.Tensor(permute_3, unsqueeze_10)
            slice_10 = torch.ops.aten.slice.Tensor(permute_3, 3, 0, 64)
            slice_11 = torch.ops.aten.slice.Tensor(
                permute_3, 3, 64, 9223372036854775807
            )
            permute_3 = None
            neg = torch.ops.aten.neg.default(slice_11)
            slice_11 = None
            cat_1 = torch.ops.aten.cat.default([neg, slice_10], 3)
            neg = slice_10 = None
            mul_6 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_11)
            cat_1 = None
            add_1 = torch.ops.aten.add.Tensor(mul_5, mul_6)
            mul_5 = mul_6 = None
            mul_7 = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_10)
            unsqueeze_10 = None
            slice_12 = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64)
            slice_13 = torch.ops.aten.slice.Tensor(
                permute_4, 3, 64, 9223372036854775807
            )
            permute_4 = None
            neg_1 = torch.ops.aten.neg.default(slice_13)
            slice_13 = None
            cat_2 = torch.ops.aten.cat.default([neg_1, slice_12], 3)
            neg_1 = slice_12 = None
            mul_8 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_11)
            cat_2 = unsqueeze_11 = None
            add_2 = torch.ops.aten.add.Tensor(mul_7, mul_8)
            mul_7 = mul_8 = None
            slice_14 = torch.ops.aten.slice.Tensor(mul_2, 0, 0, 9223372036854775807)
            mul_2 = None
            slice_15 = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 9223372036854775807)
            slice_14 = None
            slice_16 = torch.ops.aten.slice.Tensor(slice_15, 2, 0, 9223372036854775807)
            slice_15 = None
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                slice_16, [0, 7], 0.0
            )
            slice_16 = None
            slice_17 = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1)
            constant_pad_nd = None
            expand_5 = torch.ops.aten.expand.default(slice_17, [1, 32, 1, 1])
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    add_1, add_2, permute_5, expand_5, True
                )
            )
            return _scaled_dot_product_efficient_attention

        kwargs = aot_graph_input_parser(forward, device=GPU_TYPE)
        # runs successfully
        out_eager = forward(**kwargs)
        out_c = torch.compile(forward)(**kwargs)
        # dont compare philox_seed/offset
        torch.testing.assert_close(out_eager[0:2], out_c[0:2])

    def _test_pattern_fails_with_reuse(self):
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
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)

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
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_7),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

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
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_8), args, atol=2e-3)

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
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_9),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

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
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_10), args, atol=2e-3)

    def _test_pattern_fails_with_tensor_factor(self):
        # https://github.com/pytorch/pytorch/issues/99124
        class Model(torch.nn.Module):
            def __init__(self, is_inv_factor):
                super().__init__()
                self.is_inv_factor = is_inv_factor

            def forward(self, query, key, value, scale_factor) -> torch.Tensor:
                # Dividing by scale_factor makes scale_factor gradients very
                # unstable
                scale_factor = scale_factor.detach()
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
                model, args1=args, contains=False, atol=1e-3, has_fuse_pattern=False
            )

    def _test_pattern_fails_with_unsupported_mask(self):
        if not self.use_static_shapes:
            self.skipTest("Causes shape specialization. TODO: investigate")

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

    def _test_sdpa_rewriter_14(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask)
                .softmax(dim=-1)
                .matmul(v)
            )

        self._check_common(dot_prod_attention)

    def _test_sdpa_rewriter_15(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False)

    def _test_sdpa_rewriter_16(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                torch.nn.functional.dropout(
                    (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                        dim=-1
                    ),
                    p=0.4,
                    training=training,
                    inplace=False,
                )
                .to(dtype=query.dtype)
                .matmul(v)
            )

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention, args1=args, contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_16_fp32_mask(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.randn(
                query.size(1), key.size(1), dtype=torch.float, device=query.device
            ).tril(diagonal=0)
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                torch.nn.functional.dropout(
                    (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                        dim=-1
                    ),
                    p=0.4,
                    training=training,
                    inplace=False,
                )
                .to(dtype=query.dtype)
                .matmul(v)
            )

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention, args1=args, contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_17(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = torch.nn.functional.dropout(
                weights,
                p=0.4,
                training=training,
                inplace=False,
            )
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

    def _test_sdpa_rewriter_18(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
        ) -> torch.Tensor:
            # for hf_GPT2 with dropout
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (), math.sqrt(value.size(-1)), dtype=query.dtype, device=query.device
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            return (
                (
                    torch.nn.functional.dropout(
                        attn_weights.softmax(dim=-1), 0.0
                    ).matmul(value)
                ),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
            )

        tensor_shape = (4, 2, 16, 32)
        causal_mask = torch.ones(2, 2, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

    def _test_sdpa_rewriter_19(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
            attn_mask: torch.Tensor,
            training,
        ) -> torch.Tensor:
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (),
                math.sqrt(value.size(-1)),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            attn_weights = attn_weights + attn_mask
            attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
            return torch.nn.functional.dropout(
                attn_weights,
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        tensor_shape = (4, 2, 16, 32)
        causal_mask = torch.ones(16, 16, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )
        attn_mask = torch.randn((16, 16), dtype=torch.float, device=self.device)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=False,
        )

    def _test_sdpa_rewriter_20(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            q = q / math.sqrt(q.size(-1))
            scores = torch.matmul(q, k.transpose(-2, -1))
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = torch.nn.functional.dropout(
                weights,
                p=0.4,
                training=training,
                inplace=False,
            )
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

    def _test_sdpa_rewriter_21(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value)

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.float, device=self.device)
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                attn_mask,
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
            )

    def _test_sdpa_rewriter_22(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value), key, value

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            attn_mask = torch.randn((1, 1, 2, 2), dtype=torch.float, device=self.device)
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                attn_mask,
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
            )
            # test attn_mask with stride of last dim != 1
            attn_mask_ = attn_mask.transpose(2, 3)
            args[3] = attn_mask_
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
                contains=self.device == "cpu",
            )

    def _test_sdpa_rewriter_23(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            attn_mask = torch.full((1, 1, 1, 2), 0.0, device=query.device)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value), key, value

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
            )

    def _test_sdpa_rewriter_24(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            bs = query.size(0)
            n_head = query.size(1)
            seq_len = query.size(2)
            embed_dim = query.size(3)
            q = query.view(bs * n_head, seq_len, embed_dim)
            k = key.reshape(bs * n_head, seq_len, embed_dim)
            v = value.reshape(bs * n_head, seq_len, embed_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(bs, n_head, seq_len, seq_len) + attn_mask
            attn_weights = attn_weights.view(bs * n_head, seq_len, seq_len)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.view(bs, n_head, seq_len, embed_dim)
            return attn_output

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 16, 16), dtype=torch.float, device=self.device)
        args = [
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=False,
            check_train=False,
        )

    def _test_sdpa_rewriter_25(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )
            return attn_weights.matmul(value)

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.half, device=self.device)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )

    def _test_sdpa_rewriter_26(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )

            return attn_weights.matmul(value), key, value

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.half, device=self.device)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )

    def _test_sdpa_rewriter_27(self):
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            attn_mask = torch.full(
                (1, 1, 1, 2), 0.0, dtype=torch.half, device=query.device
            )
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )
            return attn_weights.matmul(value), key, value

        tensor_shape = (4, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )


if HAS_XPU_AND_TRITON or (HAS_CUDA_AND_TRITON and PLATFORM_SUPPORTS_FUSED_ATTENTION):

    class SDPAPatternRewriterGpuTests(TestSDPAPatternRewriterTemplate):
        device = GPU_TYPE
        test_sdpa_rewriter_1_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        test_sdpa_rewriter_1_freezing = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1_freezing
        )
        test_insignificant_strides = (
            TestSDPAPatternRewriterTemplate._test_insignificant_strides
        )
        test_pattern_fails_with_reuse_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        test_sdpa_rewriter_3_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_3
        test_sdpa_rewriter_4_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_4
        test_sdpa_rewriter_5_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        test_sdpa_rewriter_6_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_6
        test_sdpa_rewriter_7_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_7
        test_sdpa_rewriter_8_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_8
        test_sdpa_rewriter_9_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_9
        test_sdpa_rewriter_10_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_10
        )
        test_pattern_fails_with_tensor_factor_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        test_sdpa_rewriter_13_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.half
        )
        test_sdpa_rewriter_14_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_14
        )
        test_sdpa_rewriter_15_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_15
        )
        # Pattern 16 is disabled on NVIDIA CUDA (disable_cuda=True) but enabled on ROCm
        if TEST_WITH_ROCM:
            test_sdpa_rewriter_16_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16
            )
            test_sdpa_rewriter_16_fp32_mask_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16_fp32_mask
            )
        test_sdpa_rewriter_17_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_17
        )
        test_sdpa_rewriter_18_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )
        test_sdpa_rewriter_20_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_20
        )
        test_sdpa_rewriter_21_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_21
        )
        test_sdpa_rewriter_22_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_22
        )
        test_sdpa_rewriter_23_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_23
        )
        test_sdpa_rewriter_24_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_24
        )
        if HAS_XPU_AND_TRITON:
            test_sdpa_rewriter_25_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_25
            )
            test_sdpa_rewriter_26_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_26
            )
            test_sdpa_rewriter_27_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_27
            )

    class SDPAPatternRewriterGpuDynamicTests(SDPAPatternRewriterGpuTests):
        use_static_shapes = False


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
        test_sdpa_rewriter_14_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_14
        )
        test_sdpa_rewriter_15_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_15
        )
        test_sdpa_rewriter_16_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16
        )
        test_sdpa_rewriter_16_fp32_mask_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16_fp32_mask
        )
        test_sdpa_rewriter_17_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_17
        )
        test_sdpa_rewriter_18_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )
        test_sdpa_rewriter_20_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_20
        )
        test_sdpa_rewriter_21_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_21
        )
        test_sdpa_rewriter_22_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_22
        )
        test_sdpa_rewriter_23_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_23
        )
        test_sdpa_rewriter_24_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_24
        )

    class SDPAPatternRewriterCpuDynamicTests(SDPAPatternRewriterCpuTests):
        use_static_shapes = False


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
