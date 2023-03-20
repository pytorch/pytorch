# Owner(s): ["module: inductor"]
import math
import re
import unittest
from types import FunctionType

import torch
import torch._inductor.config
import torch._inductor.sdpa_pattern_rewriter as sdpa_pattern_rewriter
import torch.functional as F
import torch.nn as nn
from torch._dynamo.test_case import run_tests, TestCase
from torch.backends.cuda import sdp_kernel
from torch.fx import GraphModule
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_SDPA, TEST_CUDA
from torch.testing._internal.common_utils import freeze_rng_state, IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class CfgNode:
    n_embd: int = 128
    n_head: int = 4
    resid_pdrop: float = 0.2
    attn_pdrop: float = 0.1
    block_size: int = 12
    vocab_size: int = 1000
    n_layer = 12

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# CausalSelfAttention block from Karpathy's MinGPT
# copied from https://github.com/karpathy/minGPT/blob/90420ee978fed95e6eb7c9add728d33bb890fe39/mingpt/model.py
# under MIT License ( https://github.com/karpathy/minGPT/blob/90420ee978fed95e6eb7c9add728d33bb890fe39/LICENSE )
# In this test, we are showing that we can replace this module..
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: CfgNode):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def _safe_str(target):
    if isinstance(target, FunctionType):
        return target.__name__
    else:
        return re.sub(r"at 0x[a-f0-9]+", "at [address]", str(target))


def create_graph_desc(gm: GraphModule):
    return "\n".join(
        [
            "    ".join(
                [
                    _safe_str(n.op),
                    _safe_str(n.name),
                    _safe_str(n.target),
                    _safe_str(n.args),
                    _safe_str(n.kwargs),
                ]
            )
            for n in gm.graph.nodes
        ]
    )


class TestSDPAPatternRewriter(TestCase):
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_1(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 4.0)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )
            expected = dot_prod_attention(*args)
            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                actual = efficient_dot_prod_attention(*args)

                torch.testing.assert_close(actual, expected)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_attn_weights_not_dead(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                attn_weights = (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                )
                return attn_weights.matmul(value), attn_weights

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    transpose    transpose    (arg1, -2, -1)    {}
call_function    matmul    <built-in method matmul of type object at [address]>    (arg0, transpose)    {}
call_method    div    div    (matmul, 4.0)    {}
call_method    softmax    softmax    (div,)    {'dim': -1}
call_method    matmul_1    matmul    (softmax, arg2)    {}
output    output    output    ([matmul_1, softmax],)    {}""",
            )
            expected = dot_prod_attention(*args)
            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                actual = efficient_dot_prod_attention(*args)

                torch.testing.assert_close(actual, expected)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_2(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .mul(1.0 / math.sqrt(key.shape[-1]))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': 0.25}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )

            expected = dot_prod_attention(*args)
            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                actual = efficient_dot_prod_attention(*args)

                torch.testing.assert_close(actual, expected)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_3(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return torch.nn.functional.dropout(
                    torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                    p=0.4,
                    training=True,
                    inplace=False,
                ).matmul(value)

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 3.0)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.4, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )

            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                with freeze_rng_state():
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)
                torch._inductor.config.scaled_dot_product_attention_fusion = False
                with freeze_rng_state():
                    inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    expected = inefficient_dot_prod_attention(*args)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag
            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_4(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return torch.nn.functional.dropout(
                    torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                    p=0.2,
                    training=True,
                    inplace=False,
                ).matmul(value)

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': None, 'dropout_p': 0.2, 'is_causal': False, 'scale_factor': 0.4}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )

            # Now check that the result is identical
            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                with freeze_rng_state():
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)
                torch._inductor.config.scaled_dot_product_attention_fusion = False
                with freeze_rng_state():
                    inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    expected = inefficient_dot_prod_attention(*args)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag
            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_5(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
                return (
                    torch.matmul(query, key.transpose(-2, -1))
                    .div(0.1)
                    .masked_fill(attn_mask, float("-inf"))
                    .softmax(dim=-1)
                    .matmul(value)
                )

            tensor_shape = (2, 4, 8, 16)
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((1, 1, 8, 8), device="cuda")
                > 0,  # Note the >0 to turn this into a binary mask
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
placeholder    arg3    arg3    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 0.1)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': arg3, 'dropout_p': 0.0, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )

            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                with freeze_rng_state():
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)
                torch._inductor.config.scaled_dot_product_attention_fusion = False
                with freeze_rng_state():
                    inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    expected = inefficient_dot_prod_attention(*args)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag
            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system"
    )
    def test_sdpa_rewriter_6(self):
        with sdp_kernel(enable_flash=False, enable_math=True):

            def dot_prod_attention(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor,
            ) -> torch.Tensor:
                """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
                assert query.dim() == 4
                assert key.dim() == 4
                assert value.dim() == 4
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
            batch_size, n_head, seq_len, embed_dim = tensor_shape
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((1, 1, 8, 8), device="cuda")
                > 0,  # Note the >0 to turn this into a binary mask
            ]
            # See whether the optimization matches our test pattern and is replaced
            gm: torch.fx.GraphModule = torch._dynamo.export(dot_prod_attention, *args)[
                0
            ]
            gm = sdpa_pattern_rewriter.fuse_scaled_dot_product_attention(gm)
            graph_desc = create_graph_desc(gm)
            self.assertExpectedInline(
                graph_desc,
                """\
placeholder    arg0    arg0    ()    {}
placeholder    arg1    arg1    ()    {}
placeholder    arg2    arg2    ()    {}
placeholder    arg3    arg3    ()    {}
call_method    contiguous    contiguous    (arg0,)    {}
call_method    contiguous_1    contiguous    (arg1,)    {}
call_method    contiguous_2    contiguous    (arg2,)    {}
call_function    truediv    <built-in function truediv>    (1.0, 0.1)    {}
call_function    _scale_factor_dot_product_attention    _scale_factor_dot_product_attention    (contiguous, contiguous_1, contiguous_2)    {'attn_mask': arg3, 'dropout_p': 0.3, 'is_causal': False, 'scale_factor': truediv}
output    output    output    ([_scale_factor_dot_product_attention],)    {}""",
            )

            saved_flag = torch._inductor.config.scaled_dot_product_attention_fusion
            torch._inductor.config.scaled_dot_product_attention_fusion = True
            try:
                with freeze_rng_state():
                    efficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    actual = efficient_dot_prod_attention(*args)
                torch._inductor.config.scaled_dot_product_attention_fusion = False
                with freeze_rng_state():
                    inefficient_dot_prod_attention = torch.compile(dot_prod_attention)
                    expected = inefficient_dot_prod_attention(*args)
            finally:
                torch._inductor.config.scaled_dot_product_attention_fusion = saved_flag
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
