# Copyright (c) Meta Platforms, Inc. and affiliates
# pyre-ignore-all-errors[6]

import math

from typing import Optional, Union

import torch
from torch.distributed._tensor import DTensor as DT
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.tensor.parallel._view_with_dim_change import (
    _view_with_sharding_dim_change,
)

__all__ = ["TensorParallelMultiheadAttention"]


# TODO: Add a test to test equivalence between our Multihead Attention
# with other mainstream ones (Megatron-LM or PyTorch).
def _stride_same_as_shard(
    tensor: torch.Tensor, tp_size: int, chunk_dim: int, cat_dim: int
) -> torch.Tensor:
    """
    Adjust local tensor's stride same as the sharded situation.
    So that view result will keeps the same.
    """
    if isinstance(tensor, DT):
        return tensor
    view_size = list(tensor.size())
    view_size[chunk_dim] //= tp_size
    return torch.cat(
        [t.view(*view_size) for t in tensor.chunk(tp_size, dim=chunk_dim)],
        dim=cat_dim,
    ).contiguous()


class TensorParallelMultiheadAttention(torch.nn.Module):
    """
    Multi-head Attention block from Transformer models.
    Since we need some customizations for the attention layer,
    we are writing a customized but mathematically equivalent
    attention module as defined in torch.nn.

    Note that:
    We now only support the case when it's self attention with
    limited input args and we also assume that the input tensor
    has a dimension of three. Although we do implement the logic
    for multihead attention, it was not fully tested.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        self_attention: bool = True,
    ) -> None:
        super().__init__()
        self.device: torch.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.num_heads = num_heads
        self.hidden_size = embed_dim
        self.hidden_size_per_attention_head: int = self.hidden_size // num_heads
        self.scale: float = self.hidden_size_per_attention_head**-0.5
        if self_attention:
            self.qkv: torch.nn.Module = torch.nn.Linear(
                embed_dim, embed_dim * 3, bias=add_bias_kv, device=self.device
            )
            torch.nn.init.xavier_uniform_(self.qkv.weight)
            if add_bias_kv:
                torch.nn.init.zeros_(self.qkv.bias)
        else:
            self.query: torch.nn.Module = torch.nn.Linear(
                embed_dim, embed_dim, bias=add_bias_kv, device=self.device
            )
            self.key: torch.nn.Module = torch.nn.Linear(
                embed_dim, embed_dim, bias=add_bias_kv, device=self.device
            )
            self.value: torch.nn.Module = torch.nn.Linear(
                embed_dim, embed_dim, bias=add_bias_kv, device=self.device
            )
            torch.nn.init.xavier_uniform_(self.query.weight)
            torch.nn.init.xavier_uniform_(self.key.weight)
            torch.nn.init.xavier_uniform_(self.value.weight)
            if add_bias_kv:
                torch.nn.init.zeros_(self.query.bias)
                torch.nn.init.zeros_(self.key.bias)
                torch.nn.init.zeros_(self.value.bias)
        self.proj: torch.nn.Module = torch.nn.Linear(
            embed_dim, embed_dim, bias=bias, device=self.device
        )
        torch.nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        if bias:
            torch.nn.init.zeros_(self.proj.bias)
        self.tp_size = tp_size
        self.hidden_size = embed_dim
        self.norm_factor: float = math.sqrt(self.hidden_size_per_attention_head)
        self.self_attention = self_attention

    def forward(
        self,
        query: Union[torch.Tensor, DT],
        key: Union[torch.Tensor, DT],
        value: Union[torch.Tensor, DT],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Union[torch.Tensor, DT]:
        b, sq, h = query.shape
        sk = key.size(1)
        nh = self.num_heads
        hn = self.hidden_size_per_attention_head

        # x: [b, sq/sk/sv, h]
        # ===================
        # Permute. [sq/sk/sv, b, h]
        # ===================
        if not self.self_attention:
            # =====================
            # Query, Key, and Value
            # =====================
            query = query.permute(1, 0, 2).contiguous()
            key = key.permute(1, 0, 2).contiguous()
            value = value.permute(1, 0, 2).contiguous()

            # Attention heads [sq/sk/sv, b, h] --> [sq/sk/sv * b, (nh * hn)]
            query = query.view(-1, h)
            key = key.view(-1, h)
            value = value.view(-1, h)

            query_layer = _view_with_sharding_dim_change(
                self.query(query), 1, (sq, b * nh, hn)
            )
            key_layer = _view_with_sharding_dim_change(
                self.key(key), 1, (sk, b * nh, hn)
            )
            value_layer = _view_with_sharding_dim_change(
                self.value(value), 1, (sk, b * nh, hn)
            )
        else:
            assert torch.equal(query, key) and torch.equal(
                query, value
            ), "inputs are different for self-attention."
            # =====================
            # Query
            # =====================
            query = query.permute(1, 0, 2).contiguous()

            # Attention heads [sq, b, h] --> [sq * b, (nh * 3 * hn)]
            query = query.view(-1, h)
            mixed_x_layer = self.qkv(query)

            # [sq * b, 3 * h] --> [sq, b, nh, 3 * hn]
            mixed_x_layer = _view_with_sharding_dim_change(
                mixed_x_layer, 2, (sq, b, nh, 3 * hn)
            )

            # [sq, b, nh, 3 * hn] --> 3 [sq, b, nh, hn]
            last_dim = mixed_x_layer.dim() - 1
            last_dim_size = mixed_x_layer.size(last_dim) // 3
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                last_dim_size, dim=last_dim
            )

            query_layer = _stride_same_as_shard(query_layer, self.tp_size, 2, 1)
            key_layer = _stride_same_as_shard(key_layer, self.tp_size, 2, 1)
            value_layer = _stride_same_as_shard(value_layer, self.tp_size, 2, 1)
            # [sq, b, nh, hn] -> [sq, b * nh, hn]
            query_layer = _view_with_sharding_dim_change(
                query_layer, 1, (sq, b * nh, -1)
            )
            key_layer = _view_with_sharding_dim_change(key_layer, 1, (sq, b * nh, -1))
            value_layer = _view_with_sharding_dim_change(
                value_layer, 1, (sq, b * nh, -1)
            )

        # ===================================
        # Raw attention scores. [b, nh, s, s]
        # ===================================

        factor = self.tp_size if isinstance(query_layer, DT) else 1
        # preallocating result tensor: [b * nh, sq, sk]
        matmul_result = torch.empty(
            b * nh // factor,
            sq,
            sk,
            dtype=query_layer.dtype,
            device=self.device,
        )
        if isinstance(query_layer, DT):
            matmul_result = DT.from_local(
                matmul_result,
                query_layer.device_mesh,
                [Shard(0)],
                run_check=False,
            )

        # Raw attention scores. [b * nh, sq, sk]
        attn = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * nh, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * nh, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # ===============
        # Attention probs
        # ===============
        attn = attn.softmax(dim=-1)

        # =========================
        # Context layer. [sq * b, hidden]
        # =========================

        # bmm: [b * nh, sq, hn]
        context_layer = torch.bmm(attn, value_layer.transpose(0, 1))

        # change view [nh, b, sq, hn]
        context_layer = context_layer.view(nh, b, sq, hn)

        # [nh, b, sq, hn] --> [sq, b, nh, hn]
        context_layer = context_layer.permute(2, 1, 0, 3).contiguous()

        # [sq, b, nh, hn] --> [sq * b, hidden]
        context_layer = _view_with_sharding_dim_change(
            context_layer.contiguous(), 1, (-1, self.hidden_size)
        )

        # =================
        # Projection. [sq, b, h]
        # =================
        output = self.proj(context_layer).view(sq, b, h)

        # ===================
        # Permute. [b, sq, h]
        # ===================
        output = output.permute(1, 0, 2)

        return output

    def copy(self, that: torch.nn.MultiheadAttention) -> None:
        # TODO: current implementation assume `self` is a self attention module
        assert (
            self.hidden_size == that.embed_dim
        ), "embed_dim must be equal in TensorParallelMultiheadAttention.copy()!"

        if that.in_proj_weight is not None:
            self.qkv.register_parameter("weight", that.in_proj_weight)
        if that.in_proj_bias is not None:
            self.qkv.register_parameter("bias", that.in_proj_bias)
        if that.out_proj.weight is not None:
            # TODO: The use of Parameter is to avoid `mypy` issue caused
            # by the `tensor` type annotation on Linear.weight to which
            # a Parameter object is actually assigned
            self.proj.register_parameter(
                "weight", torch.nn.Parameter(that.out_proj.weight)
            )
        if that.out_proj.bias is not None:
            self.proj.register_parameter("bias", that.out_proj.bias)
