import math
import torch
import torch.nn as nn

from einops import rearrange

import hydra

from flash_attn.flash_blocksparse_attn_interface import flash_blocksparse_attn_func
from flash_attn.flash_blocksparse_attn_interface import convert_blockmask
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


class FlashBlocksparseAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, sparsity_config, softmax_temp=None, attention_dropout=0.0,
                 max_seq_length=2048, device=None, dtype=None):
        super().__init__()
        self.sparsity_config = hydra.utils.instantiate(sparsity_config)
        self.softmax_temp = softmax_temp
        self.dropout_p = attention_dropout

        # initialize sparse layout and register as buffer
        max_seq_length = ((max_seq_length + 256 - 1) // 256) * 256
        layout = self.sparsity_config.make_layout(max_seq_length)
        self.register_buffer("layout", layout)
        blockmask_converted = convert_blockmask(self.layout, causal=False)
        self.register_buffer("blockmask_converted", blockmask_converted)
        # logger.info(f'Attention class {self.__class__}: saving={self.layout.float().mean()}')

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False, convert_mask=True):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        assert not need_weights
        assert attn_mask is None
        assert qkv.dtype == torch.float16
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            # Convert mask to take a subset
            seqlen_rounded = ((seqlen + 256 - 1) // 256) * 256
            assert seqlen_rounded // 16 <= self.layout.shape[0], seqlen_rounded // 256 <= self.layout.shape[1]
            blockmask = self.layout[:seqlen_rounded // 16, :seqlen_rounded // 256]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                output = flash_blocksparse_attn_func(
                    qkv, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0,
                    max_s, softmax_scale=self.softmax_temp, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                key_padding_mask_bool = key_padding_mask.bool_matrix
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask_bool)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_blocksparse_attn_func(
                    x_unpad, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0,
                    max_s, softmax_scale=self.softmax_temp, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            seqlen = max_s
            # Convert mask to take a subset
            seqlen_rounded = ((seqlen + 256 - 1) // 256) * 256
            assert seqlen_rounded // 16 <= self.layout.shape[0], seqlen_rounded // 256 <= self.layout.shape[1]
            blockmask = self.layout[:seqlen_rounded // 16, :seqlen_rounded // 256]
            if convert_mask:
                output = flash_blocksparse_attn_func(
                    qkv, cu_seqlens, blockmask, self.dropout_p if self.training else 0.0,
                    max_s, softmax_scale=self.softmax_temp, causal=causal
                )
            else:
                output = flash_blocksparse_attn_func(
                    qkv, cu_seqlens, self.blockmask_converted, self.dropout_p if self.training else 0.0,
                    max_s, softmax_scale=self.softmax_temp, causal=causal,
                    convert_mask=False,
                )

        return output, None


class FlashBlocksparseMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, sparsity_config, bias=True, batch_first=True,
                 attention_dropout=0.0, causal=False, max_seq_length=2048,
                 device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashBlocksparseAttention(
            sparsity_config, attention_dropout=attention_dropout,
            max_seq_length=max_seq_length, **factory_kwargs
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, x_ignored_, x_ignored_1_, attn_mask=None, key_padding_mask=None,
                need_weights=False):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
